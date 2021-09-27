import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                continue

            ##### here, we need a custom bit to calculate 2D bbox based on 3D bbox
            #### actually, instead we now expect a 2D box in  last 4 positions, and 3D corners in last 16
            #bbox_annotation_2D = bbox_annotation[:,-4:]
            
            xmin,_ = torch.min(bbox_annotation[:,[0,2,4,6]],dim = 1)
            xmax,_ = torch.max(bbox_annotation[:,[0,2,4,6]],dim = 1)
            ymin,_ = torch.min(bbox_annotation[:,[1,3,5,7]],dim = 1)
            ymax,_ = torch.max(bbox_annotation[:,[1,3,5,7]],dim = 1)
            
            xmin2,_ = torch.min(bbox_annotation[:,[8,10,12,14]],dim = 1)
            xmax2,_ = torch.max(bbox_annotation[:,[8,10,12,14]],dim = 1)
            ymin2,_ = torch.min(bbox_annotation[:,[9,11,13,15]],dim = 1)
            ymax2,_ = torch.max(bbox_annotation[:,[9,11,13,15]],dim = 1)
            
            xmin = torch.min(xmin,xmin2).unsqueeze(1)
            xmax = torch.max(xmax,xmax2).unsqueeze(1)
            ymin = torch.min(ymin,ymin2).unsqueeze(1)
            ymax = torch.max(ymax,ymax2).unsqueeze(1)
            bbox_annotation_2D = torch.cat((xmin,ymin,xmax,ymax),dim = 1)
            
            IoU = calc_iou(anchors[0, :, :], bbox_annotation_2D) # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0
            

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]


                ### HERE, we'll need to redefine this logic for 3D bbox formulation
                # normalize coordinates by width and height
                # normalize tail length by dividing then taking log
                
                # gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # # clip widths to 1
                # gt_widths  = torch.clamp(gt_widths, min=1)
                # gt_heights = torch.clamp(gt_heights, min=1)

                # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                # targets_dw = torch.log(gt_widths / anchor_widths_pi)
                # targets_dh = torch.log(gt_heights / anchor_heights_pi)
                
                # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # targets = targets.t()
                targets = assigned_annotations[:,:-1] # remove classifications from targets
                
                # regression is x,y,wx,wy,lx,ly,hx,hy - need to convert to corner coordinates
                # fbl fbr bbl bbr ftl ftr btl btr - alternate w first, then l, then h
            
                regression = regression[positive_indices,:]
                
                # try to introduce bias so all directions are equally possible anglewise /???
                #regression[:,2:] -= 0.5             
                
                preds = torch.zeros([regression.shape[0],20],requires_grad = True).cuda()
                preds[:,0] = regression[:,0] - regression[:,2] - regression[:,4] - regression[:,6]
                preds[:,1] = regression[:,1] - regression[:,3] - regression[:,5] - regression[:,7]
                preds[:,2] = regression[:,0] + regression[:,2] - regression[:,4] - regression[:,6]
                preds[:,3] = regression[:,1] + regression[:,3] - regression[:,5] - regression[:,7]
                preds[:,4] = regression[:,0] - regression[:,2] + regression[:,4] - regression[:,6]
                preds[:,5] = regression[:,1] - regression[:,3] + regression[:,5] - regression[:,7]
                preds[:,6] = regression[:,0] + regression[:,2] + regression[:,4] - regression[:,6]
                preds[:,7] = regression[:,1] + regression[:,3] + regression[:,5] - regression[:,7]
                
                preds[:,8]  = regression[:,0] - regression[:,2] - regression[:,4] + regression[:,6]
                preds[:,9]  = regression[:,1] - regression[:,3] - regression[:,5] + regression[:,7]
                preds[:,10] = regression[:,0] + regression[:,2] - regression[:,4] + regression[:,6]
                preds[:,11] = regression[:,1] + regression[:,3] - regression[:,5] + regression[:,7]
                preds[:,12] = regression[:,0] - regression[:,2] + regression[:,4] + regression[:,6]
                preds[:,13] = regression[:,1] - regression[:,3] + regression[:,5] + regression[:,7]
                preds[:,14] = regression[:,0] + regression[:,2] + regression[:,4] + regression[:,6]
                preds[:,15] = regression[:,1] + regression[:,3] + regression[:,5] + regression[:,7]
                preds[:,16:20] = regression[:,8:12]
                
                targets[:,[0,2,4,6,8,10,12,14,16,18]] = (targets[:,[0,2,4,6,8,10,12,14,16,18]] - anchor_ctr_x_pi.unsqueeze(1).repeat(1,10)) / anchor_widths_pi.unsqueeze(1).repeat(1,10)
                targets[:,[1,3,5,7,9,11,13,15,17,19]] = (targets[:,[1,3,5,7,9,11,13,15,17,19]] - anchor_ctr_y_pi.unsqueeze(1).repeat(1,10)) / anchor_heights_pi.unsqueeze(1).repeat(1,10)
                

                # std_dev
                #targets = targets/(0.1*torch.ones([10]).cuda())
               

                negative_indices = 1 + (~positive_indices)

                #regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_diff = torch.abs(targets - preds)

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    
        classification_losses = [item.cuda() for item in classification_losses]
        regression_losses = [item.cuda() for item in regression_losses]
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
