# 3D-playground
Let's face it: I know how I develop code, and the first repo I make to work on 3D detection/tracking is going to be messy and full of extraneous files, so I'm embracing that. Once I have a working MVP I'll migrate to a new repository.

## Guidelines

## Key Commands
- `shift +` fills the frame buffer. This is the first step when starting the labeler, and generally takes 20 minutes or so, so do this before you get your morning coffee etc.
- `8` and `9` - advance or un-advance all cameras by one frame
- `-` and `+` - advance or un-advance all cameras by a larger step (10 - 20 frames)
- `[` and `]` - cycle through camera views
- `q` - save and quit. Don't press this after you've buffered frames, or you'll have to re-buffer. Best to just leave the labeler running
- `w` - save and don't quit.
- `u` - undo the last labeling change. Only can undo one change at a time so be careful
- `a` - click a vehicle and then press enter to create a new box (with a new unique ID)
- `r` - click a vehicle and then press enter to delete that vehicle in all frames (BE CAREFUL AND PROBABLY DON'T USE THIS)
- `s` - Click on a box and drag it either along the lanes or perpendicular to the lanes to shift its position in the x/y direction
- `d` - Click on a box and drag it in the length/width/height direction to change the vehicle's dimension. Note this change is reflected across all boxes for that object. To adjust height, right click until an "R" appears on the top header, and likewise right click until it disappears to move back to adjusting length/width.
- `c` - The first time you click a box, copies that box (I try and always click on the back bottom left of the vehicle). Each subsequent time you click on a vehicle, the box is pasted at the clicked location. Used to copy a box for a vehicle from one frame to others.
- `v` - Click on a vehicle, type the vehicle class, and press enter, to change that vehicle's class in all frames
- `h` - Click and drag as if adjusting height to shift the homography height for the camera (this adjusts how tall a fixed height (say 1 foot) looks within the frame. Each camera should be roughly correct but can be tweaked as needed so that most vehicle boxes that have reasonable heights (in feet) have boxes that appear to match their height well
