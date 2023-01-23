# bad_apple_net
Neural network that remembers "Bad apple!!" frames in weights

To train model:
* put video to *source.mp4*
* run *video2frames&audio*
* run *prepare.py*
* train in train.ipynb (~550 epochs is ok)

Or [download model](https://drive.google.com/file/d/1UeVGBrrnQKbWCY0L7lUXbQjQfhi9ROpd/view?usp=share_link) (you should put in in *models* folder)

To create video (with ready model):
* (if not yet) put video to *source.mp4*
* (if not yet) run *video2frames&audio*
* run *gen_frames.py*
* run *back2video*
