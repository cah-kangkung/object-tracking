# Fish Movement Tracking with GMM and Kalman Filter

The fish movement tracking system is a command-line based system that is created with the python programming language. This system aims to track fish and calculate the average number of fish in a video. To run the system, users need to install python version 3 (three) first, as well as prepare a dataset in the form of a video that you want to track the objects inside.

## Command args
 * `-p` = Video path
 * `-n` = Video name
 * `-kes` = Kernel Erotion Size
 * `-ked` = Kernel Dilation Size
 * `-ss` = Screenshoots array (which frame would be screenshoted)

 ## Command Example
```
python main.py -p datasets/9908-compressed.mp4 -n 9908 -kes 3 -kds 9 -ss 120,230,290
```