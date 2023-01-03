# Fish Movement Tracking with GMM and Kalman Filter

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