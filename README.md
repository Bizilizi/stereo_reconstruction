

##Setting up this project: 
Required packages: 
- Eigen v3.4
- OpenCV 
- ...

1. Download Eigen v3.4 from https://gitlab.com/libeigen/eigen/-/releases/3.4-rc1 and place Eigen folder (folder nme is "eigen-3.4") under libs/
2. Download and install opencv (Check https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html for full documentation)
<pre> <code> git clone https://github.com/opencv/opencv.git
  git -C opencv checkout master
  mkdir build && cd build
  cmake .. 
  make 
  sudo make install
</code></pre>

