

##Setting up this project
Required packages: 
- Eigen v3.4
- OpenCV 
- Ceres
- SDK provided by the dataset

The dataset itself is already included in the repository and does not have to be downloaded separately. 

1. Download Eigen v3.4 from https://gitlab.com/libeigen/eigen/-/releases/3.4-rc1 and place Eigen folder (the folder name should be "eigen-3.4") under libs/
2. Download and install opencv (Check https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html for full documentation)
<pre> <code> git clone https://github.com/opencv/opencv.git
  git -C opencv checkout master
  mkdir build && cd build
  cmake .. 
  make 
  sudo make install
</code></pre>

OpenCV uses dynamic linking which means that some functions that depend on third party libraries might only fail during runtime. A quick hack if some kind of error occurs is to simply install all required dependencies using:
<pre><code> sudo apt-get install libopencv-* </code></pre>
After that, opencv has to be rebuild and reinstalled. 

3. Ceres - TBD

4. SDK - TDB 
Can be downloaded here: https://vision.middlebury.edu/stereo/submit3/

### Data visualization 
OpenCV does not support PFM files, so we can use CVKIT to display the disparity images. 
You can download it here: https://github.com/roboception/cvkit and make sure to follow the install instructions including third-party software. 

1. Displaying disparity image in greyscale: 
<pre><code> sv image.png </code></pre>

2. Displaying disparity image as 3D model: 
<pre><code> plyv image_disp.pfm </code></pre>

TODO 