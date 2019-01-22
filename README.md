#H1 Stereo egomotion (SEGO) library

Contains a method to find SE3 transform between two poses of a rectified stereo rig. For a method description, please see the paper
 "Stereo relative pose from line and point feature triplets" by A. Vakhitov, V. Lempitsky and Y. Zheng, ECCV 2018

The code compiles to a library. You can see an example of its usage in main.cpp.

Prerequisities:

*OpenCV 3.0 and higher
*Eigen 3.1 and higher

To download & build:

git clone https://github.com/alexander-vakhitov/sego.git
cd sego
cmake .
make

