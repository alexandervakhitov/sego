/**
* This file is a part of SEGO.
*
* (C) 2018 Alexander Vakhitov <alexander.vakhitov at gmail dot com>
* For more information see <https://github.com/alexander-vakhitov/sego>
*
* SEGO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SEGO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with SEGO. If not, see <http://www.gnu.org/licenses/>.
*
*/


#ifndef SM_SEGO_PLUECKER_H
#define SM_SEGO_PLUECKER_H
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

//a method to find relative pose between two stereo cameras
//a stereo camera is rectified and has baseline [1,0,0] (that is, first view of the camera is the right one).
//feature coordinates are normalized (i.e., multiplied by (K)^(-1) for a camera internal calibration matrix K)
// If the baseline is [-1,0,0], please set is_right_left = true, otherwise false
//we enumerate views as 0,1 - first and second views of the first camera,
//2,3 - first an second views of the second camera
//projs: rows - no. of points, columns - 4 (no. of views), type - Vec2d (CV64FC2)
//lprojs: rows - no. of lines, columns - 4 (no. of views), type - Vec3d (CV64FC3)
//vis_p: rows - no. of points, columns - 4 (no. of views), type - uchar (CV_8UC1),
// vis_p.at<uchar>(i,j) == 0 : point i is invisible at the view j, 1 - visible
//vis_l: rows - no. of lines, columns - 4 (no. of views), type - uchar (CV_8UC1),
// vis_l.at<uchar>(i,j) == 0 : line i is invisible at the view j, 1 - visible
//is_det_check = true: use our modification to the quadric solver,
// false: use original 3 quadric solver (my implementation), from Kukelova et al., 2016.
//Rs, ts: vectors of solutions. Rs[i], ts[i] is a valid relative pose of a stereo
// camera 2 (views 2,3) with respect to a stereo camera 1 (views 0,1)
bool sego_solver(const cv::Mat& projs, const cv::Mat& lprojs,
                 const cv::Mat& vis_p, const cv::Mat& vis_l, bool is_det_check, bool is_right_left,
                 std::vector<Eigen::Matrix3d>* Rs, std::vector<Eigen::Vector3d>* ts);


#endif //SM_SEGO_PLUECKER_H
