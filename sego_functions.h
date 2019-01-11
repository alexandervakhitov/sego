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


#ifndef SEGO_FUNCTIONS_H
#define SEGO_FUNCTIONS_H

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

void Quat2Rot(const Eigen::VectorXd &q, Eigen::Matrix3d *R);

void GenerateCrossProductMatrix(const Eigen::Vector3d& v, Eigen::Matrix3d* M_p);

void RowReducePivot(Eigen::Matrix<double,6,13>* A);

Eigen::Matrix<double, 10, 10> BuildQ();

Eigen::Vector3d FromVec3d(const cv::Vec3d& l_cv);

bool TFromRUsingTrifocalLines(const cv::Mat& projs, const cv::Mat& lprojs, const cv::Mat& vis_p, const cv::Mat& vis_l,
                              const Eigen::Matrix3d& R, Eigen::Vector3d* T);

bool TriangulateLine(const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1, const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                     const Eigen::Vector3d& leq1, const Eigen::Vector3d& leq2, Eigen::Vector3d* X1, Eigen::Vector3d* X2);

bool EpiSEgo(const cv::Mat &projs, const cv::Mat &lprojs,
             const cv::Mat &vis_p, const cv::Mat &vis_l, bool id_det_check,
             std::vector<Eigen::Matrix3d> *Rs, std::vector<Eigen::Vector3d> *ts);

void pluecker_assign(double (*M1)[10], double (*M2)[10], double (*Mc)[120]);

bool SolveWith3Quadric(const cv::Mat &projs, const cv::Mat &lprojs, const cv::Mat &vis_p, const cv::Mat &vis_l,
                       bool is_det_check, std::vector<Eigen::Matrix3d> *Rs, std::vector<Eigen::Vector3d> *ts);

bool GenerateEquationsEpipolar(const cv::Mat &projs, const cv::Mat &lprojs, const cv::Mat &vis_p, const cv::Mat &vis_l,
                               Eigen::Matrix<double, 4, 18> *A, Eigen::Vector3d *pt_shift);

#endif //SEGO_FUNCTIONS_H
