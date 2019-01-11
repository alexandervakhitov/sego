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

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include <opencv/cxeigen.hpp>
#include "sego.h"
#include "sego_functions.h"
void GenerateVectorizedRotation(bool is_direct, const Eigen::Vector3d& X, Eigen::Matrix<double, 3, 9>* C_p);
void GetRxMat(const Eigen::Vector3d &Xm, const Eigen::Vector3d &p3, bool is_direct, Eigen::Matrix<double, 3, 9> *A);

void AnalyzeProjection(const cv::Mat &vis, int ind, const cv::Mat &projs,
                       bool *is_direct, bool *is_stereoshift, Eigen::Vector3d *p1,
                       Eigen::Vector3d *p2, Eigen::Vector3d *third_proj)
{
//    std::cout << " check case " << std::endl;
    for (int i = 0; i < 4; i++)
    {
//        std::cout << (int)vis.at<uchar>(ind, i) << " " ;
    }
//    std::cout << std::endl;
    *is_direct = false;
    if ((int)vis.at<uchar>(ind, 0) + (int)vis.at<uchar>(ind, 1) == 2)
    {
        *is_direct = true;
    }
    *is_stereoshift = false;
    if ((*is_direct && (int)vis.at<uchar>(ind, 3) == 1) ||
            (!(*is_direct) && (int)vis.at<uchar>(ind, 1) == 1))
    {
        *is_stereoshift = true;
    }
    int c1, c2, c3;
    if (*is_direct) {
        c1 = 0;
        c2 = 1;
        if (*is_stereoshift) {
            c3 = 3;
        } else {
            c3 = 2;
        }
    } else {
        c1 = 2;
        c2 = 3;
        if (*is_stereoshift) {
            c3 = 1;
        } else {
            c3 = 0;
        }
    }

    if (projs.type() == CV_64FC2)
    {
        *p1 << projs.at<cv::Vec2d>(ind, c1)[0], projs.at<cv::Vec2d>(ind, c1)[1], 1.0;
        *p2 << projs.at<cv::Vec2d>(ind, c2)[0], projs.at<cv::Vec2d>(ind, c2)[1], 1.0;
        *third_proj << projs.at<cv::Vec2d>(ind, c3)[0], projs.at<cv::Vec2d>(ind, c3)[1], 1.0;
    } else {
        *p1 << projs.at<cv::Vec3d>(ind, c1)[0], projs.at<cv::Vec3d>(ind, c1)[1], projs.at<cv::Vec3d>(ind, c1)[2];
        *p2 << projs.at<cv::Vec3d>(ind, c2)[0], projs.at<cv::Vec3d>(ind, c2)[1], projs.at<cv::Vec3d>(ind, c2)[2];
        *third_proj << projs.at<cv::Vec3d>(ind, c3)[0], projs.at<cv::Vec3d>(ind, c3)[1], projs.at<cv::Vec3d>(ind, c3)[2];
    }
}

void GetURxPoint(const Eigen::Vector3d &p1, bool is_direct, const Eigen::Vector3d &p3,
                 const Eigen::Vector3d &u, Eigen::Matrix<double, 1, 9> *c_row)
{
    Eigen::Matrix<double, 3, 9> C;
    GetRxMat(p1, u, is_direct, &C);
    *c_row = p3.transpose() * C;
}

void GetRUx(const Eigen::Vector3d &X, bool is_direct, const Eigen::Vector3d &p3,
            const Eigen::Vector3d &pt_shift, Eigen::Matrix<double, 1, 9> *c_row)
{
    Eigen::Vector3d X_cross = X.cross(pt_shift);
    Eigen::Matrix<double, 3, 9> C;
    GenerateVectorizedRotation(is_direct, X_cross, &C);
    *c_row = p3.transpose()  * C;
}

void GeneratePointEpipolarEquations(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3d &p3,
                                    const Eigen::Vector3d &pt_shift, const Eigen::Vector3d &u, bool is_direct,
                                    bool is_stereoshift,
                                    Eigen::Matrix<double, 2, 18> *A)
{
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    Eigen::Vector3d p3_normed = p3/p3.norm();

    Eigen::Matrix<double, 1, 9> car1_row, car2_row, cr1_row, cr2_row;
    if (is_direct)
    {
        GetURxPoint(p1, is_direct, p3_normed, u, &car1_row);
        GetURxPoint(p2, is_direct, p3_normed, u, &car2_row);
        GetRUx(p1, is_direct, p3_normed, pt_shift, &cr1_row);
        GetRUx(p2, is_direct, p3_normed, pt_shift + t2, &cr2_row);
    } else {
        GetRUx(p1, is_direct, p3_normed, u, &car1_row);
        GetRUx(p2, is_direct, p3_normed, u, &car2_row);
        GetURxPoint(p1, is_direct, p3_normed, pt_shift, &cr1_row);
        GetURxPoint(p2, is_direct, p3_normed, pt_shift, &cr2_row);
        Eigen::Matrix<double, 1, 9> cr2_add_row;
        GetRUx(p2, is_direct, p3_normed, t2, &cr2_add_row);
        cr2_row = cr2_row + cr2_add_row;
    }
    if (is_stereoshift)
    {
        Eigen::Matrix<double, 1, 9> css1_row;
        GetURxPoint(p1, is_direct, p3_normed, t2, &css1_row);
        Eigen::Matrix<double, 1, 9> css2_row;
        GetURxPoint(p2, is_direct, p3_normed, t2, &css2_row);
        cr1_row = cr1_row + css1_row;
        cr2_row = cr2_row + css2_row;
    }
    A->block<1,9>(0,0) = cr1_row;
    A->block<1,9>(0,9) = car1_row;
    A->block<1,9>(1,0) = cr2_row;
    A->block<1,9>(1,9) = car2_row;
}

inline void GenerateVectorizedRotation(bool is_direct, const Eigen::Vector3d& X, Eigen::Matrix<double, 3, 9>* C_p)
{
    Eigen::Matrix3i inds;
    inds << 0, 1, 2,
            3, 4, 5,
            6, 7, 8;
    if (!is_direct)
    {
        inds = Eigen::Matrix3i(inds.transpose());
    }
    C_p->setZero();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (*C_p)(i, inds(j, i)) = X(j);
        }
    }
}

//Generate matrix A such that Ar = [p3]_x R Xm, where r is a vectorized form of R
void GetRxMat(const Eigen::Vector3d &Xm, const Eigen::Vector3d &p3, bool is_direct,
              Eigen::Matrix<double, 3, 9> *A)
{
    Eigen::Matrix<double, 3, 9> C;
    GenerateVectorizedRotation(is_direct, Xm, &C);
    Eigen::Matrix3d p3_x;
    GenerateCrossProductMatrix(p3, &p3_x);
    *A = p3_x * C;
}

void GetURxMat(const Eigen::Vector3d &p1, bool is_direct, const Eigen::Matrix3d &p3x,
               const Eigen::Vector3d &u, Eigen::Matrix<double, 3, 9> *c_row)
{
    Eigen::Matrix<double, 3, 9> C;
    GenerateVectorizedRotation(is_direct, p1, &C);
    Eigen::Matrix3d ux;
    GenerateCrossProductMatrix(u, &ux);
    *c_row = p3x.transpose() * ux * C;
}


bool GenerateLineEpipolarEquations(const Eigen::Vector3d &l1, const Eigen::Vector3d &l2, const Eigen::Vector3d &l3,
                                   const Eigen::Vector3d pt_shift, const Eigen::Vector3d &u, bool is_direct,
                                   bool is_stereoshift,
                                   Eigen::Matrix<double, 2, 18> *A)
{
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    Eigen::Vector3d X1, X2;
    if (!TriangulateLine(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), t2, l1, l2, &X1, &X2))
    {
        return false;
    }
    Eigen::Vector3d Xcp = X1.cross(X2);
    Eigen::Vector3d dX = X2-X1;
    Eigen::Matrix<double, 3, 9> r_eqs;
    GetRxMat(Xcp, l3, is_direct, &r_eqs);
    Eigen::Matrix3d l3xt;
    GenerateCrossProductMatrix(l3, &l3xt);
    l3xt = Eigen::Matrix3d(l3xt.transpose());
    Eigen::Matrix<double, 3, 9> ar_eqs;
    Eigen::Matrix<double, 3, 9> r_eqs_add;
    if (is_direct)
    {
        GetURxMat(dX, is_direct, l3xt, u, &ar_eqs);
        Eigen::Vector3d Xm = -pt_shift.cross(dX);
        GetRxMat(Xm, l3, is_direct, &r_eqs_add);
    } else {
        Eigen::Vector3d nDx = -u.cross(dX);
        GetRxMat(nDx, l3, is_direct, &ar_eqs);
        GetURxMat(dX, is_direct, l3xt, pt_shift, &r_eqs_add);
    }
    if (is_stereoshift)
    {
        Eigen::Matrix<double, 3, 9> r_eqs_stereo_add;
        GetURxMat(dX, is_direct, l3xt, t2, &r_eqs_stereo_add);
        r_eqs = r_eqs + r_eqs_stereo_add;
    }
    Eigen::Matrix<double, 3, 18> A_tmp;
    A_tmp.block<3,9>(0,0) = r_eqs + r_eqs_add;
    A_tmp.block<3,9>(0,9) = ar_eqs;
    *A = Eigen::Matrix<double,2,18>(A_tmp.block<2,18>(0,0));
    return true;
}


bool GenerateEquationsEpipolar(const cv::Mat &projs, const cv::Mat &lprojs, const cv::Mat &vis_p, const cv::Mat &vis_l,
                               Eigen::Matrix<double, 4, 18> *A, Eigen::Vector3d *pt_shift)
{
    Eigen::Vector3d t2;
    t2 << 1,0,0;
    cv::Mat t2_cv;
    cv::eigen2cv(t2, t2_cv);

    cv::Point2d pt1(projs.at<cv::Vec2d>(0, 0)[0], projs.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2d pt2(projs.at<cv::Vec2d>(0, 1)[0], projs.at<cv::Vec2d>(0, 1)[1]);
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64FC1);
    cv::Mat eye_mat = cv::Mat::eye(3, 3, CV_64FC1);
    eye_mat.copyTo(P0(cv::Rect(0, 0, 3, 3)));
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64FC1);
    eye_mat.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    t2_cv.copyTo(P1(cv::Rect(3, 0, 1, 3)));
    cv::Mat pts_3d;
    cv::triangulatePoints(P0, P1, std::vector<cv::Point2d>{pt1}, std::vector<cv::Point2d>{pt2}, pts_3d);
    *pt_shift <<  pts_3d.at<double>(0, 0), pts_3d.at<double>(1, 0), pts_3d.at<double>(2,0);
    *pt_shift = *pt_shift / pts_3d.at<double>(3, 0);

    Eigen::Vector3d u ;
    u << projs.at<cv::Vec2d>(0, 2)[0],projs.at<cv::Vec2d>(0, 2)[1],1.0;

    int a_ind = 0;

    for (int pi = 1; pi < vis_p.rows; pi++)
    {
        bool is_direct, is_stereoshift;
        Eigen::Vector3d p1, p2, p3;
        AnalyzeProjection(vis_p, pi, projs, &is_direct, &is_stereoshift, &p1, &p2, &p3);
        Eigen::Matrix<double, 2, 18> A_curr;
        GeneratePointEpipolarEquations(p1, p2, p3, *pt_shift, u, is_direct, is_stereoshift, &A_curr);
        A->block<2,18>(2*a_ind, 0) = A_curr;
        a_ind += 1;
    }

    for (int li = 0; li < vis_l.rows; li++)
    {
        bool is_direct, is_stereoshift;
        Eigen::Vector3d l1, l2, l3;
        AnalyzeProjection(vis_l, li, lprojs, &is_direct, &is_stereoshift, &l1, &l2, &l3);
        Eigen::Matrix<double, 2, 18> A_ln;
        if (!GenerateLineEpipolarEquations(l1, l2, l3, *pt_shift, u, is_direct, is_stereoshift, &A_ln))
        {
            return false;
        }
        A->block<2,18>(2*a_ind, 0) = A_ln;
        a_ind += 1;
    }
    return true;
}