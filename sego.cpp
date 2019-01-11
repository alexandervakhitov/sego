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


#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <fstream>
#include <iostream>
#include <opencv/cxmisc.h>
#include "sego.h"
#include "sego_functions.h"

using namespace Eigen;
using namespace cv;

extern "C"
{
// LU factorization of a general matrix
void dgetrf_(
        const int &M,
        const int &N,
        double *A,
        const int &LDA,
        int *IPIV,
        int &INFO
) __attribute__((stdcall));
}

void SwapViews(Mat* m);

void SwapCameras(Mat *projs)
{
    if (projs->rows == 0)
    {
        return;
    }
    Mat projs_2_cams;
    (*projs)(Rect(0,0,2,projs->rows)).copyTo(projs_2_cams);
    (*projs)(Rect(2,0,2,projs->rows)).copyTo((*projs)(Rect(0,0,2,projs->rows)));
    projs_2_cams.copyTo((*projs)(Rect(2,0,2,projs->rows)));
}

void CheckChangeCams(Mat *projs, Mat *lprojs, Mat *vis_p, Mat *vis_l,
                     bool *changed_cams, bool *changed_views)
{
    *changed_cams = false;
    *changed_views = false;
    if (vis_p->rows == 0)
    {
        return;
    }
    if (int(vis_p->at<uchar>(0, 0)) + int(vis_p->at<uchar>(0, 1)) < 2)
    {
        SwapCameras(projs);
        SwapCameras(lprojs);
        SwapCameras(vis_p);
        SwapCameras(vis_l);
        *changed_cams = true;
    }
    if ((int)vis_p->at<uchar>(0, 2) == 0)
    {
        *projs = -*projs;
        for (int li = 0; li < lprojs->rows; li++)
        {
            for (int ci = 0; ci < 4; ci++)
            {
                for (int i = 0; i < 2; i++)
                {
                    lprojs->at<Vec3d>(li, ci)[i] *= -1;
                }
            }
        }
        SwapViews(projs);
        SwapViews(lprojs);
        SwapViews(vis_p);
        SwapViews(vis_l);
        *changed_views = true;
    }
}

void CorrectChangedCams(bool changed_cams, bool changed_views, std::vector<Matrix3d> *Rs,
                        std::vector<Vector3d> *ts)
{
    if (changed_views)
    {
        Matrix3d T = Matrix3d::Identity();
        T(0,0) = -1;
        T(1,1) = -1;
        Vector3d t2;
        t2<<1,0,0;
        for (int i = 0; i < Rs->size(); i++)
        {
            (*Rs)[i] = T * (*Rs)[i] * T;
            Vector3d t4 = T * (*ts)[i];
            (*ts)[i] = (*Rs)[i] * t2 + t4 - t2;
        }
    }
    if (changed_cams)
    {
        for (int i = 0; i < Rs->size(); i++)
        {
            (*Rs)[i] = Matrix3d((*Rs)[i].transpose());
            (*ts)[i] = -(*Rs)[i] * (*ts)[i];
        }
    }
}

void find_feature_classes(const Mat& vis_p, const Mat& vis_l, std::vector<int>* feat_type,
                          std::vector<int>* ip_num)
{
    feat_type->clear();
    ip_num->clear();
    for (int i = 1; i < vis_p.rows; i++)
    {
        int cur_pt_ip = 2;
        if ((int)vis_p.at<uchar>(i, 2) == 0 || (int)vis_p.at<uchar>(i, 3) == 0)
        {
            cur_pt_ip = 1;
        }
        ip_num->push_back(cur_pt_ip);
        feat_type->push_back(1);
    }
    for (int li=0; li < vis_l.rows; li++)
    {
        int cur_ln_ip = 2;
        if ((int)vis_l.at<uchar>(li, 2) == 0 || (int)vis_l.at<uchar>(li, 3) == 0)
        {
            cur_ln_ip = 1;
        }
        ip_num->push_back(cur_ln_ip);
        feat_type->push_back(2);
    }
}

int FindCaseId(const Mat &vis_p, const Mat &vis_l)
{
    std::vector<int> feat_type, ip_num;
    find_feature_classes(vis_p, vis_l, &feat_type, &ip_num);
    int case_num = 6;
    if (feat_type[0] == 1 && feat_type[1] == 2)
    {
        if (ip_num[0] == 1 && ip_num[1] == 2)
        {
            case_num = 1;
        } else {
            case_num = 2;
        }
    }
    if (feat_type[0] == 1 && feat_type[1] == 1)
    {
        case_num = 3;
    }
    if (feat_type[0] == 2 && feat_type[1] == 2 && feat_type.size() == 2)
    {
        if (ip_num[0] == 2 && ip_num[1] == 2)
        {
            case_num = 4;
        } else {
            case_num = 5;
        }
    }
    if (ip_num.size() == 2 && ip_num[0] == 1 && ip_num[1] == 1)
    {
        if (feat_type[0] == 1 && feat_type[1] == 1)
        {
            case_num = 7;
        }
        if (feat_type[0] == 1 && feat_type[1] == 2)
        {
            case_num = 8;
        }
        if (feat_type[0] == 2 && feat_type[1] == 2)
        {
            case_num = 9;
        }
    }
    return case_num;
}


bool EpiSEgo(const Mat &projs_cur, const Mat &lprojs_cur,
             const Mat &vis_p_cur, const Mat &vis_l_cur, bool id_det_check,
             std::vector<Matrix3d> *Rs, std::vector<Vector3d> *ts)
{


    Matrix<double, 4, 18> A;
    Vector3d pt_shift;
    if (!GenerateEquationsEpipolar(projs_cur, lprojs_cur, vis_p_cur, vis_l_cur, &A, &pt_shift))
    {
        return false;
    }

    Matrix<double, 10, 10> Q = BuildQ();
    Matrix<double, Dynamic,Dynamic, RowMajor> M1 = A.block<4,9>(0,0) * Q.block<9,10>(0,0);
    Matrix<double, Dynamic,Dynamic, RowMajor> M2 = A.block<4,9>(0,9) * Q.block<9,10>(0,0);

    double* Mcp = new double[120*240];
    for (int i = 0; i < 120*240; i++)
    {
        Mcp[i] = 0.0;
    }

    pluecker_assign((double (*)[10]) M1.data(), (double (*)[10]) M2.data(), (double (*)[120]) Mcp);
    Map<Matrix<double, 240, 120, RowMajor>> Mc(Mcp);

    const int reduce_pos = 72;
//faster, but less accurate: Eigen solution Mc'*Mc
//    PartialPivLU<Matrix<double, Dynamic, Dynamic, RowMajor>> lu(Mc.transpose() * Mc);
//    MatrixXd u = lu.matrixLU().triangularView<Upper>();
//    MatrixXd U = u;
//    MatrixXd L = lu.matrixLU().triangularView<UnitLower>();
//    int row_num = U.rows();
//slower, but more accurate: LAPACK solution
    MatrixXd U = Mc;
    int* ipiv = new int[120];
    int info;
    int64 t_sart_lu = cv::getTickCount();
    dgetrf_(240, 120, U.data(), 240, ipiv, info);
    int64 t_end_lu = cv::getTickCount();
    //make U upper-triangular
    for (int i = 0; i < U.rows(); i++)
    {
        int im = std::min(i, int(U.cols()));
        for (int j = 0; j < im; j++)
        {
            U(i,j) = 0.0;
        }
    }
    int row_num = 240;
    //find block of amatrix U for variable expression
    int ind = 0;
    while (ind < row_num && U.block<1,reduce_pos+1>(ind, 0).norm() > 0)
    {
        ind += 1;
    }
    int col_num = Mc.cols();
    MatrixXd U1 = U.block(ind, reduce_pos+1, row_num-ind, col_num-reduce_pos-1);

    const int n_r = 15;
    Matrix<double, n_r, n_r> Ud = U1.block<n_r, n_r>(0,0);
    Matrix<double, n_r, 32> Cb = U1.block<n_r, 32>(0, n_r);
    //express unknowns using known
    MatrixXd A_red = Ud.fullPivHouseholderQr().solve(Cb);

    //fill action matrix
    std::vector<int> action_inds{-5,-8,-10,74,-6,-7,75,-9,76,77,-22,-23,-24,-25,78,-29,-30,79,80,-32,81,-26,-27,82,83,-28,84,85,-31,86,87,88};
    Matrix<double, 32, 32> A_fin;
    A_fin.setZero();
    int c_ind = 0;
    for (int i = 0; i < 32; i++)
    {
        if (action_inds[i] < 0)
        {
            A_fin(i, -action_inds[i]-1) = 1;
        } else {
            A_fin.row(i) = - A_red.row(c_ind);
            c_ind += 1;
        }
    }

    //solve for the unknowns using eigendecomposition of the action matrix
    EigenSolver<Matrix<double, 32, 32>> es(A_fin);
    std::vector<double> bs, cs, ds;
    for (int i = 0; i < 32; i++)
    {
        if (fabs(es.eigenvalues()[i].imag()) == 0.0)
        {
            double c = es.eigenvalues()[i].real();
            VectorXcd v = es.eigenvectors().col(i);
            double b = v(1).real()/v(0).real();
            double d = v(10).real()/v(0).real();
            bs.push_back(b);
            cs.push_back(c);
            ds.push_back(d);
        }
    }

    //find candidate rotations and translations
    for (int i = 0; i < bs.size(); i++)
    {
        VectorXd q(4);
        q(0) = 1.0;
        q(1) = bs[i];
        q(2) = cs[i];
        q(3) = ds[i];
        q = q / q.norm();
        Matrix3d R;
        Quat2Rot(q, &R);

        Matrix<double, 10, 1> v;
        double a = q(0);
        double b = q(1);
        double c = q(2);
        double d = q(3);
        v << a*b, a*c, a*d, b*c, b*d, c*d, a*a, b*b, c*c, d*d;
        Matrix<double, 4, 1> m2v = M2*v;
        MatrixXd e = -1*m2v.fullPivHouseholderQr().solve(M1*v);
        Vector3d p3;
        Vec2d p3_cv = projs_cur.at<Vec2d>(0, 2);
        p3 << p3_cv[0], p3_cv[1], 1.0;
        Vector3d tf = p3 * e(0,0);
        Rs->push_back(R);
        ts->push_back(tf - R*pt_shift);
    }
    
    delete[] Mcp;

    return true;
}

void SwapColumns(Mat &m, int i, int j)
{
    if (m.rows == 0)
        return;

    Mat c;
    m.col(j).copyTo(c);
    m.col(i).copyTo(m.col(j));
    c.copyTo(m.col(i));
}

void SwapViews(Mat* m)
{
    SwapColumns(*m, 0, 1);
    SwapColumns(*m, 2, 3);
}

void PrepareTempMat(const Mat& mat, Mat* temp_mat_p)
{
    if (mat.rows>0)
    {
        mat.copyTo(*temp_mat_p);
    }
}

bool sego_solver(const Mat& projs, const Mat& lprojs,
                 const Mat& vis_p, const Mat& vis_l, bool is_det_check, bool is_right_left,
                 std::vector<Matrix3d>* Rs, std::vector<Vector3d>* ts)
{


    Mat projs_cur, lprojs_cur, vis_p_cur, vis_l_cur;
    PrepareTempMat(projs, &projs_cur);
    PrepareTempMat(lprojs, &lprojs_cur);
    PrepareTempMat(vis_p, &vis_p_cur);
    PrepareTempMat(vis_l, &vis_l_cur);

    if (!is_right_left)
    {
        SwapViews(&projs_cur);
        SwapViews(&lprojs_cur);
        SwapViews(&vis_p_cur);
        SwapViews(&vis_l_cur);
    }

    //we need to swap the cameras if the first point has the 2nd camera as main;
    //we need to swap the views if the first point has the projection on th 2nd view of the non-min camera
    bool changed_cams, changed_views;
    CheckChangeCams(&projs_cur, &lprojs_cur, &vis_p_cur, &vis_l_cur, &changed_cams, &changed_views);

    int k = FindCaseId(vis_p_cur, vis_l_cur);

    bool ret_val;
    if (k >= 6)
    {
        ret_val = SolveWith3Quadric(projs_cur, lprojs_cur, vis_p_cur, vis_l_cur, is_det_check, Rs, ts);
    } else {
        ret_val = EpiSEgo(projs_cur, lprojs_cur, vis_p_cur, vis_l_cur, is_det_check, Rs, ts);
    }

    CorrectChangedCams(changed_cams, changed_views, Rs, ts);

    if (!is_right_left)
    {
        Vector3d t12;
        t12.setZero();
        t12(0) = -1.0;
        for (int si = 0; si < ts->size(); si++)
        {
            (*ts)[si] = (*ts)[si] + (*Rs)[si] * t12 - t12;
        }
    }

    return ret_val;
}