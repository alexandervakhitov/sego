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
#include <random>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <iterator>
#include "sego.h"


using namespace Eigen;
using namespace std;
using namespace cv;

Vector2d project_point(const Vector3d& X, const Matrix4d& T)
{
    Vector3d X1 = T.block<3,3>(0,0)*X + T.block<3,1>(0,3);
    if (X1(2)<0)
    {
        std::cout << " behind cam " << std::endl;
    }
    return X1.segment<2>(0)/X1(2);
}


Vector3d project_point_homo(const Vector3d& X, const Matrix4d& T)
{
    Vector3d X1 = T.block<3,3>(0,0)*X + T.block<3,1>(0,3);
    return X1/X1(2);
}

Vector3d project_line(const pair<Vector3d, Vector3d>& endpoints, const Matrix4d& T)
{
    Vector3d x1 = project_point_homo(endpoints.first, T);
    Vector3d x2 = project_point_homo(endpoints.second, T);
    Vector3d lproj1 = x1.cross(x2);
    return lproj1 / lproj1.segment<2>(0).norm();
}

typedef vector<pair<Matrix4d, Matrix4d>, Eigen::aligned_allocator<Eigen::Matrix4d>> camvector;

void build_test(bool is_right_left, camvector* cameras_p,
                vector<vector<pair<Vector2d, Vector2d>>>* point_projections_p,
                vector<vector<pair<Vector3d, Vector3d>>>* line_projections_p)
{
    //generate 3 points with coordinates in [-1,1]

    vector<Vector3d> points;
    for (int i = 0; i < 3; i++)
    {
        Vector3d X;
        X.setRandom();
        points.push_back(X);
    }

    //generate 3 lines
    vector<std::pair<Vector3d, Vector3d>> endpoints;
    for (int i = 0; i < 3; i++)
    {
        Vector3d X1, X2;
        X1.setRandom();
        X2.setRandom();
        endpoints.push_back(make_pair(X1, X2));
    }

    // generate cam at a distance [2,4] from the origin and looking at the origin
    camvector cameras;
    default_random_engine generator;
    uniform_real_distribution<double> distribution(2.0,4.0);
    Vector3d t12;
    t12.setZero();
    if (is_right_left)
    {
        t12(0) = 1.0;
    } else {
        t12(0) = -1.0;
    };
    for (int i = 0; i < 2; i++)
    {
        Matrix4d T;
        T.setIdentity();
        Vector3d c1 = Vector3d::Random();
        c1 = c1/c1.norm() * distribution(generator);
        Vector3d r1, r3;
        r1.setZero();
        r1(0) = 1.0;
        r3 = -c1/c1.norm();
        Vector3d r2 = r3.cross(r1);
        r2 = r2 / r2.norm();
        r1 = r2.cross(r3);
        T.block<1,3>(0,0) = r1.transpose();
        T.block<1,3>(1,0) = r2.transpose();
        T.block<1,3>(2,0) = r3.transpose();
//        std::cout << " ortho " << T.block<3,3>(0,0) * T.block<3,3>(0,0).transpose() << std::endl;
//        std::cout << c1 << std::endl;
//        std::cout << T.block<3,3>(0,0) * c1 << std::endl;
        T.block<3,1>(0,3) = -T.block<3,3>(0,0) * c1;

//        std::cout << T << std::endl;

        Matrix4d T2 = T;
        T2.block<3,1>(0,3) += t12;
        cameras.push_back(make_pair(T, T2));
    }
    //cam, point, first view - second view
    vector<vector<pair<Vector2d, Vector2d>>> point_projections;
    for (int ci = 0; ci < 2; ci++)
    {
        Matrix4d T1 = cameras[ci].first;
        Matrix4d T2 = cameras[ci].second;
        vector<pair<Vector2d, Vector2d>> projs;
        for (int pi = 0; pi < 3; pi++)
        {
            Vector2d proj1 = project_point(points[pi], T1);
            Vector2d proj2 = project_point(points[pi], T2);
            projs.push_back(make_pair(proj1, proj2));
        }
        point_projections.push_back(projs);
    }
    vector<vector<pair<Vector3d, Vector3d>>> line_projections;
    for (int ci = 0; ci < 2; ci++)
    {
        Matrix4d T1 = cameras[ci].first;
        Matrix4d T2 = cameras[ci].second;
        vector<pair<Vector3d, Vector3d>> lineprojs_for_cam;
        for (int li = 0; li < 3; li++)
        {
            Vector3d l1 = project_line(endpoints[li], T1);
            Vector3d l2 = project_line(endpoints[li], T2);
            lineprojs_for_cam.push_back(make_pair(l1, l2));
        }
        line_projections.push_back(lineprojs_for_cam);
    }
    *cameras_p = cameras;
    *point_projections_p = point_projections;
    *line_projections_p = line_projections;
}

void fill_projs(const vector<int>& skip_view_ids, const vector<vector<pair<Vector2d, Vector2d>>>& point_projections,
               Mat* projs, Mat* vis_p)
{
    int n_pt = skip_view_ids.size();
    *projs = Mat(n_pt, 4, CV_64FC2);
    *vis_p = Mat(n_pt, 4, CV_8UC1);
    for (int i = 0; i < n_pt; i++)
    {
        for (int vi = 0; vi < 4; vi++)
        {
            if (skip_view_ids[i] == vi)
            {
                vis_p->at<uchar>(i, vi) = 0;
            } else {
                vis_p->at<uchar>(i, vi) = 1;
                int ci = int(vi / 2);
                if (vi % 2 == 0)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        projs->at<Vec2d>(i, vi)[j] = point_projections[ci][i].first[j];
                    }
                } else {
                    for (int j = 0; j < 2; j++)
                    {
                        projs->at<Vec2d>(i, vi)[j] = point_projections[ci][i].second[j];
                    }
                }
            }
        }
    }
}

void fill_l_projs(const vector<int>& skip_view_ids, const vector<vector<pair<Vector3d, Vector3d>>>& line_projections,
                Mat* l_projs, Mat* vis_l)
{
    int n_ln = skip_view_ids.size();
    *l_projs = Mat(n_ln, 4, CV_64FC3);
    *vis_l = Mat(n_ln, 4, CV_8UC1);
    for (int i = 0; i < n_ln; i++)
    {
        for (int vi = 0; vi < 4; vi++)
        {
            if (skip_view_ids[i] == vi)
            {
                vis_l->at<uchar>(i, vi) = 0;
            } else {
                vis_l->at<uchar>(i, vi) = 1;
                int ci = int(vi / 2);
                if (vi % 2 == 0)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        l_projs->at<Vec3d>(i, vi)[j] = line_projections[ci][i].first[j];
                    }
                } else {
                    for (int j = 0; j < 3; j++)
                    {
                        l_projs->at<Vec3d>(i, vi)[j] = line_projections[ci][i].second[j];
                    }
                }
            }
        }
    }
}

vector<int> make_skip_view_vector(int cam_skip_id, int n)
{
    vector<int> skip_view_ids;
    for (int i = 0; i < n; i++)
    {
        int skip_id = cam_skip_id*2 + rand() % 2;
        skip_view_ids.push_back(skip_id);
    }
    return skip_view_ids;
}

void make_minimal_test(bool is_right_left, int test_id, Mat* projs_p, Mat* vis_pp, Mat* lprojs_p, Mat* vis_lp, Matrix4d* T_ans_p)
{
    camvector cameras;
    vector<vector<pair<Vector2d, Vector2d>>> point_projections;
    vector<vector<pair<Vector3d, Vector3d>>> line_projections;
    build_test(is_right_left, &cameras, &point_projections, &line_projections);
//    std::cout << cameras[0].second.block<3,1>(0,3).transpose()  - cameras[0].first.block<3,1>(0,3).transpose() << std::endl;
    *T_ans_p = cameras[1].first * cameras[0].first.inverse();
    Mat projs, vis_p, lprojs, vis_l;
    if (test_id==0)
    {
//        int cam_skip_id = rand() % 2;
//        auto skip_view_ids = make_skip_view_vector(cam_skip_id, 2);
        fill_projs(vector<int>{2,3}, point_projections, &projs, &vis_p);
//        auto line_skip_view_ids = make_skip_view_vector(1-cam_skip_id, 1);
        fill_l_projs(vector<int>{0}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 1)
    {
        fill_projs(vector<int>{2,0}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{2}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 2)
    {
        fill_projs(vector<int>{1,0,2}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 3)
    {
        fill_projs(vector<int>{2}, point_projections, &projs, &vis_p);
//        auto line_skip_view_ids = make_skip_view_vector(1-cam_skip_id, 1);
        fill_l_projs(vector<int>{0,1}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 4)
    {
        fill_projs(vector<int>{0}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{0,2}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 5)
    {
        fill_projs(vector<int>{}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{0,1,2}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 6)
    {
        fill_projs(vector<int>{0,1,0}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 7)
    {
        fill_projs(vector<int>{0,1}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{1}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 8)
    {
        fill_projs(vector<int>{1}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{0,1}, line_projections, &lprojs, &vis_l);
    }
    if (test_id == 9)
    {
        fill_projs(vector<int>{}, point_projections, &projs, &vis_p);
        fill_l_projs(vector<int>{2,3,2}, line_projections, &lprojs, &vis_l);
    }
    projs.copyTo(*projs_p);
    lprojs.copyTo(*lprojs_p);
    vis_p.copyTo(*vis_pp);
    vis_l.copyTo(*vis_lp);
}

void run_tests(int problem_type, bool is_right_left)
{
    Mat projs, vis_p, lprojs, vis_l;
    Matrix4d T_ans;
    int success_cnt = 0;

    double t_agg = 0;
    for (int it = 0; it < 100; it++)
    {
        make_minimal_test(is_right_left, problem_type, &projs, &vis_p, &lprojs, &vis_l, &T_ans);
        vector<Vector3d> ts;
        vector<Matrix3d> Rs;
        int64 t_start = cv::getTickCount();
        sego_solver(projs, lprojs, vis_p, vis_l, true, is_right_left, &Rs, &ts);
        int64 t_end = cv::getTickCount();

        t_agg += (t_end-t_start)/cv::getTickFrequency();
//        std::cout << " true t " << T_ans.block<3,1>(0,3).transpose() << std::endl;
//        std::cout << " true r " << T_ans.block<1,3>(0,0) << std::endl;
        for (int i = 0; i < ts.size(); i++)
        {
            Matrix4d T_est;
            T_est.setIdentity();
            T_est.block<3,3>(0,0) = Rs[i];
            T_est.block<3,1>(0,3) = ts[i];
            Matrix4d dT = T_est * T_ans.inverse() - Matrix4d::Identity();
//            std::cout << "   est t " << T_est.block<3,1>(0,3).transpose() << std::endl;
//            std::cout << " est r " << T_est.block<1,3>(0,0) << std::endl;
            if (dT.norm() < 1e-4)
            {
                success_cnt++;
            }
        }
//        std::cout << " --- " << std::endl;
    }
    std::cout << " test " << problem_type << " rl " << is_right_left << " : " << t_agg/100 << " sec, " << success_cnt << " of 100" << std::endl;


}

void load_R(const std::string& fpath, Matrix3d* Rp)
{
    std::ifstream f_in(fpath);
    std::string line;
    std::vector<string> file_lines;
    while (f_in >> line) {
        file_lines.push_back(line);
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (*Rp)(j, i) = atof(file_lines[3*i+j].c_str());
        }
    }
}

void load_t(const std::string& fpath, Vector3d* tp)
{
    std::ifstream f_in(fpath);
    std::string line;
    std::vector<string> file_lines;
    while (f_in >> line) {
        file_lines.push_back(line);
    }
    for (int j = 0; j < 3; j++)
    {
        (*tp)[j] = atof(file_lines[j].c_str());
    }

}

void load_projs(bool is_points, const std::string& fpath, Mat* vis_mat, Mat* proj_mat)
{
    std::ifstream f_in(fpath);
    std::string line;
    std::vector<string> file_lines;
    while (f_in >> line) {
        file_lines.push_back(line);
    }
    int n = file_lines.size()/12;
    if (is_points) {
        *proj_mat = cv::Mat(n, 4, CV_64FC2);
    } else {
        *proj_mat = cv::Mat(n, 4, CV_64FC3);
    }
    *vis_mat = cv::Mat(n, 4, CV_8UC1);

    for (int i = 0; i < n; i++)
    {

        for (int ci = 0; ci < 4; ci++)
        {
            for (int j = 0; j < 3; j++)
            {
                float v = atof(file_lines[12*i+ci*3+j].c_str());
                if (!is_points) {
                    proj_mat->at<cv::Vec3d>(i, ci)[j] = v;
                }
                if (is_points && j < 2)
                {
                    proj_mat->at<cv::Vec2d>(i, ci)[j] = v;
                }
                if (j == 1)
                {
                    if (v == -1.0)
                    {
                        vis_mat->at<uchar>(i, ci) = 0;
                    } else {
                        vis_mat->at<uchar>(i, ci) = 1;
                    }
                }
            }
        }
    }
}


int main() {
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            bool is_right_left = bool(j);
            run_tests(i, is_right_left);
        }
    }

    return 0;
}