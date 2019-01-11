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


#ifndef SM_QUADRIC_SOLVER_H
#define SM_QUADRIC_SOLVER_H

#include <vector>
#include <Eigen/Dense>

bool SolveQuadricSystem(const Eigen::Matrix<double, 3, 10> &M, bool is_det_check, std::vector<double> *bs,
                        std::vector<double> *cs, std::vector<double> *ds);

#endif //SM_QUADRIC_SOLVER_H
