//
// Created by gabriel on 21.06.21.
//

#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include "Eigen.h"
void eightPointAlgorithm(const std::vector<cv::KeyPoint>& keypointsLeft,
                         const std::vector<cv::KeyPoint>& keypointsRight,
                         const std::vector<cv::DMatch>& matches,
                         const Matrix3f& cameraLeft,
                         const Matrix3f& cameraRight,
                         Matrix4f& pose);

/**
 * Reconstructs depth of corresponding 2D points in two views by triangulation.
 * @param R (3,3) Rotation matrix
 * @param T (3) Translation vector
 * @param xLeft (3, N) homogenous coordinates for matched points in left picture
 * @param xRight (3, N) homogenous coordinates for matched points in left picture
 * @param xLeftReconstructed output
 * @param xRightReconstructed output
 * @return success boolean indicating whether all depths are positive (otherwise reconstruction failed)
 */
bool structureReconstruction(const Matrix3f& R, const Vector3f& T, const MatrixXf& xLeft, const MatrixXf& xRight, MatrixXf& xLeftReconstructed, MatrixXf& xRightReconstructed);

/**
 * Converts a vector to its corresponding skew symmetric matrix.
 * @param vec Vector to b e converted
 * @return Skew symmetric matrix
 */
Matrix3f vectorAsSkew(const Vector3f& vec);

#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H
