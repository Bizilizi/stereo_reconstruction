//
// Created by gabriel on 21.06.21.
//

#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include "Eigen.h"

class EightPointAlgorithm {
/**
 * Estimates the extrinsics of a two view perspective
 * @param matchesLeft (3, N) stacked matched keypoints of left image
 * @param matchesRight (3, N) stacked matched keypoints of right image
 * @param cameraLeft (3, 3) camera matrix (intrinsics) of left image
 * @param cameraRight (3, 3) camera matrix (intrinsics) of right image
 * */
public:
    EightPointAlgorithm(const Matrix3Xf &matchesLeft,
                        const Matrix3Xf &matchesRight,
                        const Matrix3f &cameraLeft,
                        const Matrix3f &cameraRight);

    void run();

    /**
     * Getters
     */

    const Matrix3Xf &getMatchesLeft() const;

    const Matrix3Xf &getMatchesRight() const;

    const Matrix3f &getCameraLeft() const;

    const Matrix3f &getCameraRight() const;

    const Matrix4f &getPose() const;

    const Matrix3f &getEssentialMatrix() const;

    Matrix3f getFundamentalMatrix() const;

    const Matrix3Xf &getPointsLeftReconstructed() const;

    const Matrix3Xf &getPointsRightReconstructed() const;

    /**
     * Setters
     */

    void setMatches(const Matrix3Xf &leftMatches, const Matrix3Xf &rightMatches);

    void setCameraRight(const Matrix3f &camera);

    void setCameraLeft(const Matrix3f &camera);


private:
    /**
     * Reconstructs depth of corresponding 2D points in two views by triangulation.
     * @param R (3, 3) Rotation matrix
     * @param T (3) Translation vector
     * @return success boolean indicating whether all depths are positive (otherwise reconstruction failed)
     */
    bool structureReconstruction(const Matrix3f &R, const Vector3f &T);

    /**
     * Projects matches into 3D by applying inverse intrinsics.
     */
    void updateData();

    // inputs
    Matrix3Xf matchesLeft;
    Matrix3Xf matchesRight;
    Matrix3f cameraLeft;
    Matrix3f cameraRight;
    Matrix3Xf pointsLeft;
    Matrix3Xf pointsRight;
    int numMatches;

    // for results
    Matrix4f pose;
    Matrix3f essentialMatrix;
    Matrix3Xf pointsLeftReconstructed;
    Matrix3Xf pointsRightReconstructed;

};

/**
 * Converts a vector to its corresponding skew symmetric matrix.
 * @param vec Vector to b e converted
 * @return Skew symmetric matrix
 */
Matrix3f vectorAsSkew(const Vector3f &vec);

/**
 * Kronecker product for two vectors.
 * @param vec1
 * @param vec2
 * @return Kronecker product
 */
VectorXf kron(const VectorXf &vec1, const VectorXf &vec2);

/**
 * Convert the matched keypoint pairs given to two Eigen matrices.
 * @param keypointsLeft vector of keypoints in left image
 * @param keypointsRight vector of keypoints in right image
 * @param matches cv::DMatch instance that contains the indices of matched keypoints
 * @param outLeft output matrix for keypoints in left picture
 * @param outRight output matrix for keypoints in right picture
 */
void transformMatchedKeypointsToEigen(const std::vector<cv::KeyPoint> &keypointsLeft,
                                      const std::vector<cv::KeyPoint> &keypointsRight,
                                      const std::vector<cv::DMatch> &matches,
                                      Matrix3Xf &outLeft,
                                      Matrix3Xf &outRight);

/**
 * Returns the column indices corresponding to unique elements.
 * @param pointMat: (3,n)-matrix containing column-wise 3d vectors
 * @param tol: max. tolerance for comparing elements
 * @return vector of unique elements id
 */
std::vector<int> uniqueColumnsInMatrix(const Matrix3Xf &pointMat, float tol=0.1f);

#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H