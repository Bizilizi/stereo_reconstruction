
#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include "../Eigen.h"

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

    /**
     * Reconstructs depth of corresponding 2D points in two views by triangulation.
     * @param R (3, 3) Rotation matrix
     * @param T (3) Translation vector
     * @return success boolean indicating whether all depths are positive (otherwise reconstruction failed)
     */
    bool structureReconstruction(const Matrix3f &R, const Vector3f &T);

    VectorXf estimateDepth(const Matrix3f &R, const Vector3f &T) const;

private:

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
 * Implements our own version of RANSAC. We propose the following steps to make the Eight-Point-Algorithm robust against outliers: (
 * (i) Randomly sample N points and run the Eight-Point until a valid reconstruction (all depth values for reconstructed points in left and right
 * image are positive)
 * (ii) Refine this results by iteratively calculating the left-to-right projection error for every of the N key points and replacing it
 * with a new random point.
 *
 * @param kpLeftMat : left keypoints provided in a Eigen Matrix
 * @param kpRightMat : right keypoints provided in a Eigen Matrix
 * @param cameraLeft : left camera intrinsics
 * @param cameraRight : right camera intrinsics
 * @return Instance of the Eight-Point-Algorithm class. NOTE: Needs to be rerun using ep.run() before accessing results
 */

EightPointAlgorithm
RANSAC(const MatrixXf &kpLeftMat, const MatrixXf &kpRightMat, const Matrix3f &cameraLeft, const Matrix3f &cameraRight);

std::vector<int> getRandomIndices(int maxIdx, int length, std::vector<int> init = {}, std::vector<int> exclude = {});

VectorXf calculateEuclideanPixelError(const MatrixXf &leftToRightProjection, const MatrixXf &matchesRight);


#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H