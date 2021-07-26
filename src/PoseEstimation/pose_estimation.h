#ifndef STEREO_RECONSTRUCTION_POSE_ESTIMATION_H
#define STEREO_RECONSTRUCTION_POSE_ESTIMATION_H

#include "../utils.h"
#include "bundle_adjustment.h"
#include "eight_point.h"
#include "../Eigen.h"

struct poseStruct {
    Matrix4f pose;
    Matrix3f fundamentalMatrix;
    Matrix3Xf keypointsLeft;
    Matrix3Xf keypointsRight;
    float reError8pt;
    float reErrorBA;
};

/**
 * Runs the whole pose estimation including the Eight-Point-Algorithm within the RANSAC loop and bundle adjustment optimization
 * @param imageLeft : left image
 * @param imageRight : right image
 * @param intrinsicsLeft : intrinsics of left camera
 * @param intrinsicsRight : intrinsics of right camera
 * @param verbose : switch for outputs
 * @return
 */

poseStruct runFullPoseEstimation(const cv::Mat &imageLeft, const cv::Mat &imageRight, const Matrix3f &intrinsicsLeft,
                                 const Matrix3f &intrinsicsRight, bool verbose = false);


#endif //STEREO_RECONSTRUCTION_POSE_ESTIMATION_H
