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

poseStruct runFullPoseEstimation(const cv::Mat &imageLeft, const cv::Mat &imageRight, const Matrix3f &intrinsicsLeft,
                                 const Matrix3f &intrinsicsRight, bool verbose = false, bool visualize = false);


#endif //STEREO_RECONSTRUCTION_POSE_ESTIMATION_H
