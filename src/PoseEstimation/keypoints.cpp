
#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "../Reconstruction/simple_mesh.h"

#include "keypoints.h"

#define N_KEYPOINTS_SIFT 250


void featureMatching(const cv::Mat &featuresLeft, const cv::Mat &featuresRight, std::vector<cv::DMatch> &outputMatches,
                     float ratio_thresh) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(featuresLeft, featuresRight, knn_matches, 2);

    // filter matches using the Lowe's ratio test
    for (auto &knn_match : knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            outputMatches.push_back(knn_match[0]);
        }
    }
}


void SIFTKeypointDetection(const cv::Mat &image, std::vector<cv::KeyPoint> &outputKeypoints, cv::Mat &outputDescriptor,
                           const float edgeThreshold, const float contrastThreshold) {
    // detect outputKeypoints using SIFT
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(N_KEYPOINTS_SIFT, 3, contrastThreshold, edgeThreshold);
    detector->detectAndCompute(image, cv::noArray(), outputKeypoints, outputDescriptor);
}


void showExtrinsicsReconstruction(const std::string &filename, const Matrix4f &pose,
                                  const Matrix3Xf &pointsLeftReconstructed, const Matrix3Xf &pointsRightReconstructed) {
    SimpleMesh outputMesh;
    for (int i = 0; i < pointsLeftReconstructed.cols(); i++) {
        SimpleMesh point = SimpleMesh::sphere(pointsLeftReconstructed.col(i), 0.5f);
        outputMesh = SimpleMesh::joinMeshes(outputMesh, point);
    }

    SimpleMesh cameraLeftMesh = SimpleMesh::camera(Matrix4f::Identity(), 0.05f);
    SimpleMesh cameraRightMesh = SimpleMesh::camera(pose, 0.05f, Vector4uc(0, 255, 0, 255));

    SimpleMesh cameraMesh = SimpleMesh::joinMeshes(cameraLeftMesh, cameraRightMesh);
    outputMesh = SimpleMesh::joinMeshes(outputMesh, cameraMesh);

    outputMesh.writeMesh(filename);
}
