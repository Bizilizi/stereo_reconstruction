//
// Created by tim on 19.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "Eigen.h"
#include "directory.h"
#include "eight_point.h"


void featureMatching(const cv::Mat& featuresLeft, const cv::Mat& featuresRight, std::vector<cv::DMatch>& outputMatches, float ratio_thresh=0.7f) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(featuresLeft, featuresRight, knn_matches, 2 );

    // filter matches using the Lowe's ratio test
    for (auto & knn_match : knn_matches)
    {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance)
        {
            outputMatches.push_back(knn_match[0]);
        }
    }
}


void SIFTKeypointDetection(const cv::Mat &image, std::vector<cv::KeyPoint> &outputKeypoints, cv::Mat& outputDescriptor,
                           const float edgeThreshold = 10, const float contrastThreshold = 0.04){
    // detect outputKeypoints using SIFT
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(100, 3, contrastThreshold, edgeThreshold);
    detector->detectAndCompute(image, cv::noArray(), outputKeypoints, outputDescriptor);

    // sort detected outputKeypoints
    // std::sort(outputKeypoints.begin(), outputKeypoints.end(), [](cv::KeyPoint& p1, cv::KeyPoint& p2) {return p1.pt.x > p2.pt.x; });

    // erase duplicates
    // outputKeypoints.erase(std::unique(outputKeypoints.begin(), outputKeypoints.end(), [](cv::KeyPoint& p1, cv::KeyPoint& p2) {
    //     return std::hypot(p1.pt.x-p2.pt.x, p1.pt.y - p2.pt.y) < 10; }), outputKeypoints.end());
}


int main(int argc, char** argv) {
    std::string image_path;
    if (argc == 1) {
        image_path = getCurrentDirectory() + "/../../data/MiddEval3/trainingH/ArtL";
    } else {
        image_path = std::string(argv[1]);
    }

    // load stereo images
    cv::Mat imageLeft = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread(image_path + "/im1.png", cv::IMREAD_COLOR);
    if ( !imageLeft.data || !imageRight.data) {
        std::cout << "No image data. Check file path!" << std::endl;
        return -1;
    }

    // find keypoints
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat featuresLeft, featuresRight;
    // FIXME: Filter out double matches!
    SIFTKeypointDetection(imageLeft, keypointsLeft, featuresLeft);
    SIFTKeypointDetection(imageRight, keypointsRight, featuresRight);

    // find correspondences
    std::vector<cv::DMatch> matches;
    featureMatching(featuresLeft, featuresRight, matches);

    // visualization of feature extraction
    cv::Mat outputImageLeft, outputImageRight;
    cv::drawKeypoints(imageLeft, keypointsLeft, outputImageLeft);
    cv::drawKeypoints(imageRight, keypointsRight, outputImageRight);
    cv::imwrite("../../results/imageLeft.png", outputImageLeft);
    cv::imwrite("../../results/imageRight.png", outputImageRight);

    // visualization of feature matching
    cv::Mat img_matches;
    cv::drawMatches( imageLeft, keypointsLeft, imageRight, keypointsRight, matches, img_matches, cv::Scalar::all(-1),
                     cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imwrite("../../results/matching_flann.png", img_matches);


    // estimate pose using the eight point algorithm
    // TODO read calib.txt to get camera matrices -> SDK?
    Matrix3f cameraLeft;
    cameraLeft << 1870, 0, 297, 0, 1870, 277, 0, 0, 1;
    Matrix3f cameraRight;
    cameraRight << 1870, 0, 397, 0, 1870, 277, 0, 0, 1;

    Matrix3Xf kpLeftMat, kpRightMat;
    transformMatchedKeypointsToEigen(keypointsLeft, keypointsRight, matches, kpLeftMat, kpRightMat);

    EightPointAlgorithm ep(kpLeftMat, kpRightMat, cameraLeft, cameraRight);
    Matrix4f pose = ep.getPose();
    Matrix3f esentialMatrix = ep.getEssentialMatrix();

    std::cout << "Pose: " << std::endl;
    std::cout << pose << std::endl;

    // showExtrinsicsReconstruction(filename, kp)

    return 0;
}


/**
* TODO:
 *
 * [1. Ransac Outlier filtern] (next week)
 *
 * 2. Optimization: Redefinement of results of 8 pt algorithm (Eigen)
 *      - BundleAdjustment (Ceres)
 *
 * 3. SDK: Reading Image Data
 *     int image_idx = 3;  // one index per scenario in trainingFiles
 *     class Dataloader(/relative/path/to/training/data )
 *     load(idx) > data struct
 *
 *     data struct{
 *     imageLeft,
 *     imageRight,
 *     intrinsics,
 *     xyz,
 *     }
 *
*/
