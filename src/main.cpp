
#include "PoseEstimation/keypoints.h"
#include "PoseEstimation/bundle_adjustment.h"
#include "PoseEstimation/pose_estimation.h"
#include "DataLoader/data_loader.h"
#include "Rectification/rectification.hpp"
#include "Reconstruction/reconstruction.h"


#define RUN_ALL 1 // set 0 to only run reconstruction based on pre-computed disparity image


enum detector_id {
    ORB,
    BRISK
};


enum descriptor_id {
    BRUTE_FORCE,
    FLANN_BASE
};


cv::Mat detectFeatures(cv::Mat image,
                       enum detector_id det_id,
                       vector<cv::KeyPoint> &keypoints) {
    // Declare detector
    cv::Ptr<cv::Feature2D> detector;

    // Define detector
    if (det_id == detector_id::ORB) {
        // Declare ORB detector
        detector = cv::ORB::create(
                500,                //nfeatures = 500
                1.2f,               //scaleFactor = 1.2f
                4,                  //nlevels = 8
                21,                 //edgeThreshold = 31
                0,                  //firstLevel = 0
                2,                  //WTA_K = 2
                cv::ORB::HARRIS_SCORE,  //scoreType = ORB::HARRIS_SCORE
                21,                 //patchSize = 31
                20                  //fastThreshold = 20
        );
    } else {
        // Declare BRISK and BRISK detectors
        detector = cv::BRISK::create(
                30,   // thresh = 30
                3,    // octaves = 3
                1.0f  // patternScale = 1.0f
        );
    }

    // Declare array for storing the descriptors
    cv::Mat descriptors;

    // Detect and compute!
    detector->detect(image, keypoints);
    detector->compute(image, keypoints, descriptors);

    return descriptors;
}


std::pair<vector<cv::Point2d>, vector<cv::Point2d> > match(cv::Mat &one,
                                                           cv::Mat &other,
                                                           enum descriptor_id descriptor,
                                                           enum detector_id detector) {
    // 1 - Get keypoints and its descriptors in both images
    vector<cv::KeyPoint> keypoints[2];
    cv::Mat descriptors[2];

    descriptors[0] = detectFeatures(one, detector, keypoints[0]);
    descriptors[1] = detectFeatures(other, detector, keypoints[1]);

    // 2 - Match both descriptors using required detector
    // Declare the matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Define the matcher
    if (descriptor == descriptor_id::BRUTE_FORCE) {
        // For ORB and BRISK descriptors, NORM_HAMMING should be used.
        // See http://sl.ugr.es/norm_ORB_BRISK
        matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
    } else {
        matcher = new cv::FlannBasedMatcher();
        // FlannBased Matcher needs CV_32F descriptors
        // See http://sl.ugr.es/FlannBase_32F
        for (size_t i = 0; i < 2; i++) {
            if (descriptors[i].type() != CV_32F) {
                descriptors[i].convertTo(descriptors[i], CV_32F);
            }
        }
    }

    // Match!
    vector<cv::DMatch> matches;
    matcher->match(descriptors[0], descriptors[1], matches);

    // 3 - Create lists of ordered keypoints following obtained matches
    vector<cv::Point2d> ordered_keypoints[2];

    for (unsigned int i = 0; i < matches.size(); i++) {
        // Get the keypoints from the matches
        ordered_keypoints[0].push_back(keypoints[0][matches[i].queryIdx].pt);
        ordered_keypoints[1].push_back(keypoints[1][matches[i].trainIdx].pt);
    }

    return std::pair<vector<cv::Point2d>, vector<cv::Point2d> >(
            ordered_keypoints[0],
            ordered_keypoints[1]);
}


cv::Mat fundamentalMat(cv::Mat &one, cv::Mat &other,
                       vector<cv::Point2d> &good_matches_1,
                       vector<cv::Point2d> &good_matches_2) {

    std::pair<vector<cv::Point2d>, vector<cv::Point2d> > matches;
    cv::Mat F;
    vector<unsigned char> mask;

    vector<cv::Point2d> first, second;
    int flag = cv::FM_8POINT;

    if (good_matches_1.empty() && good_matches_2.empty()) {
        matches = match(one,
                        other,
                        descriptor_id::BRUTE_FORCE,
                        detector_id::ORB);
        first = matches.first;
        second = matches.second;
        flag |= cv::FM_RANSAC;
    } else {
        first = good_matches_1;
        second = good_matches_2;
    }

    F = findFundamentalMat(first, second,
                           flag,
                           1., 0.99, mask);

    vector<cv::Point2d> final_1, final_2;

    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 1) {
            final_1.push_back(first[i]);
            final_2.push_back(second[i]);
        }
    }

    good_matches_1 = vector<cv::Point2d>(final_1);
    good_matches_2 = vector<cv::Point2d>(final_2);

    return F;
}


int main(){
    DataLoader dataLoader = DataLoader();

    // select scenarios by index (alphabetic position starting with 0)
    int scenarioIdx = 3;
    Data data = dataLoader.loadTrainingScenario(scenarioIdx);

#if RUN_ALL
    /**
     * 1. Estimate Extrinsics (Fundamental Matrix)
     */

    // select scenarios by index (alphabetic position starting with 0)
    poseStruct estimatedPose = runFullPoseEstimation(data.getImageLeft(), data.getImageRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight(), true, true);
    Matrix3f fundamentalMatrix = estimatedPose.fundamentalMatrix;

    std::cout << "Fundamental matrix (ours): " << std::endl << fundamentalMatrix / fundamentalMatrix.norm() << std::endl;

    /**
     * 2. Compute Disparity Map
     */

    auto img_left = data.getImageLeft();
    auto img_right = data.getImageRight();
    auto rectifier = ImageRectifier(img_left, img_right, fundamentalMatrix, estimatedPose.keypointsLeft, estimatedPose.keypointsRight);

    rectifier.computeDisparityMapRight(17, 200, 0.9);
    cv::Mat disparityImageWrite = rectifier.getDisparityMapRight();

    std::string disparityPath = "../../results/disparity_map.png";
    cv::imwrite(disparityPath, disparityImageWrite);
#endif

    /**
     * 3. Reconstruct scene
     */

    cv::Mat disparityImage = DataLoader::readGrayscaleImageAsDisparityMap(disparityPath);

    // remove outliers
    removeDisparityOutliers(disparityImage, 500, 1.5, 0.8);

    // create depth map
    float focalLength = data.getCameraMatrixRight()(0,0);
    float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)
    cv::Mat depthValues = convertDisparityToDepth(disparityImage, focalLength, baseline);

    // intrinsics
    Matrix3f intrinsics = data.getCameraMatrixRight();

    // reconstruct geometry and compute triangular mesh
    float thrMesh = 1.f;
    reconstruction(data.getImageRight(), depthValues, intrinsics, thrMesh);
}

