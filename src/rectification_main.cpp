#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "Eigen.h"
#include "directory.h"
#include "rectification/rectification.hpp"

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
						detector_id::BRISK);
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

int main(int argc, char **argv) {
	std::string image_path;
	if (argc == 1) {
		image_path =
			getCurrentDirectory() + "/../../data/MiddEval3/trainingH/Adirondack";
	} else {
		image_path = std::string(argv[1]);
	}

	// load stereo images
	cv::Mat imageLeft = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);
	cv::Mat imageRight = cv::imread(image_path + "/im1.png", cv::IMREAD_COLOR);
	if (!imageLeft.data || !imageRight.data) {
		std::cout << "No image data. Check file path!" << std::endl;
		return -1;
	}

	vector<cv::Point2d> good_matches_1, good_matches_2;
	auto F =
		fundamentalMat(imageLeft, imageRight, good_matches_1, good_matches_2);
	auto rectifier = ImageRectifier(imageLeft,
									imageRight,
									F,
									good_matches_1,
									good_matches_2);
	auto leftRectified = rectifier.getRectifiedLeft();
	auto rightRectified = rectifier.getRectifiedRight();

	cv::imwrite("../../results/rectifiedLeft.png",
				leftRectified);
	cv::imwrite("../../results/rectifiedRight.png",
				rightRectified);

	auto F_r = fundamentalMat(leftRectified,
							  rightRectified,
							  rectifier.getRectifiedLeftMatches(),
							  rectifier.getRectifiedRightMatches());

	ImageRectifier::drawRectifiedEpilines(leftRectified,
										  rightRectified,
										  F_r,
										  rectifier.getRectifiedLeftMatches(),
										  rectifier.getRectifiedRightMatches(),
										  150);

	cv::imwrite("../../results/rectifiedLeftEpilines.png",
				leftRectified);
	cv::imwrite("../../results/rectifiedRightEpilines.png",
				rightRectified);

	return 0;
}
