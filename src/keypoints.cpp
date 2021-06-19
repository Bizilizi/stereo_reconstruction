//
// Created by tim on 19.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>


int main(int argc, char** argv) {
    std::string image_path;
    if (argc == 1) {
        image_path = "/home/tim/repos/stereo_reconstruction/data/MiddEval3/trainingH/ArtL";
        std::cout << "Its me! " << std::endl;
    } else {
        image_path = std::string(argv[1]);
    }

    // Stereo Images
    cv::Mat image1 = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);

    // SIFT feature
    //cv::SiftFeatureDetector detector;
    //std::vector<cv::KeyPoint> keypoints;
    //detector.detect(image1, keypoints);

    //
    int featuresToRetain = 20;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(featuresToRetain);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image1, keypoints);

    // Add results to image and save.
    cv::Mat outputImage;
    cv::drawKeypoints(image1, keypoints, outputImage);
    cv::imwrite("../../results/out.png", outputImage);

    return 0;
}
