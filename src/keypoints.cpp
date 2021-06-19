//
// Created by tim on 19.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>


int main(int argc, char** argv) {
    std::string image_path;
    if (argc == 1) {
        image_path = "/home/tim/repos/stereo_reconstruction/data/MiddEval3/trainingH/ArtL";
    } else {
        image_path = std::string(argv[1]);
    }

    // Stereo Images
    cv::Mat image1 = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);

    // SIFT feature
    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(image1, keypoints);

    // Add results to image and save.
    cv::Mat outputImage;
    cv::drawKeypoints(image1, keypoints, outputImage);
    cv::imshow("Keypoint detection", outputImage);

    return 0;
}
