//
// Created by tim on 15.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "Eigen.h"


void displayStereoImages(cv::Mat &image1, cv::Mat &image2){
    cv::Mat stackedImages;
    cv::hconcat(image1, image2, stackedImages);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", stackedImages);
}

void displaySingleImage(cv::Mat &image1){
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image1);
}

int main(int argc, char** argv ) {


    std::string image_path;
    if (argc == 1) {
        image_path = "/home/tim/repos/stereo_reconstruction/data/MiddEval3/trainingH/ArtL";
    } else {
        image_path = std::string(argv[1]);
    }

    // Stereo Images
    cv::Mat image1 = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(image_path + "/im1.png", cv::IMREAD_COLOR);

    // Disparity map
    // TODO: Use SDK and cvkit
    // SDK: load image
    // cvkit: display image
    cv::Mat rawDispImage = cv::imread(image_path + "/disp0GT.pfm", cv::IMREAD_UNCHANGED);


    if ( !image1.data || !image2.data)
    {
        std::cout << "No image data" << std::endl;
        return -1;
    } else {
        std::cout << "Reading successful" << std::endl;
        displayStereoImages(image1, image2);
    }

    cv::waitKey(0);
}
