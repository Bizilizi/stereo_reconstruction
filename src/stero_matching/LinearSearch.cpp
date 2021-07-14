#include "LinearSearch.h"
#include <limits>
#include <iostream>

LinearSearch::LinearSearch(cv::Mat &leftImage, cv::Mat &rightImage)
: leftImage(leftImage), rightImage(rightImage) {

}

cv::Mat LinearSearch::computeDisparityMap() {
    int h1 = leftImage.rows;
    int w1 = leftImage.cols;
    int h2 = rightImage.rows;
    int w2 = rightImage.cols;

    std::cout << "h1:" << h1 << "\n";
    std::cout << "h2:" << h2 << "\n";

    cv::Mat dispMap = cv::Mat::zeros(h1, w1, CV_64F);
    int max = 0;
    for (int i = 0; i < h1; i++) {
        for (int j = 0; j < w1; j++) {

            if (leftImage.at<cv::Vec3b>(i, j) == cv::Vec3b{0, 0, 0})                 // skip black pixels which are at borders
                continue;

            int col = 0;
            cv::Vec3b pixel_ij = leftImage.at<cv::Vec3b>(i, j);
            double min = std::numeric_limits<double>::max();;
            for (int k = j-200; k < j; k++) {
                if (k < 0)
                    continue;
                cv::Vec3b pixel_ik = rightImage.at<cv::Vec3b>(i, k);
                double dist = cv::norm(pixel_ij - pixel_ik);
                if (dist < min) {
                    col = k;
                    min = dist;
                }
            }
            if (max < std::abs(j - col))
                max = std::abs(j - col);
            dispMap.at<double>(i, j) = static_cast<double>(j - col);
        }
    }
    cv::Mat dispImg;
    cv::normalize(dispMap, dispImg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return dispImg;
}