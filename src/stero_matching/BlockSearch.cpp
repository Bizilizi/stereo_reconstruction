#include "BlockSearch.h"
#include <limits>
#include <iostream>

BlockSearch::BlockSearch(cv::Mat &leftImage, cv::Mat &rightImage, uint8_t blockSize)
        : leftImage(leftImage), rightImage(rightImage), blockSize(blockSize) {

}

cv::Mat BlockSearch::computeDisparityMap() {
    int h1 = leftImage.size().height;
    int w1 = leftImage.size().width;
    int h2 = rightImage.size().height;
    int w2 = rightImage.size().width;
    std::cout << "h1:" << h1 << "\n";
    std::cout << "h2:" << h2 << "\n";
    //assert(h1 == h2);
    int height = std::min(h1, h2);

    cv::Mat dispMap = cv::Mat::zeros(h1, w1, CV_64F);
    int num = 0;
    for (int i = (blockSize-1)/2; i < height-(blockSize-1)/2; i++) {
        for (int j = (blockSize-1)/2; j < w1-(blockSize-1)/2; j++) {
            if (leftImage.at<cv::Vec3b>(i, j) == cv::Vec3b{0, 0, 0})                 // skip black pixels which are at borders
                continue;

            cv::Mat win1 = leftImage(cv::Rect(j - (blockSize-1)/2, i - (blockSize-1)/2,  blockSize, blockSize));
            int col = 0;
            double min = std::numeric_limits<double>::max();
            for (int k = j-200; k < j; k++) {
                if (k < (blockSize-1)/2 || k >= w2-(blockSize-1)/2)
                    continue;
                cv::Mat win2 = rightImage(cv::Rect(k - (blockSize-1)/2, i - (blockSize-1)/2, blockSize, blockSize));
                double dist = cv::norm(win1 - win2);
                //std::cout << "win1: " << win1 << "\n";
                //std::cout << "win2: " << win2 << "\n";
                if (dist < min) {
                    col = k;
                    min = dist;
                }
            }
            if (j - col <= 0)
                std::cout << "negative\n";
            dispMap.at<double>(i, j) = static_cast<double>(j - col);
        }
    }
    return dispMap;
}