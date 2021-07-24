#include "BlockSearch.h"
#include <limits>
#include <iostream>
#include "opencv4/opencv2/imgproc.hpp"

BlockSearch::BlockSearch(cv::Mat &leftImage,
						 cv::Mat &rightImage,
						 int blockSize,
						 int minDisparity,
						 int maxDisparity)
	: leftImage_(leftImage), rightImage_(rightImage), blockSize_(blockSize), minDisparity_(minDisparity), maxDisparity_(maxDisparity) {

}

BlockSearch::BlockSearch(cv::Mat &leftImage,
                         cv::Mat &rightImage,
                         std::vector<int> &blockSizes,
                         int minDisparity,
                         int maxDisparity)
    : leftImage_(leftImage), rightImage_(rightImage), blockSizes_(blockSizes), minDisparity_(minDisparity), maxDisparity_(maxDisparity) {

}

cv::Mat BlockSearch::computeDisparityMapLeft(double smoothFactor) {
	int h1 = leftImage_.size().height;
	int w1 = leftImage_.size().width;
	int h2 = rightImage_.size().height;
	int w2 = rightImage_.size().width;

	int height = std::min(h1, h2);
	int halfBlockSize = (blockSize_ - 1) / 2;

	cv::Mat disparityMap = cv::Mat::zeros(h1, w1, CV_64F);

	// Iterate over height
	for (int pointY = halfBlockSize; pointY < height - halfBlockSize; ++pointY) {
		// Iterate over width
		for (int pointX = halfBlockSize; pointX < w1 - halfBlockSize; ++pointX) {

			// Skip black pixels which are at borders e.g have black color
			if (leftImage_.at<cv::Vec3b>(pointY, pointX) == cv::Vec3b{0, 0, 0}) {
				//disparityMap.at<double>(pointY, pointX) = INFINITY;
				continue;
			}

			cv::Mat leftWindow = leftImage_(cv::Rect(pointX - halfBlockSize,
													 pointY - halfBlockSize,
													 blockSize_,
													 blockSize_));
			int minimumCorrespondX = 0;
			double min = std::numeric_limits<double>::max();

			for (int correspondX = pointX - maxDisparity_; correspondX < pointX; correspondX++) {

				if (correspondX < halfBlockSize || correspondX >= w2 - halfBlockSize) {
					continue;
				}

				cv::Mat rightWindow = rightImage_(cv::Rect(correspondX - halfBlockSize,
														   pointY - halfBlockSize,
														   blockSize_,
														   blockSize_));

				cv::Mat diff;
				cv::absdiff(leftWindow, rightWindow, diff);
				double dist = cv::norm(diff, cv::NORM_L2);

                if (pointY >= 1 && disparityMap.at<double>(pointY-1, pointX) == static_cast<double>(pointX-correspondX)) {
                    dist *= smoothFactor;
                }
                if (pointX >= 1 && disparityMap.at<double>(pointY, pointX-1) == static_cast<double>(pointX-correspondX)) {
                    dist *= smoothFactor;
                }

				// update minimum if found
				if (dist < min) {
					minimumCorrespondX = correspondX;
					min = dist;
				}
			}

			disparityMap.at<double>(pointY, pointX) = static_cast<double>(pointX - minimumCorrespondX);
		}
	}
	return disparityMap;
}

cv::Mat BlockSearch::computeDisparityMapRight(double smoothFactor, bool gaussianBlur) {
    int h1 = leftImage_.size().height;
    int w1 = leftImage_.size().width;
    int h2 = rightImage_.size().height;
    int w2 = rightImage_.size().width;

    cv::Mat leftBlur = leftImage_;
    cv::Mat rightBlur = rightImage_;

    if (gaussianBlur) {
        cv::GaussianBlur(leftImage_, leftBlur, cv::Size(5, 5), 0, 0);
        cv::GaussianBlur(rightImage_, rightBlur, cv::Size(5, 5), 0, 0);
    }

    int height = std::min(h1, h2);
    int halfBlockSize = (blockSize_ - 1) / 2;

    cv::Mat disparityMap = cv::Mat::zeros(h2, w2, CV_64F);

    // Iterate over height
    for (int pointY = halfBlockSize; pointY < height - halfBlockSize; ++pointY) {
        // Iterate over width
        for (int pointX = halfBlockSize; pointX < w2 - halfBlockSize; ++pointX) {

            // Skip black pixels which are at borders e.g have black color
            if (rightImage_.at<cv::Vec3b>(pointY, pointX) == cv::Vec3b{0, 0, 0}) {
                //disparityMap.at<double>(pointY, pointX) = INFINITY;
                continue;
            }

            cv::Mat rightWindow = rightBlur(cv::Rect(pointX - halfBlockSize,
                                                     pointY - halfBlockSize,
                                                     blockSize_,
                                                     blockSize_));
            int minimumCorrespondX = 0;
            double min = std::numeric_limits<double>::max();

            for (int correspondX = pointX+minDisparity_; correspondX < pointX + maxDisparity_; correspondX++) {

                if (correspondX < halfBlockSize || correspondX >= w1 - halfBlockSize) {
                    continue;
                }

                cv::Mat leftWindow = leftBlur(cv::Rect(correspondX - halfBlockSize,
                                                           pointY - halfBlockSize,
                                                           blockSize_,
                                                           blockSize_));

                cv::Mat diff;
                cv::absdiff(leftWindow, rightWindow, diff);
                double dist = cv::norm(diff, cv::NORM_L2) / std::pow(blockSize_, 2);

                if (pointY >= 1 && disparityMap.at<double>(pointY-1, pointX) == static_cast<double>(pointX-correspondX)) {
                    dist *= smoothFactor;
                }
                if (pointX >= 1 && disparityMap.at<double>(pointY, pointX-1) == static_cast<double>(pointX-correspondX)) {
                    dist *= smoothFactor;
                }

                // update minimum if found
                if (dist < min) {
                    minimumCorrespondX = correspondX;
                    min = dist;
                }
            }

            disparityMap.at<double>(pointY, pointX) = static_cast<double>(minimumCorrespondX - pointX);
        }
    }
    return disparityMap;
}