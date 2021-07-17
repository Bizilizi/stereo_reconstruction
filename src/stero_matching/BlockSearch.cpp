#include "BlockSearch.h"
#include <limits>
#include <iostream>

BlockSearch::BlockSearch(cv::Mat &leftImage,
						 cv::Mat &rightImage,
						 int blockSize,
						 int maxDisparity)
	: leftImage_(leftImage), rightImage_(rightImage), blockSize_(blockSize), maxDisparity_(maxDisparity) {

}

cv::Mat BlockSearch::computeDisparityMap() {
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
				disparityMap.at<double>(pointY, pointX) = INFINITY;
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

				double dist = cv::norm(leftWindow - rightWindow);

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