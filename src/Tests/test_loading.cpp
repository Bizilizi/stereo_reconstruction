
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "../DataLoader/data_loader.h"


void displayStereoImages(const cv::Mat &image1, const cv::Mat &image2){
    cv::Mat stackedImages;
    cv::hconcat(image1, image2, stackedImages);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", stackedImages);
}

void displaySingleImage(const cv::Mat &image1){
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image1);
}

int main(int argc, char** argv ) {

    DataLoader dataLoader = DataLoader();
    Data trainingData = dataLoader.loadTrainingScenario(11);
    Data testData = dataLoader.loadTestScenario(0);
    std::vector<Data> testDataset = dataLoader.loadTestDataset();

    //displayStereoImages(trainingData.getImageLeft(), trainingData.getImageRight());
    //displaySingleImage(testData.getImageRight());
    displaySingleImage(testDataset[14].getImageRight());
    cv::waitKey(0);

    //std::cout << trainingData.getCameraMatrixLeft() << std::endl << trainingData.getCameraMatrixRight() << std::endl;
    std::cout << testDataset[14].getCameraMatrixLeft() << std::endl << testDataset[14].getCameraMatrixRight() << std::endl;

    return 0;
}
