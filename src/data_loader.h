//
// Created by tim on 07.07.21.
//

#ifndef STEREO_RECONSTRUCTION_DATALOADER_H
#define STEREO_RECONSTRUCTION_DATALOADER_H

#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include "Eigen.h"
#include "data.h"
#include "directory.h"



class DataLoader {
/**
 * Loads data from the middlebury dataset
 */
public:
    DataLoader();

    Data loadTrainingScenario(int scenarioIndex);

    Data loadTestScenario(int scenarioIndex);

    std::vector<Data> loadTrainingDataset();

    std::vector<Data> loadTestDataset();

private:
    std::vector<std::string> trainingScenarioPaths;
    std::vector<std::string> testScenarioPaths;

    static void readCameraMatrices(const std::string &scenarioPath, Matrix3f &cameraLeft, Matrix3f &cameraRight);
    static void loadDisparityMatrices(const std::string &scenarioPath, cv::Mat &dispLeft, cv::Mat &dispRight);
};


#endif //STEREO_RECONSTRUCTION_DATALOADER_H
