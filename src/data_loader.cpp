//
// Created by tim on 07.07.21.
//

#include "data_loader.h"
#include "imageLib.h"


DataLoader::DataLoader() {
    std::string trainingPath = getCurrentDirectory() + "/../../data/MiddEval3/trainingH";
    for (const auto& scenario : std::filesystem::directory_iterator(trainingPath)) {
        trainingScenarioPaths.emplace_back(scenario.path());
    }
    std::sort(trainingScenarioPaths.begin(), trainingScenarioPaths.end());

    std::string testPath = getCurrentDirectory() + "/../../data/MiddEval3/testH";
    for (const auto& scenario : std::filesystem::directory_iterator(testPath)) {
        testScenarioPaths.emplace_back(scenario.path());
    }
    std::sort(testScenarioPaths.begin(), testScenarioPaths.end());
}

std::vector<Data> DataLoader::loadTrainingDataset() {
    std::vector<Data> trainingDataset;
    for (int index=0; index < 15; index++) {
        trainingDataset.emplace_back(loadTrainingScenario(index));
    }
    return trainingDataset;
}

std::vector<Data> DataLoader::loadTestDataset() {
    std::vector<Data> testDataset;
    for (int index=0; index < 15; index++) {
        testDataset.emplace_back(loadTestScenario(index));
    }
    return testDataset;
}

Data DataLoader::loadTrainingScenario(int scenarioIndex) {
    if (scenarioIndex > 14 || scenarioIndex < 0) {
        throw std::runtime_error("Select scenario index in range of [0; 14]");
    }
    std::string scenarioPath = trainingScenarioPaths[scenarioIndex];

    // get images
    cv::Mat imageLeft = cv::imread(scenarioPath + "/im0.png", cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread(scenarioPath + "/im1.png", cv::IMREAD_COLOR);

    // get intrinsics
    Matrix3f cameraLeft, cameraRight;
    readCameraMatrices(scenarioPath, cameraLeft, cameraRight);

    // get disparity map
    cv::Mat dispLeft, dispRight;
    loadDisparityMatrices(scenarioPath, dispLeft, dispRight);

    // get disparity masks
    cv::Mat maskLeft = cv::imread(scenarioPath + "/mask0nocc.png", cv::IMREAD_UNCHANGED);
    cv::Mat maskRight = cv::imread(scenarioPath + "/mask1nocc.png", cv::IMREAD_UNCHANGED);

    cv::imshow("hi", maskLeft);

    Data scenario = Data(imageLeft, imageRight, cameraLeft, cameraRight, dispLeft, dispRight, maskLeft, maskRight);
    return scenario;
}

Data DataLoader::loadTestScenario(int scenarioIndex) {
    if (scenarioIndex > 14 || scenarioIndex < 0) {
        throw std::runtime_error("Select scenario index in range of [0; 14]");
    }
    std::string scenarioPath = testScenarioPaths[scenarioIndex];

    // get images
    cv::Mat imageLeft = cv::imread(scenarioPath + "/im0.png", cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread(scenarioPath + "/im1.png", cv::IMREAD_COLOR);

    // get intrinsics
    Matrix3f cameraLeft, cameraRight;
    readCameraMatrices(scenarioPath, cameraLeft, cameraRight);

    Data scenario = Data(imageLeft, imageRight, cameraLeft, cameraRight);
    return scenario;
}

void DataLoader::loadDisparityMatrices(const string &scenarioPath, cv::Mat& dispLeft, cv::Mat& dispRight) {
    // read map using SDK
    CFloatImage dispRawLeft, dispRawRight;
    ReadImageVerb(dispRawLeft, (scenarioPath + "/disp0GT.pfm").c_str(), 0);
    ReadImageVerb(dispRawRight, (scenarioPath + "/disp1GT.pfm").c_str(), 0);

    // resize cv::Mat (output matrices)
    dispLeft.create(dispRawLeft.Shape().height, dispRawLeft.Shape().width, CV_32FC1);
    dispRight.create(dispRawLeft.Shape().height, dispRawLeft.Shape().width, CV_32FC1);

    // copy data (float values, 4 byte)
    size_t sz = dispLeft.rows * dispLeft.cols * sizeof(float);
    memcpy(dispLeft.data, dispRawLeft.PixelAddress(0,0,0), sz);
    memcpy(dispRight.data, dispRawLeft.PixelAddress(0,0,0), sz);
}

void DataLoader::readCameraMatrices(const std::string &scenarioPath, Matrix3f &cameraLeft, Matrix3f &cameraRight) {
    std::ifstream calibration(scenarioPath + "/calib.txt");
    // read first two lines and extract the camera matrices
    std::string line;
    for (int i=0; i<2; i++) {
        std::getline(calibration, line);
        // replace useless characters
        line.replace(line.begin(), line.begin() + 6, "");
        line.replace(line.end() - 1, line.end(), "");
        std::replace(line.begin(), line.end(), ';', ' ');
        // read out float values
        std::istringstream floatValues(line);
        float xx, xy, xz, yx, yy, yz, zx, zy, zz;
        floatValues >> xx >> xy >> xz >> yx >> yy >> yz >> zx >> zy >> zz;
        // store intrinsics
        if (i == 0) {
            cameraLeft << xx, xy, xz, yx, yy, yz, zx, zy, zz;
        }
        else if (i == 1) {
            cameraRight << xx, xy, xz, yx, yy, yz, zx, zy, zz;
        }
    }
    calibration.close();
}


