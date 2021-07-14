
#include "reconstruction.h"
#include "data_loader.h"


bool test_reconstruction_01() {
    std::string rgbPath = "../../../exercise_1/exercise_1_src/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png";
    std::string depthPath = "../../../exercise_1/exercise_1_src/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";

    cv::Mat bgrImage = cv::imread(rgbPath, cv::IMREAD_COLOR); // TODO: Fix color issue!
    cv::Mat depthImage = cv::imread(depthPath, cv::IMREAD_UNCHANGED); // 16 bit grayscale, scaled by 5000

    cv::Mat depthValues = cv::Mat(depthImage.rows, depthImage.cols, CV_32F);
    for (int h = 0; h < depthImage.rows; h++) {
        for (int w = 0; w < depthImage.cols; w++) {
            if (depthImage.at<u_int16_t>(h, w) == 0) {
                depthValues.at<float>(h, w) = MINF;
            } else {
                depthValues.at<float>(h, w) = (float) depthImage.at<u_int16_t>(h, w) / 5000.f;
            }
        }
    }

    // intrinsics
    Matrix3f intrinsics;
    intrinsics << 525.0f, 0.0f, 319.5f,
            0.0f, 525.0f, 239.5f,
            0.0f, 0.0f, 1.0f;

    reconstruction(bgrImage, depthValues, intrinsics);
    return true;
}


bool test_reconstruction_02(){
    // our approaches
    /*
    //std::string rgbPath = "../../data/rectifiedImage.jpg";
    //cv::Mat bgrImage = cv::imread(rgbPath, cv::IMREAD_COLOR);
    //std::string disparityPath = "../../data/windowSearch2.jpg";
    std::string disparityPath = "../../data/revertTeddyDisp4.png";
    cv::Mat disparityImage16 = cv::imread(disparityPath, cv::IMREAD_UNCHANGED);

    // dirty conversion
    cv::Mat disparityImage = cv::Mat(disparityImage16.rows, disparityImage16.cols, CV_32FC1);
    for (int i=0; i < disparityImage.rows; i++) {
        for (int j=0; j < disparityImage.cols; j++) {
            disparityImage.at<float>(i, j) = (float) disparityImage16.at<uint16_t>(i, j) / 5000.f;
        }
    }
    */

    // ground truth
    DataLoader dataLoader = DataLoader();
    Data trainingData = dataLoader.loadTrainingScenario(3);
    cv::Mat bgrImage = trainingData.getImageLeft();
    cv::Mat disparityImage = trainingData.getDisparityLeft();

    float focalLength = trainingData.getCameraMatrixLeft()(0,0);
    // TODO read also baseline [mm] from calib.txt
    //float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)
    float baseline = 0.193001f;  // Motorcycle
    //float baseline = 0.080f;  // Teddy

    cv::Mat depthValues = cv::Mat(disparityImage.rows, disparityImage.cols, CV_32FC1);
    for (int h = 0; h < disparityImage.rows; h++) {
        for (int w = 0; w < disparityImage.cols; w++) {
            if (disparityImage.at<float>(h, w) == 0) {
                // no depth assigned
                depthValues.at<float>(h, w) = MINF;
            } else {
                depthValues.at<float>(h, w) = focalLength * baseline / disparityImage.at<float>(h, w);
            }
        }
    }

    // intrinsics
    Matrix3f intrinsics = trainingData.getCameraMatrixLeft();

    reconstruction(bgrImage, depthValues, intrinsics);
    return true;
}


int main(){
    test_reconstruction_02();
    return 0;
}
