
#include "../Reconstruction/reconstruction.h"
#include "../DataLoader/data_loader.h"


bool test_reconstruction_01() {
    std::string rgbPath = "../../../exercise_1/exercise_1_src/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png";
    std::string depthPath = "../../../exercise_1/exercise_1_src/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";

    cv::Mat bgrImage = cv::imread(rgbPath, cv::IMREAD_COLOR);
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
    // perceptual window search/window search - test on right image!
//    std::string disparityPath = "../../results/PerceptualWindowSearch/teddy_pipl_disp.png";
    std::string disparityPath = "../../results/PerceptualWindowSearch/test_result.png";
    cv::Mat disparityImage8 = cv::imread(disparityPath, cv::IMREAD_GRAYSCALE);

    // conversion to float
    cv::Mat disparityImage = cv::Mat(disparityImage8.rows, disparityImage8.cols, CV_32FC1);
    for (int i=0; i < disparityImage.rows; i++) {
        for (int j=0; j < disparityImage.cols; j++) {
            disparityImage.at<float>(i, j) = (float) disparityImage8.at<uint8_t>(i, j);
        }
    }

    // ground truth
    DataLoader dataLoader = DataLoader();
    Data trainingData = dataLoader.loadTrainingScenario(13);
    cv::Mat image = trainingData.getImageRight();

    float focalLength = trainingData.getCameraMatrixRight()(0,0);
    float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)

    cv::Mat depthValues = convertDisparityToDepth(disparityImage, focalLength, baseline);
    // filter out large outliers naive way
    for (int i=0; i < depthValues.rows; i++) {
        for (int j=0; j < depthValues.cols; j++) {
            if (depthValues.at<float>(i, j) > 100) {
                depthValues.at<float>(i, j) = 0;
            }
        }
    }

    // intrinsics
    Matrix3f intrinsics = trainingData.getCameraMatrixRight();

    reconstruction(image, depthValues, intrinsics, 1);
    return true;
}


bool test_reconstruction_03(){
    /*
     * Testing with HitNet results
     * */

    int scenarioIdx = 13;

    DataLoader dataLoader = DataLoader();
    Data trainingData = dataLoader.loadTrainingScenario(scenarioIdx);
    cv::Mat image = trainingData.getImageLeft();

    cv::Mat disparityImage = dataLoader.loadTrainingDisparityHitNet(scenarioIdx);
//    cv::Mat disparityImage = trainingData.getDisparityLeft();
    scaleDisparityMap(disparityImage, 1);

    // std::cout << disparityImage << std::endl;

    float focalLength = trainingData.getCameraMatrixLeft()(0,0);
    float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)

    cv::Mat depthValues = convertDisparityToDepth(disparityImage, focalLength, baseline);

    // intrinsics
    Matrix3f intrinsics = trainingData.getCameraMatrixLeft();

    float thrMarchingSquares = 1;
    reconstruction(image, depthValues, intrinsics, thrMarchingSquares);
    return true;
}


bool test_reconstruction_04(){
    // perceptual window search/window search - test on right image
//    std::string disparityPath = "../../results/PerceptualWindowSearch/teddy_pipl_disp.png";
    std::string disparityPath = "../../results/PerceptualWindowSearch/test_result.png";
    cv::Mat disparityImage = DataLoader::readGrayscaleImageAsDisparityMap(disparityPath);

    removeDisparityOutliers(disparityImage, 500, 1.5, 0.8);
    //scaleDisparityMap(disparityImage, 10);

    // ground truth
    DataLoader dataLoader = DataLoader();
    Data trainingData = dataLoader.loadTrainingScenario(13);
    cv::Mat image = trainingData.getImageRight();

    float focalLength = trainingData.getCameraMatrixRight()(0,0);
    float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)

    cv::Mat depthValues = convertDisparityToDepth(disparityImage, focalLength, baseline);

    // intrinsics
    Matrix3f intrinsics = trainingData.getCameraMatrixRight();

    reconstruction(image, depthValues, intrinsics, 1);
    return true;
}


int main(){
    test_reconstruction_04();
    return 0;
}
