
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
    /*
    std::string rgbPath = "../../results/rectifiedImage.jpg";
    std::string disparityPath = "../../results/rectifiedDepth.jpg";

    cv::Mat bgrImage = cv::imread(rgbPath, cv::IMREAD_COLOR); // TODO: Fix color issue!
    cv::Mat disparityImage = cv::imread(disparityPath, cv::IMREAD_GRAYSCALE);
*/
    // convert disparity image to depth map
    DataLoader dataLoader = DataLoader();
    Data teddyData = dataLoader.loadTrainingScenario(3);

    cv::Mat disparityImage = teddyData.getDisparityLeft();

    // std::cout << disparityImage << std::endl;
    std::cout << "Rows/ cols: " << disparityImage.rows << "  " << disparityImage.cols << std::endl;

    int count = 0, countWhite = 0;

    for(int row = 0; row < disparityImage.rows; row++){
        for(int col = 0; col < disparityImage.cols; col++){
            if(disparityImage.at<float>(row, col) < 10) {
                count++;
            }
            if(disparityImage.at<float>(row, col) > 250) {
                countWhite++;
            }
        }
    }

    std::cout << "CountBlack: " << count << std::endl;
    std::cout << "CountWhite: " << countWhite << std::endl;

    cv::Mat bgrImage = teddyData.getImageLeft();
    cv::imshow("test", bgrImage);

    cv::imshow("DisparityLeft", disparityImage);

    cv::waitKey();

    float focalLength = teddyData.getCameraMatrixLeft()(0,0);
    float baseline = 0.01;  // due to normalization (extrinsics translation vector has length 1)

    cv::Mat depthValues = cv::Mat(disparityImage.rows, disparityImage.cols, CV_32F);
    for (int h = 0; h < disparityImage.rows; h++) {
        for (int w = 0; w < disparityImage.cols; w++) {
            if (disparityImage.at<u_int8_t>(h, w) == 0) {
                // no depth assigned
                depthValues.at<float>(h, w) = MINF;
            } else {
                depthValues.at<float>(h, w) = focalLength * baseline / (float) disparityImage.at<u_int8_t>(h, w);
            }
        }
    }

    // intrinsics
    Matrix3f intrinsics = teddyData.getCameraMatrixLeft();
    // Matrix3f intrinsics = Matrix3f::Identity();

    reconstruction(bgrImage, depthValues, intrinsics);
    return true;
}


int main(){
    test_reconstruction_02();
    return 0;
}
