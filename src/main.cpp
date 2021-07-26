
#include "PoseEstimation/keypoints.h"
#include "PoseEstimation/bundle_adjustment.h"
#include "PoseEstimation/pose_estimation.h"
#include "DataLoader/data_loader.h"
#include "Rectification/rectification.hpp"
#include "Reconstruction/reconstruction.h"
#include "utils.h"


#define RUN_ALL 1 // set 0 to only run reconstruction based on pre-computed disparity image (used for HitNet and PerceptualWindowSearch)

int main() {

    DataLoader dataLoader = DataLoader();
    std::string disparityPath = "../../results/disparity_map.png";

    // select scenarios by index (alphabetic position starting with 0)
    // for the final evaluations, we use  0, 5, 8, 12, 13
    int scenarioIdx = 13;
    Data data = dataLoader.loadTrainingScenario(scenarioIdx);

#if RUN_ALL
    /**
     * 1. Estimate Extrinsics (Fundamental Matrix)
     */

    // select scenarios by index (alphabetic position starting with 0)
    poseStruct estimatedPose = runFullPoseEstimation(data.getImageLeft(), data.getImageRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight(), true);
    Matrix3f fundamentalMatrix = estimatedPose.fundamentalMatrix;

    std::cout << "Estimated pose:  " << std::endl << estimatedPose.pose << std::endl;

    /**
     * 2. Compute Disparity Map
     */

    ImageRectifier rectifier = ImageRectifier(data.getImageLeft(), data.getImageRight(), fundamentalMatrix, estimatedPose.keypointsLeft, estimatedPose.keypointsRight);

    rectifier.computeDisparityMapRight(17, 0, 200, 0.9);
    cv::Mat disparityImageWrite = rectifier.getDisparityMapRight();
    cv::imwrite(disparityPath, disparityImageWrite);
#endif


    /**
     * 3. Reconstruct scene
     */

    cv::Mat disparityImage = DataLoader::readGrayscaleImageAsDisparityMap(disparityPath);

    // remove and replace outliers
    removeDisparityOutliers(disparityImage, 500, 1.5, 0.8);

    // create depth map
    float focalLength = data.getCameraMatrixRight()(0,0);
    float baseline = 1.f;  // due to normalization (extrinsics translation vector has length 1)
    cv::Mat depthValues = convertDisparityToDepth(disparityImage, focalLength, baseline);

    // intrinsics
    Matrix3f intrinsics = data.getCameraMatrixRight();

    // reconstruct geometry and compute triangular mesh
    float thrMesh = 1.f;
    reconstruction(data.getImageRight(), depthValues, intrinsics, thrMesh);
}

