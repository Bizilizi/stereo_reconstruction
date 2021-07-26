
#include "../PoseEstimation/pose_estimation.h"
#include "../PoseEstimation/keypoints.h"
#include "../DataLoader/data_loader.h"
#include "../Eigen.h"


void testVisualizationExtrinsics() {
    Matrix3Xf leftPoints, rightPoints;
    leftPoints = Matrix3f::Zero(3, 3);
    leftPoints.col(0) = Vector3f(1, 0, 0);
    leftPoints.col(1) = Vector3f(0, 1, 0);
    leftPoints.col(0) = Vector3f(0, 0, 1);

    showExtrinsicsReconstruction("8pt_reconstruction_test.off", Matrix4f::Identity(), leftPoints, rightPoints);
}


void testCaseExtrinsics() {
    // matching keypoint pairs in pixel coordinates
    MatrixXf kpLeftMat(3, 12), kpRightMat(3, 12);

    kpLeftMat << 10.0, 92.0, 8.0, 92.0, 289.0, 354.0, 289.0, 353.0, 69.0, 294.0, 44.0, 336.0, // x
            232.0, 230.0, 334.0, 333.0, 230.0, 278.0, 340.0, 332.0, 90.0, 149.0, 475.0, 433.0, //y
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; // z

    kpRightMat << 123.0, 203.0, 123.0, 202.0, 397.0, 472.0, 398.0, 472.0, 182.0, 401.0, 148.0, 447.0, // x
            239.0, 237.0, 338.0, 338.0, 236.0, 286.0, 348.0, 341.0, 99.0, 153.0, 471.0, 445.0, // y
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; // z

    // camera intrinsics
    Matrix3f cameraLeft, cameraRight;
    cameraLeft << 844.310547, 0, 243.413315, 0, 1202.508301, 281.529236, 0, 0, 1;
    cameraRight << 852.721008, 0, 252.021805, 0, 1215.657349, 288.587189, 0, 0, 1;

    EightPointAlgorithm ep(kpLeftMat, kpRightMat, cameraLeft, cameraRight);
    std::cout << "POSE: " << std::endl << ep.getPose() << std::endl;

    // check results
    Matrix4f referencePose = Matrix4f::Identity();
    referencePose(seqN(0, 3), seqN(0, 3)) << 0.9911, -0.0032, 0.1333,
            0.0032, 1.0, 0.0,
            -0.1333, 0.0004, 0.9911;
    referencePose(seqN(0, 3), 3) << -0.4427, -0.0166, 0.8965;
    float err = (ep.getPose() - referencePose).norm();
    std::cout << "Error norm: " << err << std::endl;

    Matrix3f referencePoints3D;
    referencePoints3D << -5.7313, -5.0535, -7.0558,
            -0.8539, -1.2075, 1.1042,
            20.7315, 28.1792, 25.3056;
    std::cout << "Reconstructed 3d points:" << std::endl << ep.getPointsLeftReconstructed() << std::endl;
    std::cout << "Error norm: " << (ep.getPointsLeftReconstructed()(seqN(0, 3), seqN(0, 3)) - referencePoints3D).norm()
              << std::endl;

    std::cout << "Fundamental Matrix" << ep.getFundamentalMatrix() << std::endl;
}

void testPoseEstimation(const std::vector<int> &scenarioIdx, int nRuns=5){
    int nScenarios = scenarioIdx.size();
    VectorXf error8pt = VectorXf::Zero(nScenarios*nRuns);
    VectorXf errorBA = VectorXf::Zero(nScenarios*nRuns);

    DataLoader dataLoader = DataLoader();

    for(int i = 0; i< nScenarios; i++){
        for (int n= 0; n<nRuns; n++){
            Data data = dataLoader.loadTrainingScenario(scenarioIdx.at(i));
            poseStruct estimatedPose = runFullPoseEstimation(data.getImageLeft(), data.getImageRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight(), false);
            error8pt(i*nScenarios + n) = estimatedPose.reError8pt;
            errorBA(i*nScenarios + n) = estimatedPose.reErrorBA;
        }
    }

    VectorXf centeredError8pt = error8pt.array() - error8pt.mean();
    float std8pt = std::sqrt(centeredError8pt.dot(centeredError8pt) / error8pt.size());

    VectorXf centeredErrorBA = errorBA.array() - errorBA.mean();
    float stdBA = std::sqrt(centeredErrorBA.dot(centeredErrorBA) / errorBA.size());

    std::cout << "Average error / std 8pt: " << error8pt.mean() << "   " <<  std8pt << std::endl;
    std::cout << "Average error / std Bundle Adjustment: " << errorBA.mean() << "  " << stdBA << std::endl;
}



int main() {
    DataLoader dataLoader = DataLoader();

    // select scenarios by index (alphabetic position starting with 0)
    Data data = dataLoader.loadTrainingScenario(9);
    runFullPoseEstimation(data.getImageLeft(), data.getImageRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight(), true);

}
