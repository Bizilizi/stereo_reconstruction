//
// Created by tim on 30.06.21.
//

#include <iostream>
#include "cassert"

#include "Eigen.h"
#include "BundleAdjustment.h"

float averageReconstructionError(const Matrix3Xf& matchesLeft, const Matrix3Xf& matchesRight,
                                 const Matrix3f& intrinsicsLeft, const Matrix3f& intrinsicsRight,
                                 const Matrix3f& rotation, const Vector3f& translation,
                                 const Matrix3Xf& reconstructedPointsLeft){
    int nPoints = (int) reconstructedPointsLeft.cols();

    // projection error left picture
    Matrix3Xf projectedPointsLeft;
    projectedPointsLeft = (intrinsicsLeft * reconstructedPointsLeft).cwiseQuotient(reconstructedPointsLeft.row(2).replicate(3,1));

    VectorXf errorsLeft;
    errorsLeft = (projectedPointsLeft - matchesLeft).colwise().norm();
    std::cout << "Error left: " << errorsLeft.sum() << std::endl;

    // projection error right picture
    Matrix3Xf translatedPoints, projectedPointsRight;
    translatedPoints = rotation * reconstructedPointsLeft + translation.replicate(1, nPoints);
    projectedPointsRight = (intrinsicsRight * translatedPoints).cwiseQuotient(translatedPoints.row(2).replicate(3,1));
    VectorXf errorsRight = (projectedPointsRight - matchesRight).colwise().norm();
    std::cout << "Errors right: " << errorsRight.sum() << std::endl;

    return (errorsLeft.sum() + errorsRight.sum()) / (float) nPoints;
}

void testCase01() {
    /**
     *  rotation around z-axis
     *  and translation
     */

    // Problem definition

    Matrix3f intrinsics = Matrix3f::Identity();
    Matrix3Xf matchesLeft{3, 9}, matchesRight{3, 9};        // pixel coordinates
    Matrix3Xf pointsLeftNorm{3,9}, pointsRightNorm{3,9};    // inverse kinematics applied
    Matrix3Xf pointsLeft3D{3,9}, pointsRight3D{3,9}, noisyPointsLeft3D{3,9};        // 3D points relative to coordinate frame
    VectorXf depthVector{9}, noiseVector{9}, noisyDepthVector{9};
    Matrix3f targetRotation, initRotation;
    Vector3f targetTranslation, initTranslation;

    targetTranslation << 0, 0, 5;
    initTranslation << 0, 0, 0;
    targetRotation = AngleAxisf(30*M_PI/180, Vector3f(0, 0, 1).normalized());
    initRotation = AngleAxisf(0*M_PI/180, Vector3f(0, 0, 1).normalized());

    matchesLeft << 0, 0, 0, 250, 250, 250, 500, 500, 500,
            0, 250, 500, 0, 250, 500, 0, 250, 500,
            1, 1, 1, 1, 1, 1, 1, 1, 1;
    depthVector << 10, 10, 10, 20, 20, 20, 30, 30, 30;

    noiseVector.setRandom();
    noisyDepthVector = depthVector + noiseVector * 5;

    pointsLeftNorm = intrinsics.inverse() * matchesLeft;
    pointsLeft3D = pointsLeftNorm.cwiseProduct(depthVector.transpose().replicate(3, 1));
    noisyPointsLeft3D = pointsLeftNorm.cwiseProduct(noisyDepthVector.transpose().replicate(3, 1));


    pointsRight3D = targetRotation * pointsLeft3D + targetTranslation.replicate(1, pointsRight3D.cols());
    pointsRightNorm = (intrinsics * pointsRight3D).cwiseQuotient(pointsRight3D.row(2).replicate(3, 1));
    matchesRight = intrinsics * pointsRightNorm;

    // Running optimization
    float perfectCost = averageReconstructionError(matchesLeft, matchesRight, intrinsics, intrinsics, targetRotation, targetTranslation, pointsLeft3D);
    assert(perfectCost < 1e-5);

    float initialCost = averageReconstructionError(matchesLeft, matchesRight, intrinsics, intrinsics, initRotation, initTranslation, noisyPointsLeft3D);
    std::cout << "Initial bundle adjustment costs: " << initialCost << std::endl;

    Matrix4f pose;
    auto optimizer = BundleAdjustmentOptimizer(matchesLeft, matchesRight, intrinsics, intrinsics, initRotation, initTranslation, pointsLeft3D);
    pose = optimizer.estimatePose();
    std::cout << "Final estimated pose: " << std::endl << pose << std::endl;
    std::cout << "Reference rotation: " << std::endl << targetRotation <<  std::endl;
    float finalCost = averageReconstructionError(matchesLeft, matchesRight, intrinsics, intrinsics, pose(seqN(0,3), seqN(0,3)), pose(seqN(0,3), 3), pointsLeft3D);
    std::cout << "Final bundle adjustment costs: " << finalCost << std::endl;
}

int main(int argc, char **argv) {
    testCase01();
    return 0;
}
