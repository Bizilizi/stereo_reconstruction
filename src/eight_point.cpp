//
// Created by gabriel on 21.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "Eigen.h"


void eightPointAlgorithm(const std::vector<cv::KeyPoint>& keypointsLeft, const std::vector<cv::KeyPoint>& keypointsRight, const std::vector<cv::DMatch>& matches, const Matrix3f& cameraLeft, const Matrix3f& cameraRight, Matrix4f& pose) {
    // get all matched keypoint pairs and prepare them for using Eigen in the eight point algorithm
    std::vector<Vector3f> matchesLeft, matchesRight;
    int numMatches = 0;
    for (cv::DMatch match : matches) {
        matchesLeft.emplace_back(Vector3f(keypointsLeft[match.queryIdx].pt.x, keypointsLeft[match.queryIdx].pt.y, 1));
        matchesRight.emplace_back(Vector3f(keypointsRight[match.trainIdx].pt.x, keypointsRight[match.trainIdx].pt.y, 1));
        numMatches++;
        //std::cout << Vector3f(keypointsLeft[match.queryIdx].pt.x, keypointsLeft[match.queryIdx].pt.y, 1) << std::endl;
        //std::cout << Vector3f(keypointsRight[match.trainIdx].pt.x, keypointsRight[match.trainIdx].pt.y, 1) << std::endl;
    }
    if (numMatches < 8) {
        throw std::runtime_error("Less than 8 input point pairs detected for processing the Eight Point Algorithm!");
    }

    // transform image coordinates with inverse camera matrix (inverse intrinsics)
    std::vector<Vector3f> pointsLeft, pointsRight;
    for (int i=0; i < numMatches; i++) {
        pointsLeft.emplace_back(cameraLeft.inverse() * matchesLeft[i]);
        pointsRight.emplace_back(cameraRight.inverse() * matchesRight[i]);
    }

    /** Eight Point Algorithm **/
    // compute approximation of essential matrix
    MatrixXf chi(numMatches, 9);
    for (int i=0; i < numMatches; i++) {
        chi.block(i, 0, 1, 3) = Vector3f(pointsLeft[i].x() * pointsRight[i].x(), pointsLeft[i].x() * pointsRight[i].y(), pointsLeft[i].x() * 1).transpose();
        chi.block(i, 3, 1, 3) = Vector3f(pointsLeft[i].y() * pointsRight[i].x(), pointsLeft[i].y() * pointsRight[i].y(), pointsLeft[i].y() * 1).transpose();
        chi.block(i, 6, 1, 3) = Vector3f(1 * pointsRight[i].x(), 1 * pointsRight[i].y(), 1 * 1).transpose();
    }
    JacobiSVD<MatrixXf> svdChi(chi, ComputeThinV);
    // extract essential matrix from last column of matrix V
    Matrix3f essentialMatrix = svdChi.matrixV().block(0, 8, 9, 1).reshaped(3, 3);

    // project onto normalized essential space using SVD
    JacobiSVD<Matrix3f> svdEssential(essentialMatrix, ComputeFullU | ComputeFullV);
    // correct values of SVD
    MatrixXf matrixU = svdEssential.matrixU();
    MatrixXf matrixV = svdEssential.matrixV();
    if (matrixU.determinant() < 0) {
        matrixU = -matrixU;
    }
    if (matrixV.determinant() < 0) {
        matrixV = -matrixV;
    }
    Matrix3f matrixSigma = Matrix3f::Identity();
    matrixSigma(2, 2) = 0;

    // recover displacement from essential matrix
    Matrix3f zRotation1 = Matrix3f::Zero();
    zRotation1(0, 1) = -1;
    zRotation1(1, 0) = 1;
    zRotation1(2, 2) = 1;
    Matrix3f zRotation2 = zRotation1.transpose();

    Matrix3f rotation1 = matrixU * zRotation1.transpose() * matrixV.transpose();
    Matrix3f rotation2 = matrixU * zRotation2.transpose() * matrixV.transpose();

    Matrix3f skewSymmetricT1 = matrixU * zRotation1 * matrixSigma * matrixU.transpose();
    Matrix3f skewSymmetricT2 = matrixU * zRotation2 * matrixSigma * matrixU.transpose();
    Vector3f translation1 = Vector3f(-skewSymmetricT1(1, 2), skewSymmetricT1(0, 2), -skewSymmetricT1(0, 1));
    Vector3f translation2 = Vector3f(-skewSymmetricT2(1, 2), skewSymmetricT2(0, 2), -skewSymmetricT2(0, 1));

    // TODO find the correct combination of rotation and translation - structure reconstruction???
    //std::cout << rotation1 << std::endl << std::endl;
    //std::cout << rotation2 << std::endl << std::endl;
    //std::cout << translation1 << std::endl << std::endl;
    //std::cout << translation2 << std::endl << std::endl;

    // return estimated pose
    pose = Matrix4f::Identity();
    pose.block(0, 0, 3, 3) = rotation1;
    pose.block(0, 3, 3, 1) = translation1;
}
