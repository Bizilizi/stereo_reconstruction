//
// Created by gabriel on 21.06.21.
//

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "Eigen.h"
#include "eight_point.h"
#include <algorithm>

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

    // FiXME: Mabye use this matrix notation in this function as well
    MatrixXf xLeft = MatrixXf::Zero(3, pointsLeft.size());
    MatrixXf xRight = MatrixXf::Zero(3, pointsRight.size());
    for (int i=0; i < pointsLeft.size(); i++){
        xLeft.col(i) = pointsLeft.at(i);
        xRight.col(i) = pointsRight.at(i);
    }
    // std::cout << xLeft << std::endl;

    MatrixXf xLeftReconstructed, xRightReconstructed;
    Matrix3f validRotation;
    Vector3f validTranslation;
    std::vector<Matrix3f> possibleRotations{rotation1, rotation2};
    std::vector<VectorXf> possibleTranslations;
    possibleTranslations.emplace_back(translation1);
    possibleTranslations.emplace_back(translation1);

    bool success = false;
    for (auto &rotation : possibleRotations ) {
        for (auto &translation : possibleTranslations) {
            if(structureReconstruction(rotation, translation, xLeft, xRight, xLeftReconstructed, xRightReconstructed)){
                validRotation = rotation;
                validTranslation = translation;
                success = true;
                break;
            }
        }
    }
    if(success) {
        throw std::runtime_error("Depth reconstruction failed.");
    }

    // return estimated pose
    pose = Matrix4f::Identity();
    pose.block(0, 0, 3, 3) = validRotation;
    pose.block(0, 3, 3, 1) = validTranslation;
}

bool structureReconstruction(const Matrix3f& R, const Vector3f& T, const MatrixXf& xLeft, const MatrixXf& xRight,
                             MatrixXf& xLeftReconstructed, MatrixXf& xRightReconstructed) {
    int nPoints = (int) xLeft.cols();
    MatrixXf M = MatrixXf::Zero(3*nPoints, nPoints + 1);

    // fill M matrix
    for (int i=0; i<nPoints; i++){
        M(seqN(3*i,3), i) = vectorAsSkew(xRight.col(i)) * R * xLeft.col(i);
        M(seqN(3*i,3), nPoints) = vectorAsSkew(xRight.col(i)) * T;
    }

    // get vector V corresponding to smallest eigenvalue of M'*M (least squares estimate)
    EigenSolver<MatrixXf> es(M.transpose() * M);
    auto& eigenvalues = es.eigenvalues();
    int idxSmallestEigenvalue = 0;
    float smallestEigenvalue = eigenvalues[0].real();
    for (int i=1; i < nPoints +1; i++) {
        if (eigenvalues[i].real() < smallestEigenvalue){
            smallestEigenvalue = eigenvalues[i].real();
            idxSmallestEigenvalue = i;
        }
    }
    VectorXf V = es.eigenvectors().col(idxSmallestEigenvalue).real();

    // reconstruct depth
    float scale = V(last);
    VectorXf depthVec = V(seq(0, last-1)) / scale; // make scale similar to scale of translation
    xLeftReconstructed = xLeft.cwiseProduct(depthVec.transpose().replicate(3, 1));
    xRightReconstructed = (R * xLeftReconstructed) + T.replicate(1, nPoints);

#if 0
    // Some logging
    std::cout << "Eigenvalues:" << std::endl;
    std::cout << eigenvalues << std::endl;
    std::cout << "Smallest eigenvalue and its index " << smallestEigenvalue << "   " << idxSmallestEigenvalue << std::endl;
    std::cout << "Vector corresponding to smallest eigenvalue:" << V << std::endl;

    std::cout << "Reconstructed points left:" << std::endl;
    std::cout << xLeftReconstructed << std::endl;
    std::cout << "Reconstructed points right:" << std::endl;
    std::cout << xLeftReconstructed << std::endl;
#endif
    // check depth of all reconstructed points
    bool success = (xLeftReconstructed.row(2).array() >= 0).all() && (xRightReconstructed.row(2).array() >=0).all();
    return success;
}

Matrix3f vectorAsSkew(const Vector3f &vec) {
    Matrix3f skewMatrix = Matrix3f::Zero();
    // upper triangular matrix
    skewMatrix(0,1) = -vec.z();
    skewMatrix(0, 2) = vec.y();
    skewMatrix(1, 2) = -vec.x();
    // lower triangular matrix
    skewMatrix(1, 0) = vec.z();
    skewMatrix(2, 0) = -vec.y();
    skewMatrix(2, 1) = vec.x();
    return skewMatrix;
}

void transformMatchedKeypointsToEigen(const std::vector<cv::KeyPoint>& keypointsLeft,
                                      const std::vector<cv::KeyPoint>& keypointsRight,
                                      const std::vector<cv::DMatch>& matches,
                                      MatrixXf& outLeft,
                                      MatrixXf& outRight) {
    outLeft = MatrixXf::Zero(3, matches.size());
    outRight = MatrixXf::Zero(3, matches.size());

    int i = 0;
    for (cv::DMatch match : matches) {
        outLeft.col(i) = Vector3f(keypointsLeft[match.queryIdx].pt.x, keypointsLeft[match.queryIdx].pt.y, 1);
        outRight.col(i) = Vector3f(keypointsRight[match.trainIdx].pt.x, keypointsRight[match.trainIdx].pt.y, 1);
        i++;
    }
}

