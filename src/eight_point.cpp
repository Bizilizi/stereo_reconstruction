//
// Created by gabriel on 21.06.21.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "eight_point.h"


EightPointAlgorithm::EightPointAlgorithm(const Matrix3Xf &matchesL, const Matrix3Xf &matchesR,
                                         const Matrix3f &cameraLeft, const Matrix3f &cameraRight)
        : cameraLeft{cameraLeft}, cameraRight{cameraRight} {
    // get all matched keypoint pairs and prepare them for using Eigen in the eight point algorithm

    setMatches(matchesL, matchesR);

    pointsLeftReconstructed = MatrixXf::Zero(3, numMatches);
    pointsRightReconstructed = MatrixXf::Zero(3, numMatches);

    run();
}

void EightPointAlgorithm::run() {
    updateData();
    /** Eight Point Algorithm **/
    // compute approximation of essential matrix
    MatrixXf chi(numMatches, 9);
    for (int i = 0; i < numMatches; i++) {
        chi(i, seqN(0, 9)) = kron(pointsLeft.col(i), pointsRight.col(i));
    }

    JacobiSVD<MatrixXf> svdChi(chi, ComputeThinV);
    // extract essential matrix from last column of matrix V
    Matrix3f essMatrix = svdChi.matrixV().block(0, 8, 9, 1).reshaped(3, 3);

    // project essential matrix onto normalized essential space using SVD
    JacobiSVD<Matrix3f> svdEssential(essMatrix, ComputeFullU | ComputeFullV);
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

    // check which combination of translation and rotation is valid (assigns positive depth to all points)
    MatrixXf xLeftReconstructed, xRightReconstructed;
    Matrix3f validRotation;
    Vector3f validTranslation;
    std::vector<Matrix3f> possibleRotations{rotation1, rotation2};
    std::vector<VectorXf> possibleTranslations;
    possibleTranslations.emplace_back(translation1);
    possibleTranslations.emplace_back(translation2);

    bool success = false;
    for (auto &rotation : possibleRotations) {
        for (auto &translation : possibleTranslations) {
            if (structureReconstruction(rotation, translation)) {
                validRotation = rotation;
                validTranslation = translation;
                success = true;
                break;
            }
        }
    }
    if (!success) {
        throw std::runtime_error("Depth reconstruction failed.");
    }

    // set essential matrix
    essentialMatrix = vectorAsSkew(validTranslation)*validRotation;

    // return estimated pose
    pose = Matrix4f::Identity();
    pose.block(0, 0, 3, 3) = validRotation;
    pose.block(0, 3, 3, 1) = validTranslation;
}

bool EightPointAlgorithm::structureReconstruction(const Matrix3f &R, const Vector3f &T) {
    MatrixXf M = MatrixXf::Zero(3 * numMatches, numMatches + 1);

    // fill M matrix
    for (int i = 0; i < numMatches; i++) {
        M(seqN(3 * i, 3), i) = vectorAsSkew(pointsRight.col(i)) * R * pointsLeft.col(i);
        M(seqN(3 * i, 3), numMatches) = vectorAsSkew(pointsRight.col(i)) * T;
    }

    // get vector V corresponding to smallest eigenvalue of M'*M (least squares estimate)
    EigenSolver<MatrixXf> es(M.transpose() * M);
    auto &eigenvalues = es.eigenvalues();
    int idxSmallestEigenvalue = 0;
    float smallestEigenvalue = eigenvalues[0].real();
    for (int i = 1; i < numMatches + 1; i++) {
        if (eigenvalues[i].real() < smallestEigenvalue) {
            smallestEigenvalue = eigenvalues[i].real();
            idxSmallestEigenvalue = i;
        }
    }

    VectorXf V = es.eigenvectors().col(idxSmallestEigenvalue).real();

    // reconstruct depth
    float scale = V(last);
    VectorXf depthVec = V(seq(0, last - 1)) / scale; // make scale similar to scale of translation
    MatrixXf tmpPointsLeftReconstructed = MatrixXf::Zero(3, numMatches);
    MatrixXf tmpPointsRightReconstructed = MatrixXf::Zero(3, numMatches);
    tmpPointsLeftReconstructed = pointsLeft.cwiseProduct(depthVec.transpose().replicate(3, 1));
    tmpPointsRightReconstructed = (R * tmpPointsLeftReconstructed) + T.replicate(1, numMatches);

    // check depth of all reconstructed points
    bool success =
            (tmpPointsLeftReconstructed.row(2).array() >= 0).all() && (tmpPointsRightReconstructed.row(2).array() >= 0).all();
    if (success) {
        pointsLeftReconstructed = tmpPointsLeftReconstructed;
        pointsRightReconstructed = tmpPointsRightReconstructed;
    }
    return success;
}

void EightPointAlgorithm::updateData() {
    // transform image coordinates with inverse camera matrix (inverse intrinsics)
    pointsLeft = cameraLeft.inverse() * matchesLeft;
    pointsRight = cameraRight.inverse() * matchesRight;

}

const Matrix3Xf &EightPointAlgorithm::getMatchesLeft() const {
    return matchesLeft;
}

const Matrix3Xf &EightPointAlgorithm::getMatchesRight() const {
    return matchesRight;
}

const Matrix3f &EightPointAlgorithm::getCameraLeft() const {
    return cameraLeft;
}

const Matrix3f &EightPointAlgorithm::getCameraRight() const {
    return cameraRight;
}

void EightPointAlgorithm::setMatches(const Matrix3Xf &leftMatches, const Matrix3Xf &rightMatches) {
    if (leftMatches.cols() != rightMatches.cols()) {
        throw std::runtime_error("Inputs matrices have to contain same amount of points");
    }

    std::vector<int> uniqueIdx = uniqueColumnsInMatrix(leftMatches);
    matchesLeft = leftMatches(all, uniqueIdx);
    matchesRight = rightMatches(all, uniqueIdx);

    numMatches = (int) matchesLeft.cols();
    if (numMatches < 8) {
        throw std::runtime_error("Less than 8 input point pairs detected for processing the Eight Point Algorithm!");
    }
}

void EightPointAlgorithm::setCameraRight(const Matrix3f &camera) {
    cameraRight = camera;
}

void EightPointAlgorithm::setCameraLeft(const Matrix3f &camera) {
    cameraLeft = camera;
}

const Matrix4f &EightPointAlgorithm::getPose() const {
    return pose;
}

const Matrix3f &EightPointAlgorithm::getEssentialMatrix() const {
    return essentialMatrix;
}

Matrix3f EightPointAlgorithm::getFundamentalMatrix() const {
    return cameraLeft.transpose().inverse() * essentialMatrix * cameraLeft.inverse();
}

const Matrix3Xf &EightPointAlgorithm::getPointsLeftReconstructed() const {
    return pointsLeftReconstructed;
}

const Matrix3Xf &EightPointAlgorithm::getPointsRightReconstructed() const {
    return pointsRightReconstructed;
}

Matrix3f vectorAsSkew(const Vector3f &vec) {
    Matrix3f skewMatrix = Matrix3f::Zero();
    // upper triangular matrix
    skewMatrix(0, 1) = -vec.z();
    skewMatrix(0, 2) = vec.y();
    skewMatrix(1, 2) = -vec.x();
    // lower triangular matrix
    skewMatrix(1, 0) = vec.z();
    skewMatrix(2, 0) = -vec.y();
    skewMatrix(2, 1) = vec.x();
    return skewMatrix;
}

VectorXf kron(const VectorXf &vec1, const VectorXf &vec2) {
    int n = (int) vec1.size();
    int m = (int) vec2.size();
    VectorXf out = VectorXf::Zero(n * m);

    for (int i = 0; i < n; i++) {
        out(seqN(i * m, m)) = vec1(i) * vec2;
    }

    return out;
}

void transformMatchedKeypointsToEigen(const std::vector<cv::KeyPoint> &keypointsLeft,
                                      const std::vector<cv::KeyPoint> &keypointsRight,
                                      const std::vector<cv::DMatch> &matches,
                                      Matrix3Xf &outLeft,
                                      Matrix3Xf &outRight) {
    outLeft = Matrix3Xf::Zero(3, matches.size());
    outRight = Matrix3Xf::Zero(3, matches.size());

    int i = 0;
    for (cv::DMatch match : matches) {
        outLeft.col(i) = Vector3f(keypointsLeft[match.queryIdx].pt.x, keypointsLeft[match.queryIdx].pt.y, 1);
        outRight.col(i) = Vector3f(keypointsRight[match.trainIdx].pt.x, keypointsRight[match.trainIdx].pt.y, 1);
        i++;
    }
}

std::vector<int> uniqueColumnsInMatrix(const Matrix3Xf &pointMat, float tol) {
    if (pointMat.cols() == 0 )
        return std::vector<int> {};

    std::vector<int> uniqueIdx = {0};
    for (int i=1; i < pointMat.cols(); i++){
        bool uniqueElement = true;
        for (int j : uniqueIdx){
            float diff = (pointMat.col(i) - pointMat.col(j)).norm();
            if (diff < tol) {
                uniqueElement = false;
                break;
            }
        }
        if (uniqueElement)
            uniqueIdx.emplace_back(i);
    }
    return uniqueIdx;
}
