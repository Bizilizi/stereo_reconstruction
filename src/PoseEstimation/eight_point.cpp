
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <random>
#include <algorithm>

#include "../utils.h"

#include "eight_point.h"

#define N_KEYPOINTS_8PT 12


EightPointAlgorithm::EightPointAlgorithm(const Matrix3Xf &matchesL, const Matrix3Xf &matchesR,
                                         const Matrix3f &cameraLeft, const Matrix3f &cameraRight)
        : cameraLeft{cameraLeft}, cameraRight{cameraRight} {
    // get all matched keypoint pairs and prepare them for using Eigen in the eight point algorithm

    setMatches(matchesL, matchesR);

    pointsLeftReconstructed = MatrixXf::Zero(3, numMatches);
    pointsRightReconstructed = MatrixXf::Zero(3, numMatches);

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
    // estimate depth
    VectorXf depthVec = estimateDepth(R, T);
    // reconstruct points
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

VectorXf EightPointAlgorithm::estimateDepth(const Matrix3f &R, const Vector3f &T) const{
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
    return depthVec;
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

    updateData();
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
    Matrix3f F = cameraRight.transpose().inverse() * essentialMatrix * cameraLeft.inverse();
    return F / F.norm();
}

const Matrix3Xf &EightPointAlgorithm::getPointsLeftReconstructed() const {
    return pointsLeftReconstructed;
}

const Matrix3Xf &EightPointAlgorithm::getPointsRightReconstructed() const {
    return pointsRightReconstructed;
}


std::vector<int> getRandomIndices(int maxIdx, int length, std::vector<int> init, std::vector<int> exclude) {

    std::vector<int> initExcluded;
    if (!init.empty()) {
        std::sort(init.begin(), init.end());
        std::sort(exclude.begin(), exclude.end());
        // remove excluded values from init
        std::set_difference(init.begin(), init.end(), exclude.begin(), exclude.end(),
                            std::inserter(initExcluded, initExcluded.begin()));
        // add all unique values in init to excluded
        exclude.insert(exclude.end(), initExcluded.begin(), initExcluded.end());
    }

    std::vector<int> randomIndices(maxIdx), diff;

    // fill with random indices
    std::iota(randomIndices.begin(), randomIndices.end(), 0);

    // filter out excluded indices
    std::sort(exclude.begin(), exclude.end());
    std::set_difference(randomIndices.begin(), randomIndices.end(), exclude.begin(), exclude.end(),
                        std::inserter(diff, diff.begin()));

    // shuffle result
    std::shuffle(diff.begin(), diff.end(), std::mt19937{std::random_device{}()});

    if (!init.empty()) {
        // insert initial values to front of diff
        diff.insert(diff.begin(), initExcluded.begin(), initExcluded.end());
    }
    diff.resize(length);
    return diff;
}

EightPointAlgorithm
RANSAC(const MatrixXf &kpLeftMat, const MatrixXf &kpRightMat, const Matrix3f &cameraLeft, const Matrix3f &cameraRight) {
    // hyperparameters
    int maxIter = 100;
    int numPoints = N_KEYPOINTS_8PT;
    int numPointsShuffle = 1;
    float errorThreshold = 4.;

    int numMatches = (int) kpLeftMat.cols();

    MatrixXf sampledKpLeft, sampledKpRight;
    std::vector<int> randomIndices, bestIndices;

    // get initial set
    float avgError = 9999;
    while (avgError > errorThreshold * 3) {
        // estimate extrinsics
        randomIndices = getRandomIndices(numMatches, numPoints);

        sampledKpLeft = kpLeftMat(all, randomIndices);
        sampledKpRight = kpRightMat(all, randomIndices);

        EightPointAlgorithm ep(sampledKpLeft, sampledKpRight, cameraLeft, cameraRight);
        try {
            ep.run();
        } catch (std::runtime_error &e) {
            // invalid depth computed, try next set
            continue;
        }
        Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
        MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
        leftToRightProjection = (cameraRight * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

        // compute pixel error per match
        avgError = calculateEuclideanPixelError(leftToRightProjection, ep.getMatchesRight()).sum() / (float) numPoints;
    }

    std::vector<int> currentExclude, alwaysExclude, latestAddedPoints;
    float bestError = avgError;

    // do optimization
    for (int i = 0; (i < maxIter) && (numMatches - alwaysExclude.size() > numPoints); i++) {
        std::sort(randomIndices.begin(), randomIndices.end());   // for easier debugging

        sampledKpLeft = kpLeftMat(all, randomIndices);
        sampledKpRight = kpRightMat(all, randomIndices);

        // estimate extrinsics
        EightPointAlgorithm ep(sampledKpLeft, sampledKpRight, cameraLeft, cameraRight);
        try {
            ep.run();
        } catch (std::runtime_error &e) {
            // always exclude points that made reconstruction fail (latest added points)
            alwaysExclude.insert(alwaysExclude.end(), latestAddedPoints.begin(), latestAddedPoints.end());
            randomIndices = getRandomIndices(numMatches, numPoints, randomIndices, alwaysExclude);

            latestAddedPoints.clear();
            latestAddedPoints.insert(latestAddedPoints.begin(), randomIndices.end() - numPointsShuffle,
                                     randomIndices.end());
            continue;
        }
        bestIndices = randomIndices;

        // compute projection of left image keypoints to the right one
        Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
        MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
        leftToRightProjection = (cameraRight * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

        // compute pixel error per match
        VectorXf errors = calculateEuclideanPixelError(leftToRightProjection, ep.getMatchesRight());

        float currentError = errors.sum() / (float) numPoints;

        if ((errors.array() < errorThreshold).all()) {
            // break random sampling
            return ep;
        } else {
            if (currentError > bestError) {
                // exclude latest points if error increased and sample new one
                alwaysExclude.insert(alwaysExclude.end(), latestAddedPoints.begin(), latestAddedPoints.end());
                randomIndices = getRandomIndices(numMatches, numPoints, randomIndices, alwaysExclude);

            } else {
                // save results
                bestError = currentError;
                bestIndices = randomIndices;

                // remove worst numPointsShuffle points and replace with random other points
                std::vector<int> sortedIdx(randomIndices.size());
                std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
                std::stable_sort(sortedIdx.begin(), sortedIdx.end(),
                                 [&errors](int i1, int i2) { return errors(i1) > errors(i2); });

                currentExclude.clear();
                for (int k = 0; k < numPointsShuffle; k++) {
                    currentExclude.emplace_back(randomIndices[sortedIdx[k]]);
                }
                currentExclude.insert(currentExclude.end(), alwaysExclude.begin(), alwaysExclude.end());

                randomIndices = getRandomIndices(numMatches, numPoints, randomIndices, currentExclude);
            }

            latestAddedPoints.clear();
            latestAddedPoints.insert(latestAddedPoints.begin(), randomIndices.end() - numPointsShuffle,
                                     randomIndices.end());
        }
    };
    return EightPointAlgorithm(kpLeftMat(all, bestIndices), kpRightMat(all, bestIndices), cameraLeft, cameraRight);
}

VectorXf calculateEuclideanPixelError(const MatrixXf &leftToRightProjection, const MatrixXf &matchesRight) {
    VectorXf errors = VectorXf::Zero(leftToRightProjection.cols());
    for (int col = 0; col < leftToRightProjection.cols(); col++) {
        errors(col) = sqrt(powf(leftToRightProjection(0, col) - matchesRight(0, col), 2) +
                           powf(leftToRightProjection(1, col) - matchesRight(1, col), 2));
    }
    return errors;
}



