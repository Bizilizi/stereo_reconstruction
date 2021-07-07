//
// Created by tim on 19.06.21.
//

#include <iostream>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <random>
#include <algorithm>

#include "Eigen.h"
#include "eight_point.h"
#include "SimpleMesh.h"
#include "BundleAdjustment.h"
#include "data_loader.h"


void featureMatching(const cv::Mat &featuresLeft, const cv::Mat &featuresRight, std::vector<cv::DMatch> &outputMatches,
                     float ratio_thresh = 0.7f) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(featuresLeft, featuresRight, knn_matches, 2);

    // filter matches using the Lowe's ratio test
    for (auto &knn_match : knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            outputMatches.push_back(knn_match[0]);
        }
    }
}


void SIFTKeypointDetection(const cv::Mat &image, std::vector<cv::KeyPoint> &outputKeypoints, cv::Mat &outputDescriptor,
                           const float edgeThreshold = 10, const float contrastThreshold = 0.04) {
    // detect outputKeypoints using SIFT
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(200, 3, contrastThreshold, edgeThreshold);
    detector->detectAndCompute(image, cv::noArray(), outputKeypoints, outputDescriptor);
}


VectorXf calculateEuclideanPixelError(const MatrixXf &leftToRightProjection, const MatrixXf &matchesRight) {
    VectorXf errors = VectorXf::Zero(leftToRightProjection.cols());
    for (int col = 0; col < leftToRightProjection.cols(); col++) {
        errors(col) = sqrt(powf(leftToRightProjection(0, col) - matchesRight(0, col), 2) +
                           powf(leftToRightProjection(1, col) - matchesRight(1, col), 2));
    }
    return errors;
}


void showExtrinsicsReconstruction(const std::string &filename, const Matrix4f &pose,
                                  const Matrix3Xf &pointsLeftReconstructed, const Matrix3Xf &pointsRightReconstructed) {
    SimpleMesh outputMesh;
    for (int i = 0; i < pointsLeftReconstructed.cols(); i++) {
        SimpleMesh point = SimpleMesh::sphere(pointsLeftReconstructed.col(i), 0.5f);
        outputMesh = SimpleMesh::joinMeshes(outputMesh, point);
    }

    SimpleMesh cameraLeftMesh = SimpleMesh::camera(Matrix4f::Identity(), 0.05f);
    SimpleMesh cameraRightMesh = SimpleMesh::camera(pose, 0.05f, Vector4uc(0, 255, 0, 255));

    SimpleMesh cameraMesh = SimpleMesh::joinMeshes(cameraLeftMesh, cameraRightMesh);
    outputMesh = SimpleMesh::joinMeshes(outputMesh, cameraMesh);

    outputMesh.writeMesh(filename);
}

std::vector<int> getRandomIndices(int maxIdx, int length, std::vector<int> init = {}, std::vector<int> exclude = {}) {

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
    int numPoints = 12;
    int numPointsShuffle = 1;
    float errorThreshold = 3.;

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
            // std::cout << e.what() << std::endl;
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
        std::cout << "Iteration: " << i << std::endl;
        std::cout << "Number of excluded points: " << alwaysExclude.size() << std::endl;
        std::sort(randomIndices.begin(), randomIndices.end());   // for easier debugging

        sampledKpLeft = kpLeftMat(all, randomIndices);
        sampledKpRight = kpRightMat(all, randomIndices);

        std::cout << "Random indices: " << std::endl;
        for (auto idx : randomIndices)
            std::cout << idx << "   ";

        // estimate extrinsics
        EightPointAlgorithm ep(sampledKpLeft, sampledKpRight, cameraLeft, cameraRight);
        try {
            ep.run();
        } catch (std::runtime_error &e) {
            // invalid depth computed, try next set
            std::cout << e.what() << std::endl;

            // always exclude points that made reconstruction fail (latest added points)
            alwaysExclude.insert(alwaysExclude.end(), latestAddedPoints.begin(), latestAddedPoints.end());
            randomIndices = getRandomIndices(numMatches, numPoints, randomIndices, alwaysExclude);

            latestAddedPoints.clear();
            latestAddedPoints.insert(latestAddedPoints.begin(), randomIndices.end() - numPointsShuffle,
                                     randomIndices.end());
            continue;
        }

        // compute projection of left image keypoints to the right one
        Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
        MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
        leftToRightProjection = (cameraRight * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

        // compute pixel error per match
        VectorXf errors = calculateEuclideanPixelError(leftToRightProjection, ep.getMatchesRight());

        std::cout << "Errors : " << errors.transpose() << std::endl;

        float currentError = errors.sum() / (float) numPoints;

        if ((errors.array() < errorThreshold).all()) {
            // break random sampling
            return ep;
        } else {
            if (currentError > bestError) {
                std::cout << "No improvement, current error: " << currentError << std::endl;

                // exclude latest points if error increased and sample new one
                alwaysExclude.insert(alwaysExclude.end(), latestAddedPoints.begin(), latestAddedPoints.end());
                randomIndices = getRandomIndices(numMatches, numPoints, randomIndices, alwaysExclude);

            } else {
                std::cout << "Improvement made, current error: " << currentError << std::endl;

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
    // no success
    return EightPointAlgorithm(kpLeftMat(all, bestIndices), kpRightMat(all, bestIndices), cameraLeft, cameraRight);
}


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


int main() {
    DataLoader dataLoader = DataLoader();

    // select scenarios by index (alphabetic position starting with 0)
    Data data = dataLoader.loadTrainingScenario(11);

    // find keypoints
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat featuresLeft, featuresRight;

    SIFTKeypointDetection(data.getImageLeft(), keypointsLeft, featuresLeft);
    SIFTKeypointDetection(data.getImageRight(), keypointsRight, featuresRight);

    // find correspondences
    std::vector<cv::DMatch> matches;
    featureMatching(featuresLeft, featuresRight, matches);

    // visualization of feature extraction
    cv::Mat outputImageLeft, outputImageRight;
    cv::drawKeypoints(data.getImageLeft(), keypointsLeft, outputImageLeft);
    cv::drawKeypoints(data.getImageRight(), keypointsRight, outputImageRight);
    cv::imwrite("../../results/imageLeft.png", outputImageLeft);
    cv::imwrite("../../results/imageRight.png", outputImageRight);

    // visualization of feature matching
    cv::Mat img_matches;
    cv::drawMatches(data.getImageLeft(), keypointsLeft, data.getImageRight(), keypointsRight, matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../../results/matching_flann.png", img_matches);

    // estimate pose using the eight point algorithm
    Matrix3Xf kpLeftMat, kpRightMat;
    transformMatchedKeypointsToEigen(keypointsLeft, keypointsRight, matches, kpLeftMat, kpRightMat);

    // TODO: Embed RANSAC to Eight-point class instead of using dirty solution
    EightPointAlgorithm dirtyFix(kpLeftMat, kpRightMat, data.getCameraMatrixLeft(), data.getCameraMatrixRight());

    EightPointAlgorithm ep = RANSAC(dirtyFix.getMatchesLeft(), dirtyFix.getMatchesRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight());
    ep.run();

    // TEST
    Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
    MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
    leftToRightProjection = (data.getCameraMatrixRight() * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

    std::cout << "compare matches in pixel coordinates:" << std::endl;
    std::cout << ep.getMatchesRight() << std::endl;
    std::cout << leftToRightProjection << std::endl;

    // ---------------------------------------------------------
    // Bundle Adjustment
    // ---------------------------------------------------------

    std::cout << "BUNDLE ADJUSTMENT" << std::endl;
    Matrix4f pose = ep.getPose();
    auto optimizer = BundleAdjustmentOptimizer(ep.getMatchesLeft(), ep.getMatchesRight(), data.getCameraMatrixLeft(), data.getCameraMatrixRight(), pose(seqN(0,3), seqN(0,3)), pose(seqN(0,3), 3), ep.getPointsLeftReconstructed());
    pose = optimizer.estimatePose();
    std::cout << "Final pose estimation: " << std::endl;
    std::cout << pose << std::endl;

    // testVisualizationExtrinsics();
    // testCaseExtrinsics();

    for (int i = 0; i < rightPoints3D.cols(); i++) {
        cv::circle(data.getImageRight(), cv::Point(leftToRightProjection(0, i), leftToRightProjection(1, i)), 5.0,
                   cv::Scalar(255, 0, 0), 4);
        cv::circle(data.getImageRight(), cv::Point(ep.getMatchesRight()(0, i), ep.getMatchesRight()(1, i)), 5.0,
                   cv::Scalar(0, 255, 0), 4);
        //cv::circle(data.getImageRight(), cv::Point(ep.getMatchesLeft()(0,i), ep.getMatchesLeft()(1,i)), 5.0, cv::Scalar(0, 255, 255), 4);
    }

    cv::imwrite("../../results/greatImage.png", data.getImageRight());

    return 0;
}


/**
* TODO: next week/future
 *
 * 1. RANSAC:
 *      - embed RANSAC as boolean parameter in class and set mask for indices (somewhere in set/update data)
 *
 * 2. SDK: Read and load Disparity maps
 *
 * 3. Test if it works for all scenarios: Had runtime exception once (less than 8 input points!!!)
 *
*/
