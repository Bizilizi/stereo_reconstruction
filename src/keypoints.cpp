//
// Created by tim on 19.06.21.
//

#include <iostream>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "Eigen.h"
#include "directory.h"
#include "eight_point.h"
#include "SimpleMesh.h"


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
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(50, 3, contrastThreshold, edgeThreshold);
    detector->detectAndCompute(image, cv::noArray(), outputKeypoints, outputDescriptor);
}


float calculateEuclideanAveragePixelError(const MatrixXf& leftToRightProjection, const MatrixXf& matchesRight) {
    VectorXf errors = VectorXf::Zero(leftToRightProjection.cols());
    for (int col=0; col < leftToRightProjection.cols(); col++) {
        errors(col) = sqrt(powf(leftToRightProjection(0, col) - matchesRight(0, col), 2) +
                           powf(leftToRightProjection(1, col) - matchesRight(1, col), 2));
    }
    std::cout << errors.transpose() << std::endl;
    return errors.sum() / (float) leftToRightProjection.cols();
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


EightPointAlgorithm RANSAC(const MatrixXf& kpLeftMat, const MatrixXf& kpRightMat, const Matrix3f& cameraLeft, const Matrix3f& cameraRight) {
    // hyperparameters
    int maxSamples = 100;
    int numPoints = 12;
    float errorThreshold = 3.;

    MatrixXf sampledKpLeft, sampledKpRight;
    sampledKpLeft = MatrixXf::Zero(3,numPoints);
    sampledKpRight = MatrixXf::Zero(3, numPoints);

    for (int i=0; i < maxSamples; i++){
        // create set of numPoints random indices
        std::vector<int> randomIndices;
        while (randomIndices.size() < numPoints) {
            int index = rand() % kpLeftMat.cols();
            if(std::find(randomIndices.begin(), randomIndices.end(), index) == randomIndices.end()) {
                randomIndices.emplace_back(index);
            }
        }
        // create subset of matches based on random indices
        for (int j=0; j < randomIndices.size(); j++) {
            sampledKpLeft.block(0, j, 3, 1) = kpLeftMat.block(0, randomIndices[j], 3, 1);
            sampledKpRight.block(0, j, 3, 1) = kpRightMat.block(0, randomIndices[j], 3, 1);
        }

        // estimate extrinsics
        try {
            EightPointAlgorithm ep(sampledKpLeft, sampledKpRight, cameraLeft, cameraRight);

            // compute projection of left image keypoints to the right one
            Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
            MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
            leftToRightProjection = (cameraRight * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

            std::cout << "Found a solution!" << std::endl;

            // compute pixel error
            float error = calculateEuclideanAveragePixelError(leftToRightProjection, ep.getMatchesRight());
            std::cout << error << std::endl;
            if (error < errorThreshold) {
                // break random sampling
                return ep;
            }
        } catch (std::runtime_error& e) {
            // invalid depth computed, try next set
            std::cout << e.what() << std::endl;
        }
    }
    // no success
    throw std::runtime_error("None of the estimated extrinsics was accurate enough!");
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


int main(int argc, char **argv) {
    std::string image_path;
    if (argc == 1) {
        image_path = getCurrentDirectory() + "/../../data/MiddEval3/trainingH/Pipes";
    } else {
        image_path = std::string(argv[1]);
    }

    // load stereo images
    cv::Mat imageLeft = cv::imread(image_path + "/im0.png", cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread(image_path + "/im1.png", cv::IMREAD_COLOR);
    if (!imageLeft.data || !imageRight.data) {
        std::cout << "No image data. Check file path!" << std::endl;
        return -1;
    }

    // find keypoints
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat featuresLeft, featuresRight;

    SIFTKeypointDetection(imageLeft, keypointsLeft, featuresLeft);
    SIFTKeypointDetection(imageRight, keypointsRight, featuresRight);

    // find correspondences
    std::vector<cv::DMatch> matches;
    featureMatching(featuresLeft, featuresRight, matches);

    // visualization of feature extraction
    cv::Mat outputImageLeft, outputImageRight;
    cv::drawKeypoints(imageLeft, keypointsLeft, outputImageLeft);
    cv::drawKeypoints(imageRight, keypointsRight, outputImageRight);
    cv::imwrite("../../results/imageLeft.png", outputImageLeft);
    cv::imwrite("../../results/imageRight.png", outputImageRight);

    // visualization of feature matching
    cv::Mat img_matches;
    cv::drawMatches(imageLeft, keypointsLeft, imageRight, keypointsRight, matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../../results/matching_flann.png", img_matches);


    // estimate pose using the eight point algorithm
    // TODO read calib.txt to get camera matrices -> SDK?
    Matrix3f cameraLeft, cameraRight;

    // ArtL
    //cameraLeft << 1870, 0, 297, 0, 1870, 277, 0, 0, 1;
    //cameraRight << 1870, 0, 397, 0, 1870, 277, 0, 0, 1;

    // Motorcycle
    //cameraLeft << 1998.842, 0, 588.364, 0, 1998.842, 505.864, 0, 0, 1;
    //cameraRight << 1998.842, 0, 653.919, 0, 1998.842, 505.864, 0, 0, 1;

    // PianoL
    //cameraLeft << 1426.379, 0, 712.043, 0, 1426.379, 476.526, 0, 0, 1;
    //cameraRight << 1426.379, 0, 774.722, 0, 1426.379, 476.526, 0, 0, 1;

    // Pipes
    cameraLeft << 1979.773, 0, 785.885, 0, 1979.773, 486.442, 0, 0, 1;
    cameraRight << 1979.773, 0, 824.548, 0, 1979.773, 486.442, 0, 0, 1;

    Matrix3Xf kpLeftMat, kpRightMat;
    transformMatchedKeypointsToEigen(keypointsLeft, keypointsRight, matches, kpLeftMat, kpRightMat);

    EightPointAlgorithm dirtyFix(kpLeftMat, kpRightMat, cameraLeft, cameraRight);

    EightPointAlgorithm ep = RANSAC(dirtyFix.getMatchesLeft(), dirtyFix.getMatchesRight(), cameraLeft, cameraRight);
    /*
    std::cout << "Keypoints left (in Pixel coordinates): " << std::endl;
    std::cout << ep.getMatchesLeft() << std::endl;
    std::cout << "Points left (after applying inverse intrinsics): " << std::endl;
    std::cout << cameraLeft.inverse() * ep.getMatchesLeft() << std::endl;

    Matrix3f essentialMatrix = ep.getEssentialMatrix();
    std::cout << "Essential Matrix: " << std::endl << essentialMatrix << std::endl;

    Matrix4f pose = ep.getPose();
    std::cout << "Pose: " << std::endl;
    std::cout << pose << std::endl;

    std::cout << "Reconstructed 3D points left" << std::endl;
    std::cout << ep.getPointsLeftReconstructed() << std::endl;
    showExtrinsicsReconstruction("8pt_reconstruction.off", pose, ep.getPointsLeftReconstructed(), ep.getPointsRightReconstructed());
    */

    // TEST
    Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
    MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
    leftToRightProjection = (cameraRight * rightPoints3D).cwiseQuotient(rightPoints3D.row(2).replicate(3, 1));

    std::cout << "compare matches in pixel coordinates:" << std::endl;
    std::cout << ep.getMatchesRight() << std::endl;
    std::cout << leftToRightProjection << std::endl;
    // testVisualizationExtrinsics();
    // testCaseExtrinsics();

    for (int i = 0; i < rightPoints3D.cols(); i++) {
        cv::circle(imageRight, cv::Point(leftToRightProjection(0,i), leftToRightProjection(1,i)), 5.0, cv::Scalar(255, 0, 0), 4);
        cv::circle(imageRight, cv::Point(ep.getMatchesRight()(0,i), ep.getMatchesRight()(1,i)), 5.0, cv::Scalar(0, 255, 0), 4);
        //cv::circle(imageRight, cv::Point(ep.getMatchesLeft()(0,i), ep.getMatchesLeft()(1,i)), 5.0, cv::Scalar(0, 255, 255), 4);
    }

    cv::imwrite("../../results/greatImage.png", imageRight);

    return 0;
}




/**
* TODO: next week/future
 *
 * 1. RANSAC:
 *      - just sample 2 or 3 new indices -> kick out the worst performing and set black list
 *      - embed RANSAC as boolean parameter in class and set mask for indices (somewhere in set/update data)
 *
 * 2. Optimization: Redefinement of results of 8 pt algorithm (Eigen)
 *      - BundleAdjustment (Ceres)
 *
 * 3. SDK: Reading Image Data
 *     int image_idx = 3;  // one index per scenario in trainingFiles
 *     class Dataloader(/relative/path/to/training/data )
 *     load(idx) > data struct
 *
 *     data struct{
 *     imageLeft,
 *     imageRight,
 *     intrinsics,
 *     xyz,
 *     }
 *
*/
