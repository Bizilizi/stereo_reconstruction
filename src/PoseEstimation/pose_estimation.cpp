#include "pose_estimation.h"
#include "keypoints.h"

poseStruct runFullPoseEstimation(const cv::Mat &imageLeft, const cv::Mat &imageRight, const Matrix3f &intrinsicsLeft,
                                 const Matrix3f &intrinsicsRight, bool verbose, bool visualize) {

    // -----------------------------------------------
    // SIFT Feature Extraction and Matching
    // -----------------------------------------------

    // find keypoints
    cv::Mat featuresLeft, featuresRight;
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;

    SIFTKeypointDetection(imageLeft, keypointsLeft, featuresLeft);
    SIFTKeypointDetection(imageRight, keypointsRight, featuresRight);

    // find correspondences
    std::vector<cv::DMatch> matches;
    featureMatching(featuresLeft, featuresRight, matches);

    // transformation to Eigen
    Matrix3Xf kpLeftMat, kpRightMat;
    transformMatchedKeypointsToEigen(keypointsLeft, keypointsRight, matches, kpLeftMat, kpRightMat, true);
    int numMatches = kpLeftMat.cols();
    std::cout << "Number of matches: " << numMatches << std::endl;

    // -----------------------------------------------
    // 8 Point Algorithm
    // -----------------------------------------------

    // run eight point algorithm using points sampled by RANSAC
    EightPointAlgorithm ep = RANSAC(kpLeftMat, kpRightMat, intrinsicsLeft, intrinsicsRight);
    ep.run();
    Matrix3Xf pointsLeftReconstructed8pt = ep.getPointsLeftReconstructed();
    Matrix4f poseRANSAC = ep.getPose();
    Matrix3f R = poseRANSAC(seqN(0,3), seqN(0,3));
    Vector3f T = poseRANSAC(seqN(0,3), 3);

    // filter out points with high reprojection error: reconstruct depth of all matches first, then compute error, then filter
    // depth
    ep.setMatches(kpLeftMat, kpRightMat);
    VectorXf depthVec = ep.estimateDepth(R, T);

    // reconstruct points
    MatrixXf pointsLeftReconstructed = MatrixXf::Zero(3, numMatches);
    MatrixXf pointsRightReconstructed = MatrixXf::Zero(3, numMatches);
    pointsLeftReconstructed = (intrinsicsLeft.inverse() * kpLeftMat).cwiseProduct(depthVec.transpose().replicate(3, 1));
    pointsRightReconstructed = (R * pointsLeftReconstructed) + T.replicate(1, numMatches);
    MatrixXf leftToRightProjection = MatrixXf::Zero(3, numMatches);
    leftToRightProjection = (intrinsicsRight * pointsRightReconstructed).cwiseQuotient(pointsRightReconstructed.row(2).replicate(3, 1));

    VectorXf reError8pt = calculateEuclideanPixelError(leftToRightProjection, kpRightMat);
    std::cout << "Errors:" << std::endl <<  reError8pt.array() << std::endl;
    VectorXi filterMask = ((reError8pt.array() > reError8pt.array().mean() * 2) + (depthVec.array() < 0)).cast<int>();
    // std::cout << filterMask << std::endl;

    // filter
    int uniqueElements = (filterMask.array() < 1).count();
    Matrix3Xf kpLeftFiltered = MatrixXf::Ones(3, uniqueElements);
    Matrix3Xf kpRightFiltered = MatrixXf::Ones(3, uniqueElements);
    Matrix3Xf pointsLeftReconstructedFiltered = MatrixXf::Ones(3, uniqueElements);
    int idxFiltered = 0;
    for (int i = 0; i < filterMask.size(); i++){
        if (filterMask(i) < 1){
            kpLeftFiltered(all, idxFiltered) = kpLeftMat(all, i);
            kpRightFiltered(all, idxFiltered) = kpRightMat(all, i);
            pointsLeftReconstructedFiltered(all, idxFiltered) = pointsLeftReconstructed(all, i);
            idxFiltered++;
        }
    }
    // std::cout << kpLeftMat << std::endl;
    std::cout << kpLeftFiltered << std::endl;
    std::cout << pointsLeftReconstructedFiltered << std::endl;

    // -----------------------------------------------
    // Bundle Adjustment
    // -----------------------------------------------
    const int maxPoints = 60; // due to templates in ceras

    Matrix3Xf kpLeftReduced = Matrix3Xf::Zero(3, maxPoints);
    Matrix3Xf kpRightReduced = Matrix3Xf::Zero(3, maxPoints);
    Matrix3Xf pointsLeftReconstructedReduced = Matrix3Xf::Zero(3, maxPoints);

    if (maxPoints > kpLeftFiltered.cols()){
        throw std::runtime_error("Bundle Adjustment takes fix number of points, not enough features found");
    } else {
        kpLeftReduced = kpLeftFiltered(all, seqN(0, maxPoints));
        kpRightReduced = kpRightFiltered(all, seqN(0, maxPoints));
        pointsLeftReconstructedReduced = pointsLeftReconstructedFiltered(all, seqN(0, maxPoints));
    }
    std::cout << "MARKER" << std::endl;
    std::cout << kpLeftFiltered << std::endl;
    std::cout << "reconstructed points" << std::endl;
    std::cout << pointsLeftReconstructedFiltered << std::endl;

    auto optimizer = BundleAdjustmentOptimizer(kpLeftReduced, kpRightReduced, intrinsicsLeft,
                                               intrinsicsRight, R, T, pointsLeftReconstructedReduced);
    Matrix4f pose = optimizer.estimatePose();

    if (visualize){
        // Matches
        cv::Mat img_matches;
        cv::drawMatches(imageLeft,  keypointsLeft, imageRight, keypointsRight, matches, img_matches,
                        cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite("../../results/SIFT_matches.png", img_matches);

        // bundle adjustment left to right projection: left image
        Matrix3Xf optimized3DPoints = optimizer.getOptimized3DPoints();
        Matrix3Xf projected3DPointsLeft = intrinsicsLeft * optimized3DPoints.cwiseQuotient(optimized3DPoints.row(2).replicate(3,1));
        Matrix3Xf pointsRightReconstructed8pt = Matrix3Xf::Zero(3, pointsLeftReconstructed8pt.cols());
        Matrix3Xf leftToRightProjection = Matrix3Xf::Zero(3, pointsLeftReconstructed8pt.cols());
        pointsRightReconstructed8pt = (R * pointsLeftReconstructed8pt) + T.replicate(1, pointsLeftReconstructed8pt.cols());
        leftToRightProjection = intrinsicsRight * pointsRightReconstructed8pt.cwiseQuotient(pointsRightReconstructed8pt.row(2).replicate(3,1));;
        std::cout << leftToRightProjection << std::endl;

        for (int i = 0; i < optimized3DPoints.cols(); i++) {
            cv::circle(imageLeft, cv::Point(projected3DPointsLeft(0, i), projected3DPointsLeft(1, i)), 3,
                       cv::Scalar(255, 0, 0), 5);
            cv::circle(imageLeft, cv::Point(kpLeftReduced(0, i), kpLeftReduced(1, i)), 5.0,
                       cv::Scalar(0, 255, 0), 3);
            if (i < leftToRightProjection.cols()) {
                cv::circle(imageLeft, cv::Point(leftToRightProjection(0, i), leftToRightProjection(1, i)), 5.0,
                           cv::Scalar(0, 0, 255), 3);
            }
        }

        // bundle adjustment left to right projection: right image
        for (int i = 0; i < optimized3DPoints.cols(); i++) {
            cv::circle(imageLeft, cv::Point(projected3DPointsLeft(0, i), projected3DPointsLeft(1, i)), 3,
                       cv::Scalar(255, 0, 0), 5);
            cv::circle(imageLeft, cv::Point(kpLeftReduced(0, i), kpLeftReduced(1, i)), 5.0,
                       cv::Scalar(0, 255, 0), 3);
            if (i < leftToRightProjection.cols()) {
                cv::circle(imageLeft, cv::Point(leftToRightProjection(0, i), leftToRightProjection(1, i)), 5.0,
                           cv::Scalar(0, 0, 255), 3);
            }
        }

        cv::imwrite("../../results/reconstruction_error_bundle_adjustment.png", imageLeft);

        // 3D pose after bundle adjustment
        showExtrinsicsReconstruction("../../results/reconstruction_error_bundle_adjustment.off", pose, optimized3DPoints, optimized3DPoints);
    }

    if (verbose) {
        Matrix3Xf optimized3DPointsLeft = optimizer.getOptimized3DPoints();
        std::cout << optimized3DPointsLeft << std::endl;
        float reErrorBundleAdjustment = averageReconstructionError(kpLeftReduced, kpRightReduced, intrinsicsLeft, intrinsicsRight, pose(seqN(0,3), seqN(0,3)), pose(seqN(0,3), 3), optimized3DPointsLeft);
        float reError8ptAlgorithm = averageReconstructionError(kpLeftReduced, kpRightReduced, intrinsicsLeft, intrinsicsRight, poseRANSAC(seqN(0,3), seqN(0,3)), poseRANSAC(seqN(0,3), 3), pointsLeftReconstructedReduced);

        std::cout << "------------ Pose Estimation ------------" << std::endl;
        std::cout << "Number of SIFT matches: " << numMatches << std::endl;
        std::cout << "Number outliers: " << (filterMask.array() > 0).count() <<  std::endl;
        std::cout << "Average reprojection error after 8pt algorithm: " << reError8ptAlgorithm << std::endl;
        std::cout << "Average reprojection error after bundle adjustment: " << reErrorBundleAdjustment << std::endl;
        std::cout << "Estimated pose: " << pose << std::endl;
    }

    poseStruct result = poseStruct{ pose, optimizer.getFundamentalMatrix(), kpLeftReduced, kpRightReduced };
    return result;
}