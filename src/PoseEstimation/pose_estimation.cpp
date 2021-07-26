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
    Matrix3Xf leftKeypoints8pt = ep.getMatchesLeft();

    Matrix4f poseRANSAC = ep.getPose();
    Matrix3f R = poseRANSAC(seqN(0,3), seqN(0,3));
    Vector3f T = poseRANSAC(seqN(0,3), 3);

#if 0
    if (visualize){

        // 8pt
        Matrix3Xf pointsRightReconstructed8pt = Matrix3Xf::Zero(3, pointsLeftReconstructed8pt.cols());
        Matrix3Xf leftToRightProjection = Matrix3Xf::Zero(3, pointsLeftReconstructed8pt.cols());

        pointsRightReconstructed8pt = (R * pointsLeftReconstructed8pt) + T.replicate(1, pointsLeftReconstructed8pt.cols());
        leftToRightProjection = intrinsicsRight * pointsRightReconstructed8pt.cwiseQuotient(pointsRightReconstructed8pt.row(2).replicate(3,1));;

        // left image: bundle adjustment and eight point algorithm
        cv::Mat copyLeft = imageLeft;
        cv::Mat copyRight = imageRight;
        for (int i = 0; i < pointsRightReconstructed8pt.cols(); i++) {
            cv::circle(copyLeft, cv::Point(ep.getMatchesLeft()(0, i), ep.getMatchesLeft()(1, i)), 5,
                       cv::Scalar(0, 255, 0), 5);
        }

        // right image: bundle adjustment and eight point algorithm
        for (int i = 0; i < pointsRightReconstructed8pt.cols(); i++) {
            cv::circle(copyRight, cv::Point(ep.getMatchesRight()(0, i), ep.getMatchesRight()(1, i)), 5.0,
                       cv::Scalar(0, 255, 0), 5);
            cv::circle(copyRight, cv::Point(leftToRightProjection(0, i), leftToRightProjection(1, i)), 8.0,
                       cv::Scalar(0, 0, 255), 3);
        }

        cv::Mat combinedImage;
        cv::hconcat(copyLeft, copyRight, combinedImage);
        cv::imwrite("../../results/eight_point.png", combinedImage);

        // 3D pose after bundle adjustment
        showExtrinsicsReconstruction("../../results/extrinsics_eight_point.off", poseRANSAC, pointsLeftReconstructed8pt, pointsLeftReconstructed8pt);
    }
#endif

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
    // std::cout << "Errors:" << std::endl <<  reError8pt.array() << std::endl;
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

        // optimizer
        Matrix3Xf optimized3DPoints = optimizer.getOptimized3DPoints();
        Matrix3Xf projected3DPointsLeft = intrinsicsLeft * optimized3DPoints.cwiseQuotient(optimized3DPoints.row(2).replicate(3,1));
        Matrix3Xf projected3DPointsRight = (pose(seqN(0,3), seqN(0,3)) * optimized3DPoints +
                pose(seqN(0,3), 3).replicate(1, optimized3DPoints.cols()));
        projected3DPointsRight =  intrinsicsRight * projected3DPointsRight.cwiseQuotient(projected3DPointsRight.row(2).replicate(3,1));

        // 8pt
        Matrix3Xf pointsRightReconstructed8pt = Matrix3Xf::Zero(3, pointsLeftReconstructedFiltered.cols());
        Matrix3Xf leftToRightProjection = Matrix3Xf::Zero(3, pointsLeftReconstructedFiltered.cols());

        pointsRightReconstructed8pt = (R * pointsLeftReconstructedFiltered) + T.replicate(1, pointsLeftReconstructedFiltered.cols());
        leftToRightProjection = intrinsicsRight * pointsRightReconstructed8pt.cwiseQuotient(pointsRightReconstructed8pt.row(2).replicate(3,1));;

        // left image: bundle adjustment and eight point algorithm
        for (int i = 0; i < optimized3DPoints.cols(); i++) {
            cv::circle(imageLeft, cv::Point(projected3DPointsLeft(0, i), projected3DPointsLeft(1, i)), 7,
                       cv::Scalar(255, 0, 0), 2);
            cv::circle(imageLeft, cv::Point(kpLeftReduced(0, i), kpLeftReduced(1, i)), 3,
                       cv::Scalar(0, 255, 0), 3);
        }

        // right image: bundle adjustment and eight point algorithm
        for (int i = 0; i < optimized3DPoints.cols(); i++) {
            cv::circle(imageRight, cv::Point(projected3DPointsRight(0, i), projected3DPointsRight(1, i)), 7,
                       cv::Scalar(255, 0, 0), 2);
            cv::circle(imageRight, cv::Point(kpRightReduced(0, i), kpRightReduced(1, i)), 3.0,
                       cv::Scalar(0, 255, 0), 3);
            // cv::circle(imageRight, cv::Point(leftToRightProjection(0, i), leftToRightProjection(1, i)), 7.0,
            //            cv::Scalar(0, 0, 255), 2);
        }

        cv::Mat combinedImage;
        cv::hconcat(imageLeft, imageRight, combinedImage);
        cv::imwrite("../../results/reconstruction_error_bundle_adjustment.png", combinedImage);

        // 3D pose after bundle adjustment
        showExtrinsicsReconstruction("../../results/reconstruction_error_bundle_adjustment.off", pose, optimized3DPoints, optimized3DPoints);
    }

    Matrix3Xf optimized3DPointsLeft = optimizer.getOptimized3DPoints();
    float reErrorBundleAdjustment = averageReconstructionError(kpLeftReduced, kpRightReduced, intrinsicsLeft, intrinsicsRight, pose(seqN(0,3), seqN(0,3)), pose(seqN(0,3), 3), optimized3DPointsLeft);
    float reError8ptAlgorithm = averageReconstructionError(kpLeftReduced, kpRightReduced, intrinsicsLeft, intrinsicsRight, poseRANSAC(seqN(0,3), seqN(0,3)), poseRANSAC(seqN(0,3), 3), pointsLeftReconstructedReduced);

    if (verbose) {

        std::cout << "------------ Pose Estimation ------------" << std::endl;
        std::cout << "Number of SIFT matches: " << numMatches << std::endl;
        std::cout << "Number outliers: " << (filterMask.array() > 0).count() <<  std::endl;
        std::cout << "Average reprojection error after 8pt algorithm: " << reError8ptAlgorithm << std::endl;
        std::cout << "Average reprojection error after bundle adjustment: " << reErrorBundleAdjustment << std::endl;
        std::cout << "Estimated pose: " << std::endl << pose << std::endl;
    }

    poseStruct result = poseStruct{ pose, optimizer.getFundamentalMatrix(), kpLeftReduced, kpRightReduced, reError8ptAlgorithm, reErrorBundleAdjustment};
    return result;
}