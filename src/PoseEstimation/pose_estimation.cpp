#include "pose_estimation.h"
#include "keypoints.h"

poseStruct runFullPoseEstimation(const cv::Mat &imageLeft, const cv::Mat &imageRight, const Matrix3f &intrinsicsLeft,
                                 const Matrix3f &intrinsicsRight, bool verbose) {

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

    // filter out points with high reprojection error
    // 1) reconstruct depth of all matches first, 2) then compute error, 3) then filter
    // depth
    ep.setMatches(kpLeftMat, kpRightMat);
    VectorXf depthVec = ep.estimateDepth(R, T);

    // 1) reconstruct points
    MatrixXf pointsLeftReconstructed = MatrixXf::Zero(3, numMatches);
    MatrixXf pointsRightReconstructed = MatrixXf::Zero(3, numMatches);
    pointsLeftReconstructed = (intrinsicsLeft.inverse() * kpLeftMat).cwiseProduct(depthVec.transpose().replicate(3, 1));
    pointsRightReconstructed = (R * pointsLeftReconstructed) + T.replicate(1, numMatches);
    MatrixXf leftToRightProjection = MatrixXf::Zero(3, numMatches);
    leftToRightProjection = (intrinsicsRight * pointsRightReconstructed).cwiseQuotient(pointsRightReconstructed.row(2).replicate(3, 1));

    // 2 compute error
    VectorXf reError8pt = calculateEuclideanPixelError(leftToRightProjection, kpRightMat);
    VectorXi filterMask = ((reError8pt.array() > reError8pt.array().mean() * 2) + (depthVec.array() < 0)).cast<int>();

    // 3) filter invalid points (negative depth) or points with error > 2 * average_error
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

    // -----------------------------------------------
    // Compute Error Metrics
    // -----------------------------------------------

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