
#include "../PoseEstimation/keypoints.h"
#include "../PoseEstimation/bundle_adjustment.h"
#include "../DataLoader/data_loader.h"
#include "../utils.h"


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
    Data data = dataLoader.loadTrainingScenario(1);

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
    cv::drawMatches(data.getImageLeft(), keypointsLeft, data.getImageRight(), keypointsRight, matches, img_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../../results/matching_flann.png", img_matches);

    // estimate pose using the eight point algorithm
    Matrix3Xf kpLeftMat, kpRightMat;
    transformMatchedKeypointsToEigen(keypointsLeft, keypointsRight, matches, kpLeftMat, kpRightMat);

    // TODO: Embed RANSAC to Eight-point class instead of using dirty solution
    EightPointAlgorithm dirtyFix(kpLeftMat, kpRightMat, data.getCameraMatrixLeft(), data.getCameraMatrixRight());

    EightPointAlgorithm ep = RANSAC(dirtyFix.getMatchesLeft(), dirtyFix.getMatchesRight(), data.getCameraMatrixLeft(),
                                    data.getCameraMatrixRight());
    ep.run();

    // TEST
    Matrix3Xf rightPoints3D = ep.getPointsRightReconstructed();
    MatrixXf leftToRightProjection = MatrixXf::Zero(3, rightPoints3D.cols());
    leftToRightProjection = (data.getCameraMatrixRight() * rightPoints3D).cwiseQuotient(
            rightPoints3D.row(2).replicate(3, 1));

    std::cout << "compare matches in pixel coordinates:" << std::endl;
    std::cout << ep.getMatchesRight() << std::endl;
    std::cout << leftToRightProjection << std::endl;

    // ---------------------------------------------------------
    // Bundle Adjustment
    // ---------------------------------------------------------

    std::cout << "BUNDLE ADJUSTMENT" << std::endl;
    Matrix4f pose = ep.getPose();
    auto optimizer = BundleAdjustmentOptimizer(ep.getMatchesLeft(), ep.getMatchesRight(), data.getCameraMatrixLeft(),
                                               data.getCameraMatrixRight(), pose(seqN(0, 3), seqN(0, 3)),
                                               pose(seqN(0, 3), 3), ep.getPointsLeftReconstructed());
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

    // ---------------------------------------------------------
    // Test fundamental matrix with opencv results
    // ---------------------------------------------------------

    Matrix3f F_adirondack;
    F_adirondack << 1.353630396977012e-08, 2.595224518574046e-05, -0.003664439029043263,
            -2.375965046088001e-05, 6.518226848965876e-06, 0.6848607609793337,
            0.002978229678431134, -0.6892694179932449, 1;
    std::cout << "Norm 1: " << F_adirondack.norm() << std::endl;
    F_adirondack = F_adirondack / F_adirondack.norm();

    Matrix3f F_motorcycle;
    F_motorcycle << 1.091980817956205e-09, -1.077326037996396e-06, -5.992489813392534e-05,
            -1.076078736340656e-06, 5.56622420714599e-06, 0.6117147946026104,
            0.0005622610321208721, -0.615586969390141, 1;
    std::cout << "Norm 2: " << F_motorcycle.norm();
    F_motorcycle = F_motorcycle / F_motorcycle.norm();

    Matrix3f F_artl;
    F_artl << 5.006730349381669e-09, -6.911283149129414e-05, 0.02474192270203782,
            6.672591643596967e-05, -3.725089989820235e-06, 0.4516838929035329,
            -0.02386824152172776, -0.4534031147114552, 0.9999999999999999;
    F_artl = F_artl/ F_artl.norm();

    Matrix3f F = F_motorcycle;


    std::cout << std::endl <<  "COMPARISON after Bundle adjustment: " << std::endl;
    std::cout << "Error " << (optimizer.getFundamentalMatrix() - F).norm() << std::endl;
    std::cout << "Norm:" << (optimizer.getFundamentalMatrix().norm()) << std::endl;
    std::cout << optimizer.getFundamentalMatrix() << std::endl << std::endl;
    std::cout << F << std::endl;

    std::cout << std::endl <<  "COMPARISON 8 pt " << std::endl;
    std::cout << "Error " << (ep.getFundamentalMatrix() - F).norm() << std::endl;
    std::cout << "Norm:" << (ep.getFundamentalMatrix().norm()) << std::endl;
    std::cout << ep.getFundamentalMatrix() << std::endl << std::endl;
    std::cout << F << std::endl;

    return 0;
}


/**
* TODO: next week/future
 *
 * RANSAC:
 *      - embed RANSAC as boolean parameter in class and set mask for indices (somewhere in set/update data)
 *
 * Test if it works for all scenarios: Had runtime exception once (less than 8 input points!!!)
 *
 * Test fundamental matrix
 * Reconstruction Pipeline: Loading ground truth + triangulation
 *
*/