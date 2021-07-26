//
// Created by tim on 14.07.21.
//

#include "utils.h"


float computeAverageDisparity(cv::Mat& disparityMap) {
    float sum = 0;
    int validCounter = 0;
    for (int i=0; i < disparityMap.rows; i++) {
        for (int j=0; j < disparityMap.cols; j++) {
            if (!isinf(disparityMap.at<float>(i, j))) {
                sum += disparityMap.at<float>(i, j);
                validCounter++;
            }
        }
    }
    return sum / validCounter;
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
                                      Matrix3Xf &outRight,
                                      bool filterDuplicates) {
    outLeft = Matrix3Xf::Zero(3, matches.size());
    outRight = Matrix3Xf::Zero(3, matches.size());

    int i = 0;
    for (cv::DMatch match : matches) {
        outLeft.col(i) = Vector3f(keypointsLeft[match.queryIdx].pt.x, keypointsLeft[match.queryIdx].pt.y, 1);
        outRight.col(i) = Vector3f(keypointsRight[match.trainIdx].pt.x, keypointsRight[match.trainIdx].pt.y, 1);
        i++;
    }

    if (filterDuplicates) {
        std::vector<int> uniqueIdx = uniqueColumnsInMatrix(outLeft);
        Matrix3Xf tmp = Matrix3Xf::Zero(3, uniqueIdx.size());
        tmp = outLeft(all, uniqueIdx);
        outLeft = tmp;
        tmp = outRight(all, uniqueIdx);
        outRight = tmp;
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


float averageReconstructionError(const Matrix3Xf& matchesLeft, const Matrix3Xf& matchesRight,
                                 const Matrix3f& intrinsicsLeft, const Matrix3f& intrinsicsRight,
                                 const Matrix3f& rotation, const Vector3f& translation,
                                 const Matrix3Xf& reconstructedPointsLeft){
    int nPoints = (int) reconstructedPointsLeft.cols();

    // projection error left picture
    Matrix3Xf projectedPointsLeft;
    projectedPointsLeft = (intrinsicsLeft * reconstructedPointsLeft).cwiseQuotient(reconstructedPointsLeft.row(2).replicate(3,1));

    VectorXf errorsLeft;
    errorsLeft = (projectedPointsLeft - matchesLeft).colwise().squaredNorm();
    std::cout << "Error left: " << errorsLeft.sum() / (float) nPoints << std::endl;

    // projection error right picture
    Matrix3Xf translatedPoints, projectedPointsRight;
    translatedPoints = rotation * reconstructedPointsLeft + translation.replicate(1, nPoints);
    projectedPointsRight = (intrinsicsRight * translatedPoints).cwiseQuotient(translatedPoints.row(2).replicate(3,1));
    VectorXf errorsRight = (projectedPointsRight - matchesRight).colwise().squaredNorm();
    std::cout << "Errors right: " << errorsRight.sum() / (float) nPoints << std::endl;

    return (errorsLeft.sum() + errorsRight.sum()) / (2.f * nPoints);
}


void evaldisp(cv::Mat disp, cv::Mat gtdisp, cv::Mat mask, float badthresh, float maxdisp, int rounddisp)
{
    cv::Size gtShape = gtdisp.size();
    cv::Size sh = disp.size();
    cv::Size maskShape = mask.size();
    assert (gtShape == sh);
    assert (gtShape == maskShape);
    int n = 0;
    int bad = 0;
    int invalid = 0;
    float serr = 0;
    for (int y = 0; y < gtShape.height; y++) {
        for (int x = 0; x < gtShape.width; x++) {
            float gt = gtdisp.at<float>(y, x);
            if (gt == INFINITY)                      // unknown
                continue;
            float d = disp.at<float>(y, x);
            bool valid = (d != 0);
            if (valid)
                d = std::max(0.0f, std::min(maxdisp, d));
            if (valid && rounddisp)
                d = round(d);
            float err = std::abs(d - gt);
            if (mask.at<uint8_t>(y, x) != 255) {
                // do not evaluate
            } else {
                n++;
                if (valid) {
                    serr += err;
                    if (err > badthresh)
                        bad++;
                } else {
                    invalid++;
                }
            }
        }
    }

    float badpercent =  100.0 * bad / n;
    float invalidpercent =  100.0 * invalid / n;
    float totalbadpercent =  100.0 * ( bad + invalid ) / n;
    float avgErr = serr / (n - invalid);
    std::cout << "number of evaluated: " << n << "\n";
    printf("valid: %4.1f \nbad percent: %6.2f  \ninvalid percent: %6.2f  \ntotal bad percent: %6.2f \narbErr: %6.2f\n",   100.0*n/(gtShape.width * gtShape.height),
           badpercent, invalidpercent, totalbadpercent, avgErr);
}