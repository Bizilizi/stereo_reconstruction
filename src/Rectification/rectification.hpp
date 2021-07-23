#ifndef STEREO_RECONSTRUCTION_RECTIFICATION_RECTIFICATION_HPP_
#define STEREO_RECONSTRUCTION_RECTIFICATION_RECTIFICATION_HPP_

#include "../Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using std::vector;

class ImageRectifier {
    /**
     * Rectify two given matrix
     * @param leftImage left image to be rectified
     * @param rightImage right image to be rectified
     * @param fundamentalMatrix (3, 3) fundamental matrix corresponds to leftImage
     * and rightImage
     * @param leftKeyPoints vector of KeyPoint from left image
     * @param rightKeyPoints vector of KeyPoint from right image
     * @param matches vector of corresponding matches from vector keypoints
     * */
public:
    ImageRectifier(const cv::Mat &leftImage, const cv::Mat &rightImage,
                   const Matrix3Xf &fundamentalMatrix,
                   const std::vector<cv::KeyPoint> &leftKeyPoints,
                   const std::vector<cv::KeyPoint> &rightKeyPoints,
                   const std::vector<cv::DMatch> &matches);

    ImageRectifier(const cv::Mat &leftImage, const cv::Mat &rightImage,
                   cv::Mat fundamentalMatrix,
                   const std::vector<cv::Point2d> &leftKeyPoints,
                   const std::vector<cv::Point2d> &rightKeyPoints);

    ImageRectifier(const cv::Mat &leftImage,
                   const cv::Mat &rightImage,
                   const Matrix3Xf &fundamentalMatrix,
                   const Matrix3Xf &leftKeyPoints,
                   const Matrix3Xf &rightKeyPoints);

    void run();

    /**
     * Getters
     */

    const cv::Mat &getRectifiedLeft() const;

    const cv::Mat &getRectifiedRight() const;

    const cv::Mat &getDisparityMapLeft() const;
    const cv::Mat &getDisparityMapRight() const;

    vector<cv::Point2d> &getRectifiedLeftMatches();

    vector<cv::Point2d> &getRectifiedRightMatches();

    cv::Mat getH_();

    cv::Mat getHp_();

    /**
     * compute the disparity map of the left image
     */
    void computeDisparityMapLeft(int blockSize, int maxDisparity, double smoothFactor);

	/**
	* compute the disparity map of the left image
	*/
    void computeDisparityMapRight(int blockSize, int maxDisparity, double smoothFactor);

    /**
     * Setters
     */
    void setMatches(const std::vector<cv::KeyPoint> &leftKeyPoints,
                    const std::vector<cv::KeyPoint> &rightKeyPoints,
                    const std::vector<cv::DMatch> &matches);

    void setMatches(const Matrix3Xf &leftKeyPoints,
                    const Matrix3Xf &rightKeyPoints);

    void setFundamentalMatrix(const Matrix3f &fundamentalMatrix);

    static void drawRectifiedEpilines(cv::Mat &leftRectifiedImage,
                                      cv::Mat &rightRectifiedImage,
                                      cv::Mat &fundamentalMatrix,
                                      vector<cv::Point2d> &leftRectifiedMatches,
                                      vector<cv::Point2d> &rightRectifiedMatches,
                                      size_t num_lines);

private:
    // Inputs
    const cv::Mat &leftImage_;
    const cv::Mat &rightImage_;
    cv::Mat fundamentalMatrix_;
    vector<cv::Point2d> rightMatches_;
    vector<cv::Point2d> leftMatches_;

    // Projective step variables
    cv::Vec3d epipole_;
    cv::Mat A_, B_, Ap_, Bp_;
    cv::Mat w_;
    cv::Mat wp_;

    // Similarity step variables
    cv::Mat H_p_, Hp_p_;
    cv::Mat H_r_, Hp_r_;

    // Shearing step variables
    cv::Mat H_1_, H_2_, H_s_, Hp_s_;

    // Rectification step variables
    cv::Mat H_, Hp_;
    vector<cv::Point2d> rightRectifiedMatches_;
    vector<cv::Point2d> leftRectifiedMatches_;
    cv::Mat leftRectifiedImage_;
    cv::Mat rightRectifiedImage_;
    cv::Mat disparityMapLeft;
    cv::Mat disparityMapRight;

	/**
	* Computes and draw epilines from a stereo pair of images.
	*/
    static std::pair<vector<cv::Vec3d>,
            vector<cv::Vec3d>> computeEpiLines(const cv::Mat &leftImage,
                                               const cv::Mat &rightImage,
                                               const cv::Mat &fundamentalMatrix,
                                               cv::Vec3d &epipole,
                                               vector<cv::Point2d> &leftMatches,
                                               vector<cv::Point2d> &rightMatches);

	/**
	* Computes transformation H_p and Hp_p.
	*/
    void computeProjective();

	/**
	* Computes transformation H_r and Hp_r.
	*/
    void computeSimilarity();

	/**
	* Computes shearing transformation H_s_ and Hp_s_.
	*/
    void computeShearingTransforms();

	/**
	* Perform rectification on images and key points
	*/
    void rectifyImagesAndKeyPoints();

    /**
     * Computes A,B matrices.
     * @param[in]  image Image to for with matrix A,B correspond to.
     * @param[in]  mat Multiplicator matrix.
     * @param[in]  A First parameter matrix.
     * @param[in]  B Second parameter matrix.
     */
    void computeAB(const cv::Mat &image, const cv::Mat &mat, cv::Mat &A,
                   cv::Mat &B);

	/**
	* Maximizes the addend from equation 11 in the paper given the A,B matrices
	* @param[in]  A First parameter matrix.
	* @param[in]  B Second parameter matrix.
	* @return The Z vector that maximizes the addend.
	*/
    cv::Vec3d maximizeAddend(const cv::Mat &A, const cv::Mat &B);

    /**
     * Computes the minimum coordinate in Y axis of the image after the homography
     * @param[in]  image Input image.
     * @param[in]  homography Homography.
     * @return Minimum Y coordinate of \p img after \p homography is applied.
     */
    double getMinYCoordinate(const cv::Mat &image, const cv::Mat &homography);

  	/**
	* Computes the auxiliary S matrix for shearing transformation
	* @param[in]  img Image
	* @param[in]  homography Rectifying homography
	* @return Auxiliary matrix.
	*/
    cv::Mat getS(const cv::Mat &img, const cv::Mat &homography);

  	/**
	* Checks whether the image would be inverted after the homography is applied
	* @param[in]  img Image
	* @param[in]  homography Applied homography
	* @return True if the image would be inverted after the homography is applied
	*/
    bool isImageInverted(const cv::Mat &image, const cv::Mat &homography);


	/**
	* Draw corresponding epilines for provided images
	* @param[in] leftImage Left Image
	* @param[in] leftImage Left Imag
	* @param[in] leftEpilines Left Image epilines
	* @param[in] rightEpilines Right Image epilnes
	* @param[in] leftMatches Left matches points
	* @param[in] rightMatches Right matches points
	* @param[in] num_lines Number of lines to draw
	*/
    static void drawEpilines(cv::Mat &leftImage,
                             cv::Mat &rightImage,
                             vector<cv::Vec3d> &leftEpilines,
                             vector<cv::Vec3d> &rightEpilines,
                             vector<cv::Point2d> &leftMatches,
                             vector<cv::Point2d> &rightMatches,
                             size_t num_lines);
};

#endif // STEREO_RECONSTRUCTION_RECTIFICATION_RECTIFICATION_HPP_
