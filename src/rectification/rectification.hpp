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

  void run();

  /**
   * Getters
   */

  const cv::Mat &getRectifiedLeft() const;

  const cv::Mat &getRectifiedRight() const;

  vector<cv::Point2d> &getRectifiedLeftMatches();

  vector<cv::Point2d> &getRectifiedRightMatches();

  cv::Mat getH_();
  cv::Mat getHp_();

  /**
   * Setters
   */

  void setMatches(const std::vector<cv::KeyPoint> &leftKeyPoints,
				  const std::vector<cv::KeyPoint> &rightKeyPoints,
				  const std::vector<cv::DMatch> &matches);

  void setFundamentalMatrix(const Matrix3f &fundamentalMatrix);
  static void drawRectifiedEpilines(cv::Mat &leftRectifiedImage,
									cv::Mat &rightRectifiedImage,
									cv::Mat &fundamentalMatrix,
									vector<cv::Point2d> &leftRectifiedMatches,
									vector<cv::Point2d> &rightRectifiedMatches,
									size_t num_lines);

 private:
  // inputs
  const cv::Mat &leftImage_;
  const cv::Mat &rightImage_;
  cv::Mat fundamentalMatrix_;
  vector<cv::Point2d> rightMatches_;
  vector<cv::Point2d> leftMatches_;

  // projective
  cv::Vec3d epipole_;
  cv::Mat A_, B_, Ap_, Bp_;
  cv::Mat w_;
  cv::Mat wp_;

  // similarity
  cv::Mat H_p_, Hp_p_;
  cv::Mat H_r_, Hp_r_;

  // shearing
  cv::Mat H_1_, H_2_, H_s_, Hp_s_;

  // rectification
  cv::Mat H_, Hp_;
  vector<cv::Point2d> rightRectifiedMatches_;
  vector<cv::Point2d> leftRectifiedMatches_;
  cv::Mat leftRectifiedImage_;
  cv::Mat rightRectifiedImage_;

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
  void computeShearingTransforms();
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

  cv::Mat getS(const cv::Mat &img, const cv::Mat &homography);

  bool isImageInverted(const cv::Mat &image, const cv::Mat &homography);

  static void drawEpilines(cv::Mat &leftImage,
						   cv::Mat &rightImage,
						   vector<cv::Vec3d> &leftEpilines,
						   vector<cv::Vec3d> &rightEpilines,
						   vector<cv::Point2d> &leftMatches,
						   vector<cv::Point2d> &rightMatches,
						   size_t num_lines);
};

#endif // STEREO_RECONSTRUCTION_RECTIFICATION_RECTIFICATION_HPP_
