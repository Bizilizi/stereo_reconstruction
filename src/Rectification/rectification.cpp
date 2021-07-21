#include "rectification.hpp"

#include <utility>
#include "z_optimization.hpp"

ImageRectifier::ImageRectifier(const cv::Mat &leftImage,
							   const cv::Mat &rightImage,
							   const Matrix3Xf &fundamentalMatrix,
							   const std::vector<cv::KeyPoint> &leftKeyPoints,
							   const std::vector<cv::KeyPoint> &rightKeyPoints,
							   const std::vector<cv::DMatch> &matches)
	: leftImage_{leftImage}, rightImage_{rightImage} {

	setFundamentalMatrix(fundamentalMatrix);
	setMatches(leftKeyPoints, rightKeyPoints, matches);
	run();
}

ImageRectifier::ImageRectifier(const cv::Mat &leftImage,
							   const cv::Mat &rightImage,
							   cv::Mat fundamentalMatrix,
							   const vector<cv::Point2d> &leftKeyPoints,
							   const vector<cv::Point2d> &rightKeyPoints)
	: leftImage_{leftImage},
	  rightImage_{rightImage},
	  fundamentalMatrix_{std::move(fundamentalMatrix)},
	  leftMatches_{leftKeyPoints},
	  rightMatches_{rightKeyPoints} {

	run();
}

void ImageRectifier::run() {
	computeEpiLines(leftImage_,
					rightImage_,
					fundamentalMatrix_,
					epipole_,
					leftMatches_,
					rightMatches_);
	computeProjective();
	computeSimilarity();
	computeShearingTransforms();
	rectifyImagesAndKeyPoints();
}

std::pair<vector<cv::Vec3d>, vector<cv::Vec3d>> ImageRectifier::computeEpiLines(
	const cv::Mat &leftImage,
	const cv::Mat &rightImage,
	const cv::Mat &fundamentalMatrix,
	cv::Vec3d &epipole,
	vector<cv::Point2d> &leftMatches,
	vector<cv::Point2d> &rightMatches) {

	vector<cv::Vec3d> leftLines;
	vector<cv::Vec3d> rightLines;

	computeCorrespondEpilines(leftMatches, 1, fundamentalMatrix,
							  leftLines);
	computeCorrespondEpilines(rightMatches, 2, fundamentalMatrix,
							  rightLines);

	cv::Mat epipole_matrix;
	cv::SVD::solveZ(fundamentalMatrix, epipole_matrix);

	epipole[0] = epipole_matrix.at<double>(0, 0);
	epipole[1] = epipole_matrix.at<double>(1, 0);
	epipole[2] = epipole_matrix.at<double>(2, 0);

	return std::make_pair(leftLines, rightLines);
}

void ImageRectifier::computeAB(const cv::Mat &image, const cv::Mat &mat,
							   cv::Mat &A, cv::Mat &B) {
	auto width = image.cols;
	auto height = image.rows;
	auto size = 3;

	cv::Mat PPt = cv::Mat::zeros(size, size, CV_64F);

	PPt.at<double>(0, 0) = width * width - 1;
	PPt.at<double>(1, 1) = height * height - 1;

	PPt *= (width * height) / 12.0;

	double w_1 = width - 1;
	double h_1 = height - 1;

	double values[3][3] = {{w_1 * w_1, w_1 * h_1, 2 * w_1},
						   {w_1 * h_1, h_1 * h_1, 2 * h_1},
						   {2 * w_1, 2 * h_1, 4}};
	cv::Mat pcpct(size, size, CV_64F, values);

	pcpct /= 4;

	A = mat.t() * PPt * mat;
	B = mat.t() * pcpct * mat;
}

void ImageRectifier::computeProjective() {

	double values[3][3] = {{0, -epipole_[2], epipole_[1]},
						   {epipole_[2], 0, -epipole_[0]},
						   {-epipole_[1], epipole_[0], 0}};
	cv::Mat ep_mat(3, 3, CV_64F, values);

	// Compute A,B
	computeAB(leftImage_, ep_mat, A_, B_);
	computeAB(rightImage_, fundamentalMatrix_, Ap_, Bp_);

	// Initial guess for z variable
	cv::Vec3d raw_z = (normalize(maximizeAddend(A_, B_)) +
		normalize(maximizeAddend(Ap_, Bp_))) /
		2;

	// Optimizes the z solution
	raw_z[0] = NewtonRaphson(A_, B_, Ap_, Bp_, raw_z[0]);
	raw_z[1] = 1.0;
	raw_z[2] = 0.0;
	cv::Mat z = cv::Mat(raw_z);

	// Get w
	w_ = ep_mat * z;
	wp_ = fundamentalMatrix_ * z;

	w_ /= w_.at<double>(2, 0);
	wp_ /= wp_.at<double>(2, 0);

	// H_p and Hp_p projection matrix
	H_p_ = cv::Mat::eye(3, 3, CV_64F);
	H_p_.at<double>(2, 0) = w_.at<double>(0, 0);
	H_p_.at<double>(2, 1) = w_.at<double>(1, 0);

	Hp_p_ = cv::Mat::eye(3, 3, CV_64F);
	Hp_p_.at<double>(2, 0) = wp_.at<double>(0, 0);
	Hp_p_.at<double>(2, 1) = wp_.at<double>(1, 0);
}

void ImageRectifier::computeSimilarity() {
	auto min_left = getMinYCoordinate(leftImage_, H_p_);
	auto min_right = getMinYCoordinate(rightImage_, Hp_p_);

	auto offset = min_left < min_right ? min_left : min_right;
	double vp_c = -offset;

	// Get the H_r and Hp_r matrix directly
	H_r_ = cv::Mat::zeros(3, 3, CV_64F);

	H_r_.at<double>(0, 0) =
		fundamentalMatrix_.at<double>(2, 1) -
			w_.at<double>(1, 0) * fundamentalMatrix_.at<double>(2, 2);
	H_r_.at<double>(1, 0) =
		fundamentalMatrix_.at<double>(2, 0) -
			w_.at<double>(0, 0) * fundamentalMatrix_.at<double>(2, 2);

	H_r_.at<double>(0, 1) =
		w_.at<double>(0, 0) * fundamentalMatrix_.at<double>(2, 2) -
			fundamentalMatrix_.at<double>(2, 0);
	H_r_.at<double>(1, 1) = H_r_.at<double>(0, 0);

	H_r_.at<double>(1, 2) = fundamentalMatrix_.at<double>(2, 2) + vp_c;
	H_r_.at<double>(2, 2) = 1.0;

	Hp_r_ = cv::Mat::zeros(3, 3, CV_64F);

	Hp_r_.at<double>(0, 0) =
		wp_.at<double>(1, 0) * fundamentalMatrix_.at<double>(2, 2) -
			fundamentalMatrix_.at<double>(1, 2);
	Hp_r_.at<double>(1, 0) =
		wp_.at<double>(0, 0) * fundamentalMatrix_.at<double>(2, 2) -
			fundamentalMatrix_.at<double>(0, 2);

	Hp_r_.at<double>(0, 1) =
		fundamentalMatrix_.at<double>(0, 2) -
			wp_.at<double>(0, 0) * fundamentalMatrix_.at<double>(2, 2);
	Hp_r_.at<double>(1, 1) = Hp_r_.at<double>(0, 0);

	Hp_r_.at<double>(1, 2) = vp_c;
	Hp_r_.at<double>(2, 2) = 1.0;
}

cv::Mat ImageRectifier::getS(const cv::Mat &img, const cv::Mat &homography) {
	int w = img.cols;
	int h = img.rows;

	cv::Point2d a((w - 1) / 2, 0);
	cv::Point2d b(w - 1, (h - 1) / 2);
	cv::Point2d c((w - 1) / 2, h - 1);
	cv::Point2d d(0, (h - 1) / 2);

	vector<cv::Point2d> midpoints, midpoints_hat;
	midpoints.push_back(a);
	midpoints.push_back(b);
	midpoints.push_back(c);
	midpoints.push_back(d);

	perspectiveTransform(midpoints, midpoints_hat, homography);

	cv::Point2d x = midpoints_hat[1] - midpoints_hat[3];
	cv::Point2d y = midpoints_hat[2] - midpoints_hat[0];

	double coeff_a = (h * h * x.y * x.y + w * w * y.y * y.y) /
		(h * w * (x.y * y.x - x.x * y.y));
	double coeff_b = (h * h * x.x * x.y + w * w * y.x * y.y) /
		(h * w * (x.x * y.y - x.y * y.x));

	cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
	S.at<double>(0, 0) = coeff_a;
	S.at<double>(0, 1) = coeff_b;

	cv::Vec3d x_hom(x.x, x.y, 0.0);
	cv::Vec3d y_hom(y.x, y.y, 0.0);

	if (coeff_a < 0) {
		coeff_a *= -1;
		coeff_b *= -1;

		S.at<double>(0, 0) = coeff_a;
		S.at<double>(0, 1) = coeff_b;
	}

	cv::Mat EQ18 = (S * cv::Mat(x_hom)).t() * (S * cv::Mat(y_hom));

	cv::Mat EQ19 = ((S * cv::Mat(x_hom)).t() * (S * cv::Mat(x_hom))) /
		((S * cv::Mat(y_hom)).t() * (S * cv::Mat(y_hom))) -
		(1. * w * w) / (1. * h * h);

	return S;
}

void ImageRectifier::computeShearingTransforms() {

	H_1_ = H_r_ * H_p_;
	H_2_ = Hp_r_ * Hp_p_;

	cv::Mat S = getS(leftImage_, H_1_);
	cv::Mat Sp = getS(rightImage_, H_2_);

	double A =
		leftImage_.cols * leftImage_.rows + rightImage_.cols * rightImage_.rows;
	double Ap = 0;

	vector<cv::Point2f> corners(4), corners_trans(4);

	corners[0] = cv::Point2f(0, 0);
	corners[1] = cv::Point2f(leftImage_.cols, 0);
	corners[2] = cv::Point2f(leftImage_.cols, leftImage_.rows);
	corners[3] = cv::Point2f(0, leftImage_.rows);

	perspectiveTransform(corners, corners_trans, S * H_1_);
	Ap += contourArea(corners_trans);

	float min_x_1, min_y_1 = std::numeric_limits<float>::infinity();

	for (int j = 0; j < 4; j++) {
		min_x_1 = std::min(corners_trans[j].x, min_x_1);
		min_y_1 = std::min(corners_trans[j].y, min_y_1);
	}

	corners[0] = cv::Point2f(0, 0);
	corners[1] = cv::Point2f(rightImage_.cols, 0);
	corners[2] = cv::Point2f(rightImage_.cols, rightImage_.rows);
	corners[3] = cv::Point2f(0, rightImage_.rows);

	perspectiveTransform(corners, corners_trans, Sp * H_2_);
	Ap += contourArea(corners_trans);

	float min_x_2, min_y_2 = std::numeric_limits<float>::infinity();
	for (int j = 0; j < 4; j++) {
		min_x_2 = std::min(corners_trans[j].x, min_x_2);
		min_y_2 = std::min(corners_trans[j].y, min_y_2);
	}

	double scale = sqrt(A / Ap);

	double min_y = min_y_1 < min_y_2 ? min_y_1 : min_y_2;

	// We define W2 as the scale transformation and W1 as the translation
	// transformation. Then, W = W1*W2.

	cv::Mat W;
	cv::Mat Wp;

	cv::Mat W_1 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Wp_1 = cv::Mat::eye(3, 3, CV_64F);

	cv::Mat W_2 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Wp_2 = cv::Mat::eye(3, 3, CV_64F);

	W_2.at<double>(0, 0) = W_2.at<double>(1, 1) = scale;
	Wp_2.at<double>(0, 0) = Wp_2.at<double>(1, 1) = scale;

	if (isImageInverted(leftImage_, W_2 * H_1_)) {
		W_2.at<double>(0, 0) = W_2.at<double>(1, 1) = -scale;
		Wp_2.at<double>(0, 0) = Wp_2.at<double>(1, 1) = -scale;
	}

	corners[0] = cv::Point2d(0, 0);
	corners[1] = cv::Point2d(leftImage_.cols, 0);
	corners[2] = cv::Point2d(leftImage_.cols, leftImage_.rows);
	corners[3] = cv::Point2d(0, leftImage_.rows);

	perspectiveTransform(corners, corners_trans, W_2 * S * H_1_);

	min_x_1 = min_y_1 = std::numeric_limits<float>::infinity();
	for (int j = 0; j < 4; j++) {
		min_x_1 = std::min(corners_trans[j].x, min_x_1);
		min_y_1 = std::min(corners_trans[j].y, min_y_1);
	}

	corners[0] = cv::Point2d(0, 0);
	corners[1] = cv::Point2d(rightImage_.cols, 0);
	corners[2] = cv::Point2d(rightImage_.cols, rightImage_.rows);
	corners[3] = cv::Point2d(0, rightImage_.rows);

	perspectiveTransform(corners, corners_trans, Wp_2 * Sp * H_2_);

	min_x_2 = min_y_2 = std::numeric_limits<float>::infinity();
	for (int j = 0; j < 4; j++) {
		min_x_2 = std::min(corners_trans[j].x, min_x_2);
		min_y_2 = std::min(corners_trans[j].y, min_y_2);
	}

	min_y = min_y_1 < min_y_2 ? min_y_1 : min_y_2;

	W_1.at<double>(0, 2) = -min_x_1;
	Wp_1.at<double>(0, 2) = -min_x_2;

	W_1.at<double>(1, 2) = Wp_1.at<double>(1, 2) = -min_y;

	W = W_1 * W_2;
	Wp = Wp_1 * Wp_2;

	H_s_ = W * S;
	Hp_s_ = Wp * Sp;
}

cv::Vec3d ImageRectifier::maximizeAddend(const cv::Mat &A, const cv::Mat &B) {
	cv::Mat D = cv::Mat::zeros(3, 3, CV_64F);
	choleskyCustomDecomposition(A, D);

	cv::Mat D_inv = D.inv();
	cv::Mat DBD = D_inv.t() * B * D_inv;

	// Solve the DBD equation
	cv::Mat eigenvalues, eigenvectors;
	eigen(DBD, eigenvalues, eigenvectors);

	// Pick the largest eigen vector
	cv::Mat y = eigenvectors.row(0);
	cv::Mat solution = D_inv * y.t();

	return cv::Vec3d(solution.at<double>(0, 0), solution.at<double>(1, 0),
					 solution.at<double>(2, 0));
}

double ImageRectifier::getMinYCoordinate(const cv::Mat &image,
										 const cv::Mat &homography) {
	vector<cv::Point2d> corners(4), corners_trans(4);

	corners[0] = cv::Point2d(0, 0);
	corners[1] = cv::Point2d(image.cols, 0);
	corners[2] = cv::Point2d(image.cols, image.rows);
	corners[3] = cv::Point2d(0, image.rows);

	perspectiveTransform(corners, corners_trans, homography);

	auto min_y = std::numeric_limits<double>::infinity();

	for (int j = 0; j < 4; j++) {
		min_y = std::min(corners_trans[j].y, min_y);
	}

	return min_y;
}

bool ImageRectifier::isImageInverted(const cv::Mat &image,
									 const cv::Mat &homography) {
	vector<cv::Point2d> corners(2), corners_trans(2);

	corners[0] = cv::Point2d(0, 0);
	corners[1] = cv::Point2d(0, image.rows);

	perspectiveTransform(corners, corners_trans, homography);

	return corners_trans[1].y - corners_trans[0].y < 0.0;
}

void ImageRectifier::rectifyImagesAndKeyPoints() {
	H_ = H_s_ * H_r_ * H_p_;
	Hp_ = Hp_s_ * Hp_r_ * Hp_p_;

	// Get homography image of the corner coordinates from all the images
	vector<cv::Point2d> corners_all(4), corners_all_t(4);
	double min_x, min_y, max_x, max_y;
	min_x = min_y = std::numeric_limits<double>::infinity();
	max_x = max_y = -std::numeric_limits<double>::infinity();

	corners_all[0] = cv::Point2d(0, 0);
	corners_all[1] = cv::Point2d(leftImage_.cols, 0);
	corners_all[2] = cv::Point2d(leftImage_.cols, leftImage_.rows);
	corners_all[3] = cv::Point2d(0, leftImage_.rows);

	perspectiveTransform(corners_all, corners_all_t, H_);

	for (int j = 0; j < 4; j++) {
		min_x = std::min(corners_all_t[j].x, min_x);
		max_x = std::max(corners_all_t[j].x, max_x);

		min_y = std::min(corners_all_t[j].y, min_y);
		max_y = std::max(corners_all_t[j].y, max_y);
	}

	int left_img_cols = max_x - min_x;
	int left_img_rows = max_y - min_y;

	// Get homography image of the corner coordinates from all the images
	min_x = min_y = std::numeric_limits<double>::infinity();
	max_x = max_y = -std::numeric_limits<double>::infinity();

	corners_all[0] = cv::Point2d(0, 0);
	corners_all[1] = cv::Point2d(rightImage_.cols, 0);
	corners_all[2] = cv::Point2d(rightImage_.cols, rightImage_.rows);
	corners_all[3] = cv::Point2d(0, rightImage_.rows);

	perspectiveTransform(corners_all, corners_all_t, Hp_);

	for (int j = 0; j < 4; j++) {
		min_x = std::min(corners_all_t[j].x, min_x);
		max_x = std::max(corners_all_t[j].x, max_x);

		min_y = std::min(corners_all_t[j].y, min_y);
		max_y = std::max(corners_all_t[j].y, max_y);
	}

	int right_img_cols = max_x - min_x;
	int right_img_rows = max_y - min_y;

	// Apply homographies
	leftRectifiedImage_ = cv::Mat(left_img_rows, left_img_cols, CV_64F);
	rightRectifiedImage_ = cv::Mat(right_img_rows, right_img_cols, CV_64F);

	warpPerspective(leftImage_,
					leftRectifiedImage_,
					H_,
					leftRectifiedImage_.size());
	warpPerspective(rightImage_,
					rightRectifiedImage_,
					Hp_,
					rightRectifiedImage_.size());

	perspectiveTransform(leftMatches_, leftRectifiedMatches_, H_);
	perspectiveTransform(rightMatches_, rightRectifiedMatches_, Hp_);
}

const cv::Mat &ImageRectifier::getRectifiedLeft() const {
	return leftRectifiedImage_;
}

const cv::Mat &ImageRectifier::getRectifiedRight() const {
	return rightRectifiedImage_;
}

vector<cv::Point2d> & ImageRectifier::getRectifiedLeftMatches(){
	return leftRectifiedMatches_;
}

vector<cv::Point2d> & ImageRectifier::getRectifiedRightMatches(){
	return rightRectifiedMatches_;
}

cv::Mat ImageRectifier::getH_() {
    return H_;
}

cv::Mat ImageRectifier::getHp_() {
    return Hp_;
}

void ImageRectifier::setMatches(const vector<cv::KeyPoint> &leftKeyPoints,
								const vector<cv::KeyPoint> &rightKeyPoints,
								const std::vector<cv::DMatch> &matches) {
	// convert keypoints to Point2d
	for (const auto &match : matches) {
		leftMatches_.push_back(leftKeyPoints[match.queryIdx].pt);
		rightMatches_.push_back(rightKeyPoints[match.queryIdx].pt);
	}
}

void ImageRectifier::setFundamentalMatrix(const Matrix3f &fundamentalMatrix) {
	// convert fundamental matrix to cv::Mat
	eigen2cv(fundamentalMatrix, fundamentalMatrix_);
	this->fundamentalMatrix_.convertTo(fundamentalMatrix_, CV_64F);
}

void ImageRectifier::drawRectifiedEpilines(cv::Mat &leftRectifiedImage,
										   cv::Mat &rightRectifiedImage,
										   cv::Mat &fundamentalMatrix,
										   vector<cv::Point2d> &leftRectifiedMatches,
										   vector<cv::Point2d> &rightRectifiedMatches,
										   size_t num_lines) {

	cv::Vec3d rectified_epipole;

	auto rectifiedEpilines = computeEpiLines(leftRectifiedImage,
											 rightRectifiedImage,
											 fundamentalMatrix,
											 rectified_epipole,
											 leftRectifiedMatches,
											 rightRectifiedMatches);

	drawEpilines(leftRectifiedImage,
				 rightRectifiedImage,
				 rectifiedEpilines.first,
				 rectifiedEpilines.second,
				 leftRectifiedMatches,
				 rightRectifiedMatches,
				 num_lines);
}

void ImageRectifier::drawEpilines(cv::Mat &leftImage,
								  cv::Mat &rightImage,
								  vector<cv::Vec3d> &leftEpilines,
								  vector<cv::Vec3d> &rightEpilines,
								  vector<cv::Point2d> &leftMatches,
								  vector<cv::Point2d> &rightMatches,
								  size_t num_lines) {
	cv::RNG rng;
	cv::theRNG().state = clock();

	for (size_t i = 0; i < leftEpilines.size(); i++) {
		cv::Vec2d leftPoint = leftMatches[i];
		cv::Vec2d rightPoint = rightMatches[i];

		cv::Vec3d leftLine = leftEpilines[i];
		cv::Vec3d rightLine = rightEpilines[i];
		// Draws only num_lines lines
		if (i % (leftEpilines.size() / num_lines) == 0) {
			cv::Scalar color(rng.uniform(0, 255),
							 rng.uniform(0, 255),
							 rng.uniform(0, 255));

			line(leftImage,
				 cv::Point(0, -leftLine[2] / leftLine[1]),
				 cv::Point(leftImage.cols,
						   -(leftLine[2] + leftLine[0] * leftImage.cols)
							   / leftLine[1]),
				 color
			);
			circle(leftImage,
				   cv::Point2d(leftPoint[0], leftPoint[1]),
				   4,
				   color,
				   cv::FILLED);

			line(rightImage,
				 cv::Point(0,
						   -rightLine[2] / rightLine[1]),
				 cv::Point(rightImage.cols,
						   -(rightLine[2] + rightLine[0] * rightImage.cols)
							   / rightLine[1]),
				 color
			);
			circle(rightImage,
				   cv::Point2d(rightPoint[0], rightPoint[1]),
				   4,
				   color,
				   cv::FILLED);

		}
	}
}
