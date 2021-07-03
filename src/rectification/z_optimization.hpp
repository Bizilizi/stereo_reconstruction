#ifndef STEREO_RECONSTRUCTION_RECTIFICATION_Z_OPTIMIZATION_HPP_
#define STEREO_RECONSTRUCTION_RECTIFICATION_Z_OPTIMIZATION_HPP_
/**
 * Boilerplate related to NewtonRaphson optimization for z parameter
 * */

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
using std::vector;

vector<vector<double>> MatToVector(const cv::Mat& matrix) {
	vector< vector<double> > result;

	for (size_t i = 0; i < matrix.rows; i++) {
		vector<double> row;

		for (size_t j = 0; j < matrix.cols; j++) {
			row.push_back(matrix.at<double>(i,j));
		}

		result.push_back(row);
	}

	return result;
}

double Function(const cv::Mat &A, const cv::Mat &B, const cv::Mat &Ap,
				const cv::Mat &Bp, double x) {
	vector<vector<double>> a = MatToVector(A);
	vector<vector<double>> b = MatToVector(B);
	vector<vector<double>> ap = MatToVector(Ap);
	vector<vector<double>> bp = MatToVector(Bp);

	double summ_1 = (2 * ap[0][0] * x + ap[1][0] + ap[0][1]) /
		(x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1]);

	double den_summ_2 = x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1];
	den_summ_2 = den_summ_2 * den_summ_2;

	double summ_2 = ((2 * bp[0][0] * x + bp[1][0] + bp[0][1]) *
		(x * (ap[0][0] * x + ap[0][1]) + ap[1][0] * x + ap[1][1])) /
		den_summ_2;

	double summ_3 = (2 * a[0][0] * x + a[1][0] + a[0][1]) /
		(x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1]);

	double den_summ_4 = x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1];
	den_summ_4 = den_summ_4 * den_summ_4;

	double summ_4 = ((2 * b[0][0] * x + b[1][0] + b[0][1]) *
		(x * (a[0][0] * x + a[0][1]) + a[1][0] * x + a[1][1])) /
		den_summ_4;

	return summ_1 - summ_2 + summ_3 - summ_4;
}

double derivative(const cv::Mat &A, const cv::Mat &B, const cv::Mat &Ap,
				  const cv::Mat &Bp, double x) {
	vector<vector<double>> a = MatToVector(A);
	vector<vector<double>> b = MatToVector(B);
	vector<vector<double>> ap = MatToVector(Ap);
	vector<vector<double>> bp = MatToVector(Bp);

	double summ_1 = (2 * ap[0][0]) /
		(x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1]);

	double
		den_summ_2 = (x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1]);
	den_summ_2 = den_summ_2 * den_summ_2;

	double summ_2 = (2 * bp[0][0] *
		(x * (ap[0][0] * x + ap[0][1]) + ap[1][0] * x + ap[1][1])) /
		den_summ_2;

	double
		den_summ_3 = (x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1]);
	den_summ_3 = den_summ_3 * den_summ_3;

	double summ_3 = (2 * (2 * ap[0][0] * x + ap[1][0] + ap[0][1]) *
		(2 * bp[0][0] * x + bp[1][0] + bp[0][1])) /
		den_summ_3;

	double
		den_summ_4 = (x * (bp[0][0] * x + bp[0][1]) + bp[1][0] * x + bp[1][1]);
	den_summ_4 = den_summ_4 * den_summ_4 * den_summ_4;

	double aux_num_summ_4 = (2 * bp[0][0] * x + bp[1][0] + bp[0][1]);
	aux_num_summ_4 = aux_num_summ_4 * aux_num_summ_4;

	double summ_4 = (2 * aux_num_summ_4 *
		(x * (ap[0][0] * x + ap[0][1]) + ap[1][0] * x + ap[1][1])) /
		den_summ_4;

	double summ_5 =
		(2 * a[0][0]) / (x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1]);

	double den_summ_6 = (x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1]);
	den_summ_6 = den_summ_6 * den_summ_6;

	double summ_6 =
		(2 * b[0][0] * (x * (a[0][0] * x + a[0][1]) + a[1][0] * x + a[1][1])) /
			den_summ_6;

	double den_summ_7 = (x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1]);
	den_summ_7 = den_summ_7 * den_summ_7;

	double summ_7 = (2 * (2 * a[0][0] * x + a[1][0] + a[0][1]) *
		(2 * b[0][0] * x + b[1][0] + b[0][1])) /
		den_summ_7;

	double den_summ_8 = (x * (b[0][0] * x + b[0][1]) + b[1][0] * x + b[1][1]);
	den_summ_8 = den_summ_8 * den_summ_8 * den_summ_8;

	double aux_num_summ_8 = (2 * b[0][0] * x + b[1][0] + b[0][1]);
	aux_num_summ_8 = aux_num_summ_8 * aux_num_summ_8;

	double summ_8 = (2 * aux_num_summ_8 *
		(x * (a[0][0] * x + a[0][1]) + a[1][0] * x + a[1][1])) /
		den_summ_8;

	return summ_1 - summ_2 - summ_3 + summ_4 + summ_5 - summ_6 - summ_7
		+ summ_8;
}

double NewtonRaphson(const cv::Mat &A, const cv::Mat &B, const cv::Mat &Ap,
					 const cv::Mat &Bp, double init_guess) {
	double current = init_guess;

	double fx = Function(A, B, Ap, Bp, current);
	double dfx = derivative(A, B, Ap, Bp, current);

	int iterations = 0;

	do {
		current = current - fx / dfx;

		fx = Function(A, B, Ap, Bp, current);
		dfx = derivative(A, B, Ap, Bp, current);

		iterations++;
	} while (abs(fx) > 1e-15 && iterations < 150);

	return current;
}

bool choleskyCustomDecomposition(const cv::Mat &A, cv::Mat &L) {

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j <= i; j++) {
			double sum = 0;
			for (int k = 0; k < j; k++) {
				sum += L.at<double>(i, k) * L.at<double>(j, k);
			}

			L.at<double>(i, j) = A.at<double>(i, j) - sum;
			if (i == j) {
				if (L.at<double>(i, j) < 0.0) {
					if (L.at<double>(i, j) > -1e-5) {
						L.at<double>(i, j) *= -1;
					} else {
						return false;
					}
				}
				L.at<double>(i, j) = sqrt(L.at<double>(i, j));
			} else {
				L.at<double>(i, j) /= L.at<double>(j, j);
			}
		}
	}

	L = L.t();

	return true;
}
#endif // STEREO_RECONSTRUCTION_RECTIFICATION_Z_OPTIMIZATION_HPP_