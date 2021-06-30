//
// Created by tim on 22.06.21.
//

#ifndef STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
#define STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H

#include <stdio.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Eigen.h"

/**
 * NOTE: Taken from lecture, exercise 5
 * Helper methods for writing Ceres cost functions.
 */
template<typename T>
static inline void fillVector(const Vector3f &input, T *output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

template<typename T>
static inline void fillVector(const T *const input, T *output) {
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[2];
}

Vector3f rotationToAngleAxis(const Matrix3f &rotation) {
    AngleAxisf aa(rotation);
    return aa.angle() * aa.axis();
}

/**
 * NOTE: Taken from lecture, exercise 5
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template<typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T *const array) : m_array{array} {}

    void setZero() {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    void setData(Vector3f angleAxisRotation, Vector3f translation) {
        for (int i = 0; i < 3; i++)
            m_array[i] = T(angleAxisRotation(i));
        for (int i = 3; i < 6; i++)
            m_array[i] = T(translation(i-3));
    }

    T *getData() const {
        return m_array;
    }

    void print() {
        std::cout << "Printing pose: " << std::endl;
        for (int i=0; i < 6; i++)
            std::cout << m_array[i] << " ";
        std::cout << std::endl;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T *inputPoint, T *outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T *rotation = m_array;
        const T *translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double> &poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double *pose = poseIncrement.getData();
        double *rotation = pose;
        double *translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);
        matrix(0, 1) = float(rotationMatrix[3]);
        matrix(0, 2) = float(rotationMatrix[6]);
        matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);
        matrix(1, 1) = float(rotationMatrix[4]);
        matrix(1, 2) = float(rotationMatrix[7]);
        matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);
        matrix(2, 1) = float(rotationMatrix[5]);
        matrix(2, 2) = float(rotationMatrix[8]);
        matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    T *m_array;
};

/*
class BundleAdjustmentConstraint {
public:
    BundleAdjustmentConstraint(const Vector3f &keypoint, const Matrix3f &intrinsics, const int idx) :
            m_keypoint{keypoint}, m_intrinsics{intrinsics}, m_idx(idx) {}

    template <typename T>
    bool operator()(const T *const vars, T *residuals) const {
        // get pose from vars
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T *const>(vars));

        // get corresponding point from vars

        T point3D[3];
        fillVector(point3D, vars + 6 + m_idx * 3);

        // transform 3d point using current pose
        T transformedPoint3D[3]
        poseIncrement.apply(point3D, &transformedPoint3D);

        // use Eigen to backproject 3D point to 2D image plane
        Vector<T, 3, 1> projectedPoint{transformedPoint3D};
        projectedPoint = (m_intrinsics.cast<T>() * projectedPoint) / projectedPoint[2];

        // residual
        residuals[0] = T(m_keypoint[0]) - projectedPoint[0];
        residuals[1] = T(m_keypoint[0]) - projectedPoint[0];
    }

    static ceres::CostFunction *
    create(const Vector3f &sourcePoint, const Vector3f &targetPoint, const Vector3f &targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>( //FIXME TODO
                new BundleAdjustmentConstraint(sourcePoint, targetPoint, targetNormal, weight)
        );
    }


private:
    const Vector3f m_keypoint;          // in homogenous coordinates
    const Matrix3f m_intrinsics;
    const int m_idx;
};
*/


class SimpleConstraint {
public:
    SimpleConstraint(const Vector3f &keypoint, const Matrix3f &intrinsics, const Vector3f &point3D,
                     bool applyTransformation) :
            m_keypoint{keypoint}, m_intrinsics{intrinsics}, m_point3D(point3D), m_applyTransformation{applyTransformation} {}

    template <typename T>
    bool operator()(const T *const vars, T *residuals) const {
        // get pose from vars
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T *const>(vars));
        // poseIncrement.print();

        T transformedPoint3D[3];

        if (m_applyTransformation) {
            // transform 3d point using current pose
            T point3D[3];
            fillVector(m_point3D, &point3D[0]);
            poseIncrement.apply(&point3D[0], &transformedPoint3D[0]);
        } else {
            // only apply projection 3D > 2D
            fillVector(m_point3D, &transformedPoint3D[0]);
        }

        // use Eigen to backproject 3D point to 2D image plane
        Vector<T, 3> projectedPoint{transformedPoint3D};
        projectedPoint = (m_intrinsics.cast<T>() * projectedPoint) / projectedPoint[2];

        // residual
        residuals[0] = T(m_keypoint[0]) - projectedPoint[0];
        residuals[1] = T(m_keypoint[1]) - projectedPoint[1];

        return true;
    }

    static ceres::CostFunction *
    create(const Vector3f &keypoint, const Matrix3f &intrinsics, const Vector3f &point3D, bool applyTransform) {
        return new ceres::AutoDiffCostFunction<SimpleConstraint, 2, 6>(
                new SimpleConstraint(keypoint, intrinsics, point3D, applyTransform)
        );
    }


private:
    const Vector3f m_keypoint;          /* in pixel coordinates */
    const Matrix3f m_intrinsics;
    const Vector3f m_point3D;
    const bool m_applyTransformation;
};

class BundleAdjustmentOptimizer {
public:
    BundleAdjustmentOptimizer(const Matrix3Xf &matchesLeft,
                              const Matrix3Xf &matchesRight,
                              const Matrix3f &intrinsicsLeft,
                              const Matrix3f &intrinsicsRight,
                              const Matrix3f &initRotation,
                              const Vector3f &initTranslation,
                              const Matrix3Xf &leftPoints3D) : // Todo: Remove and add to optimization
            m_matchesLeft{matchesLeft},
            m_matchesRight{matchesRight},
            m_intrinsicsLeft{intrinsicsLeft},
            m_intrinsicsRight{intrinsicsRight},
            m_initRotation{initRotation},
            m_initTranslation{initTranslation},
            m_leftPoints3D{leftPoints3D} {};

    Matrix4f estimatePose() {
        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        ceres::Problem problem;
        prepareConstraints(poseIncrement, problem);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        return PoseIncrement<double>::convertToMatrix(poseIncrement);
    }

private:

    void prepareConstraints(PoseIncrement<double> &poseIncrement, ceres::Problem &problem) {
        int nPoints = (int) m_matchesLeft.cols();

        poseIncrement.setData(rotationToAngleAxis(m_initRotation), m_initTranslation);

        // TODO
        std::cout << "Comparison AngleAxis conversion: " << std::endl;
        std::cout << m_initRotation << std::endl;
        std::cout << PoseIncrement<double>::convertToMatrix(poseIncrement) << std::endl;

        for (int i = 0; i < nPoints; i++) {
            // left points: do not apply transformation
            bool applyTransformation = false;
            problem.AddResidualBlock(SimpleConstraint::create(m_matchesLeft.col(i), m_intrinsicsLeft, m_leftPoints3D.col(i), applyTransformation),
                                     nullptr, poseIncrement.getData());
            // right points: apply transformation
            applyTransformation = true;
            problem.AddResidualBlock(SimpleConstraint::create(m_matchesRight.col(i), m_intrinsicsRight, m_leftPoints3D.col(i), applyTransformation),
                                     nullptr, poseIncrement.getData());
        }

    }

    void configureSolver(ceres::Solver::Options &options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 10;
        options.num_threads = 1;
    }

    /**
     * Member variables
     */

    Matrix3Xf m_matchesLeft;
    Matrix3Xf m_matchesRight;
    Matrix3f m_intrinsicsLeft;
    Matrix3f m_intrinsicsRight;
    Matrix3f m_initRotation;
    Vector3f m_initTranslation;
    Matrix3Xf m_leftPoints3D;

};

#endif //STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
