//
// Created by tim on 22.06.21.
//

#ifndef STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
#define STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H

#include <stdio.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Eigen.h"

#define N_POINTS 9      //TODO: change to 12 later

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


class BundleAdjustmentConstraint {
public:
    BundleAdjustmentConstraint(const Vector3f &keypoint, const Matrix3f &intrinsics, int idx, bool applyTransformation) :
            m_keypoint{keypoint}, m_intrinsics{intrinsics}, m_idx{idx}, m_applyTransformation{applyTransformation} {}

     template <typename T>
    bool operator()(const T *const vars, T *residuals) const {
        // get pose from vars
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T *const>(vars));
        // poseIncrement.print();

        T transformedPoint3D[3];

        if (m_applyTransformation) {
            // transform 3d point using current pose
            T point3D[3];
            fillVector(vars + 6 + 3*m_idx, &point3D[0]);
            poseIncrement.apply(&point3D[0], &transformedPoint3D[0]);
        } else {
            // only apply projection 3D > 2D
            fillVector(vars + 6 + 3*m_idx, &transformedPoint3D[0]);
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
    create(const Vector3f &keypoint, const Matrix3f &intrinsics, int idx, bool applyTransform) {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentConstraint, 2, 6 + 3 * N_POINTS>(
                new BundleAdjustmentConstraint(keypoint, intrinsics, idx, applyTransform)
        );
    }


private:
    const Vector3f m_keypoint;          // in homogenous coordinates
    const Matrix3f m_intrinsics;
    const int m_idx;
    const bool m_applyTransformation;
};


#if 0
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

#endif

class BundleAdjustmentOptimizer {
public:
    BundleAdjustmentOptimizer(const Matrix3Xf &matchesLeft,
                              const Matrix3Xf &matchesRight,
                              const Matrix3f &intrinsicsLeft,
                              const Matrix3f &intrinsicsRight,
                              const Matrix3f &initRotation,
                              const Vector3f &initTranslation,
                              const Matrix3Xf &initLeftPoints3D) : // Todo: Remove and add to optimization
            m_matchesLeft{matchesLeft},
            m_matchesRight{matchesRight},
            m_intrinsicsLeft{intrinsicsLeft},
            m_intrinsicsRight{intrinsicsRight},
            m_initRotation{initRotation},
            m_initTranslation{initTranslation},
            m_initLeftPoints3D{initLeftPoints3D} {
        if (initLeftPoints3D.cols() != N_POINTS){
            throw std::runtime_error("BundleAdjustmentOptimizer: Number of points does not match.");
        }
        m_optimizedLeftPoints3D = MatrixXf::Zero(3, N_POINTS);
    };

    Matrix3Xf getOptimized3DPoints(){
        return m_optimizedLeftPoints3D;
    };

    Matrix4f estimatePose() {
        double vars[6 + N_POINTS*3];
        auto poseIncrement = PoseIncrement<double>(vars);
        ceres::Problem problem;
        prepareConstraints(poseIncrement, vars, problem);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        // Run the solver
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        // Save and return results
        m_optimizedPose = PoseIncrement<double>::convertToMatrix(poseIncrement);

        for(int i = 0; i < N_POINTS; i++){
            m_optimizedLeftPoints3D(0, i) = vars[6 + 3*i];
            m_optimizedLeftPoints3D(1, i) = vars[6 + 3*i +1];
            m_optimizedLeftPoints3D(2, i) = vars[6 + 3*i +2];
        }

        return m_optimizedPose;
    }

private:

    void prepareConstraints(PoseIncrement<double> &poseIncrement, double* vars, ceres::Problem &problem) {
        int nPoints = (int) m_matchesLeft.cols();

        // inititalize pose and 3D points
        poseIncrement.setData(rotationToAngleAxis(m_initRotation), m_initTranslation);
        for(int i = 0; i < N_POINTS; i++){
            vars[6+ i*3] = m_initLeftPoints3D(0, i);
            vars[6+ i*3 +1] = m_initLeftPoints3D(1, i);
            vars[6+ i*3 +2] = m_initLeftPoints3D(2, i);
        }

        // initialize residuals
        for (int i = 0; i < nPoints; i++) {
            // left points: do not apply transformation
            bool applyTransformation = false;
            problem.AddResidualBlock(BundleAdjustmentConstraint::create(m_matchesLeft.col(i), m_intrinsicsLeft, i, applyTransformation),
                                     nullptr, poseIncrement.getData());
            // right points: apply transformation
            applyTransformation = true;
            problem.AddResidualBlock(BundleAdjustmentConstraint::create(m_matchesRight.col(i), m_intrinsicsRight, i, applyTransformation),
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
    Matrix3Xf m_initLeftPoints3D;

    Matrix4f m_optimizedPose;
    Matrix3Xf m_optimizedLeftPoints3D;
};

#endif //STEREO_RECONSTRUCTION_BUNDLEADJUSTMENT_H
