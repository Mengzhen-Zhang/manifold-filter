use nalgebra::{RealField, ComplexField, SMatrix, SVector, Scalar};

use crate::diff::Diff;

pub mod product;
pub use product::*;

pub mod rn;
pub use rn::*;

pub mod so2;
pub use so2::*;

pub trait Manifold<S, const A_DIM: usize, const T_DIM: usize> {
    type TangentVector;

    // x_next = x ⊕ δx
    fn retract(&self, delta: &Self::TangentVector) -> Self;

    // δx = x₂ ⊖ x₁
    fn local_lift(&self, other: &Self) -> Self::TangentVector;

    // TₓM → TₓA
    fn pushforward_jacobian(&self) -> SMatrix<S, A_DIM, T_DIM>;

    // jacobian vector product
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, A_DIM>;

    fn vector_to_tangent(vec: &SVector<S, T_DIM>) -> Self::TangentVector;

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, T_DIM>;
}

pub trait InvariantLieGroup<S, const A_DIM: usize, const T_DIM: usize>: Manifold<S, A_DIM, T_DIM> {
    // lie exp
    fn exp(omega: &Self::TangentVector) -> Self;

    // lie log
    fn log(&self) -> Self::TangentVector;

    // lie group multiplication
    fn compose(&self, other: &Self) -> Self;

    // lie group inverse
    fn inverse(&self) -> Self;

    // Ad_g
    fn adjoint(&self) -> SMatrix<S, T_DIM, T_DIM>;

    // ad_omega
    fn small_adjoint(omega: &Self::TangentVector) -> SMatrix<S, T_DIM, T_DIM>;
}

pub trait TimeVaryingConstraint<S, const A_DIM: usize, const T_DIM: usize, const C_DIM: usize> {
    /// The user implements the math using a raw function pointer or closure wrapped by your Diff engines
    fn constraint_engine(&self) -> &impl Diff<S, A_DIM, C_DIM>;

    #[inline(always)]
    fn evaluate(&self, ambient_state: &SVector<S, A_DIM>) -> SVector<S, C_DIM> {
        self.constraint_engine().eval(ambient_state)
    }

    /// Evaluates the explicit ambient Jacobian matrix H_ambient (C_DIM x A_DIM)
    #[inline(always)]
    fn ambient_jacobian(&self, ambient_state: &SVector<S, A_DIM>) -> SMatrix<S, C_DIM, A_DIM> {
        self.constraint_engine().jacobian(ambient_state)
    }
}

impl InvariantLieGroup<f64, 2, 1> for ComplexRotation {
    fn exp(omega: &Self::TangentVector) -> Self {
        Self { z: SVector::<f64, 2>::new(omega.cos(), omega.sin()) }
    }

    fn log(&self) -> Self::TangentVector {
        self.z[1].atan2(self.z[0])
    }

    fn compose(&self, other: &Self) -> Self {
        let x = self.z[0] * other.z[0] - self.z[1] * other.z[1];
        let y = self.z[0] * other.z[1] + self.z[1] * other.z[0];
        Self { z: SVector::<f64, 2>::new(x, y) }
    }

    fn inverse(&self) -> Self {
        Self { z: SVector::<f64, 2>::new(self.z[0], -self.z[1]) }
    }

    fn adjoint(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::identity() // SO(2) is abelian, Ad_g = Identity
    }

    fn small_adjoint(_omega: &Self::TangentVector) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::zeros()    // ad_omega = 0
    }
}

#[cfg(test)]
mod integration_tests {
    use nalgebra::{Vector1, Vector2, Vector3, SMatrix};

    use super::*;
    use crate::dual::Dual;
    use crate::diff::{AutoDiff, Diff}; // Assuming AutoDiff is exposed in your diff module

    // -------------------------------------------------------------------------
    // 1. Defining a Concrete Constraint using your AutoDiff Engine
    // Let's constrain our RealLine state to sit on a moving boundary:
    // h(x, t) = x - 2.0 * t = 0
    // -------------------------------------------------------------------------
    pub struct MovingBoundaryConstraint {
        engine: AutoDiff<f64, 1, 1>,
    }

    impl MovingBoundaryConstraint {
        pub fn new() -> Self {
            // Function signature matching your AutoDiff framework: f(x, y)
            fn constraint_formula(x: &SVector<Dual<f64>, 1>, y: &mut SVector<Dual<f64>, 1>) {
                // Fixed target position at 5.0 for this simple snapshot test
                y[0] = x[0] - Dual::from_re(5.0); 
            }
            Self {
                engine: AutoDiff::new(constraint_formula),
            }
        }
    }

    impl TimeVaryingConstraint<f64, 1, 1, 1> for MovingBoundaryConstraint {
        fn constraint_engine(&self) -> &impl Diff<f64, 1, 1> {
            &self.engine
        }
    }

    // -------------------------------------------------------------------------
    // 2. Running Test Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_complex_rotation_manifold() {
        // Initialize at 0 radians (1.0, 0.0)
        let rot = ComplexRotation::identity();
        
        // Push it forward by pi/2 radians (approx 1.57079632679)
        let delta: f64 = core::f64::consts::FRAC_PI_2;
        let next_rot = rot.retract(&delta);

        // Verify state maps to the unit sphere sheet correctly
        assert!((next_rot.z[0] - 0.0).abs() < 1e-9);
        assert!((next_rot.z[1] - 1.0).abs() < 1e-9);

        // Lift it back down to tangent space
        let lifted_delta = rot.local_lift(&next_rot);
        assert!((lifted_delta - core::f64::consts::FRAC_PI_2).abs() < 1e-9);

        // Verify pushforward matrix-free action: J * v
        // For identity, J = [0, 1]^T. If v = 2.0, J*v = [0, 2.0]^T
        let v: f64 = 2.0;
        let ambient_vel = rot.apply_pushforward(&v);
        assert_eq!(ambient_vel, Vector2::new(0.0, 2.0));
    }

    #[test]
    fn test_stable_product_space() {
        // Blueprint: A1=2, T1=1, A2=1, T2=1, A_TOTAL=3, T_TOTAL=2
        let state = ProductSpace::<ComplexRotation, RealLine, 2, 1, 1, 1, 3, 2> {
            m1: ComplexRotation::identity(),
            m2: RealLine { x: Vector1::new(10.0) },
        };

        // Shift position and rotate simultaneously
        let delta = (core::f64::consts::FRAC_PI_2, 5.0);
        let next_state  = state.retract(&delta);

        assert!((next_state.m1.z[0] - 0.0).abs() < 1e-9);
        assert_eq!(next_state.m2.x[0], 15.0);

        // Verify the decoupled Matrix-Free application of the global Jacobian
        let tangent_vel = (2.0, -3.0);
        let ambient_vel = state.apply_pushforward(&tangent_vel);
        
        // Complex Rotation slice: [-0.0 * 2.0, 1.0 * 2.0] = [0.0, 2.0]
        // Real Line slice: [-3.0]
        assert_eq!(ambient_vel, Vector3::new(0.0, 2.0, -3.0));
    }

    #[test]
    fn test_autodiff_constraint_bridge() {
        let constraint = MovingBoundaryConstraint::new();
        let line_state = RealLine { x: Vector1::new(7.0) };

        // Evaluate residual: 7.0 - 5.0 = 2.0
        let residual = constraint.evaluate(&line_state.x);
        assert_eq!(residual, Vector1::new(2.0));

        // Evaluate AutoDiff Jacobian: d/dx (x - 5) = 1.0
        let h_ambient = constraint.ambient_jacobian(&line_state.x);

        assert_eq!(h_ambient, SMatrix::<f64, 1, 1>::new(1.0));
    }
}
