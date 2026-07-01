use nalgebra::{ComplexField, RealField, SMatrix, SVector, Scalar};

use crate::diff::Diff;

pub mod product;
pub use product::*;

pub mod rn;
pub use rn::*;

pub mod so2;
pub use so2::*;

pub mod so3;
pub use so3::*;

pub mod dynamics;
pub use dynamics::*;

pub mod product_macro;

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

    fn from_ambient(vec: &SVector<S, A_DIM>) -> Self;

    fn to_ambient(&self) -> SVector<S, A_DIM>;

    /// Parallel transport `T_xM → T_{x⊕δ}M` of a tangent vector along the
    /// geodesic by `delta`. Used to carry the tangent covariance between frames:
    /// directly for the measurement reset (`P ← T P Tᵀ`), and composed at `±δ`
    /// for the predict-step adjoint (`Ad_{Exp(−δ)} = T(δ)·T(−δ)⁻¹`).
    ///
    /// The default is the differentiated-retraction form `J_⊕(x⊕δ)⁺ · J_⊕(x)` —
    /// a good first-order approximation for any manifold (flat ⇒ identity). Lie
    /// groups may override with the exact transport — for SO(3), the half-angle
    /// rotation `R(−δ/2)` (given by the exponential map).
    fn parallel_transport(&self, delta: &Self::TangentVector) -> SMatrix<S, T_DIM, T_DIM>
    where
        S: RealField,
        Self: Sized,
    {
        let j_old = self.pushforward_jacobian();
        let j_new = self.retract(delta).pushforward_jacobian();
        let jnt = j_new.transpose();
        let pinv = (&jnt * &j_new)
            .try_inverse()
            .expect("parallel_transport: rank-deficient pushforward")
            * jnt;
        pinv * j_old
    }
}

pub trait InvariantLieGroup<S, const A_DIM: usize, const T_DIM: usize>:
    Manifold<S, A_DIM, T_DIM>
{
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

impl<S: RealField + Copy> InvariantLieGroup<S, 2, 1> for ComplexRotation<S> {
    fn exp(omega: &Self::TangentVector) -> Self {
        let o = *omega;
        Self {
            z: SVector::<S, 2>::new(o.cos(), o.sin()),
        }
    }

    fn log(&self) -> Self::TangentVector {
        self.z[1].atan2(self.z[0])
    }

    fn compose(&self, other: &Self) -> Self {
        let x = self.z[0] * other.z[0] - self.z[1] * other.z[1];
        let y = self.z[0] * other.z[1] + self.z[1] * other.z[0];
        Self {
            z: SVector::<S, 2>::new(x, y),
        }
    }

    fn inverse(&self) -> Self {
        Self {
            z: SVector::<S, 2>::new(self.z[0], -self.z[1]),
        }
    }

    fn adjoint(&self) -> SMatrix<S, 1, 1> {
        SMatrix::<S, 1, 1>::identity() // SO(2) is abelian, Ad_g = Identity
    }

    fn small_adjoint(_omega: &Self::TangentVector) -> SMatrix<S, 1, 1> {
        SMatrix::<S, 1, 1>::zeros() // ad_omega = 0
    }
}

#[cfg(test)]
mod integration_tests {
    use nalgebra::{SMatrix, Vector1, Vector2, Vector3};

    use super::*;

    // -------------------------------------------------------------------------
    // 2. Running Test Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_complex_rotation_manifold() {
        // Initialize at 0 radians (1.0, 0.0)
        let rot = ComplexRotation::<f64>::identity();

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
        let state = ProductSpace::<ComplexRotation<f64>, RealLine, 2, 1, 1, 1, 3, 2> {
            m1: ComplexRotation::identity(),
            m2: RealLine {
                x: Vector1::new(10.0),
            },
        };

        // Shift position and rotate simultaneously
        let delta = (core::f64::consts::FRAC_PI_2, 5.0);
        let next_state = state.retract(&delta);

        assert!((next_state.m1.z[0] - 0.0).abs() < 1e-9);
        assert_eq!(next_state.m2.x[0], 15.0);

        // Verify the decoupled Matrix-Free application of the global Jacobian
        let tangent_vel = (2.0, -3.0);
        let ambient_vel = state.apply_pushforward(&tangent_vel);

        // Complex Rotation slice: [-0.0 * 2.0, 1.0 * 2.0] = [0.0, 2.0]
        // Real Line slice: [-3.0]
        assert_eq!(ambient_vel, Vector3::new(0.0, 2.0, -3.0));
    }
}
