use crate::manifold::{InvariantLieGroup, Manifold};
use nalgebra::{Quaternion, RealField, SMatrix, SVector, UnitQuaternion};

#[derive(Clone, Debug)]
pub struct SO3<S: RealField> {
    /// Ambient Space
    pub uq: UnitQuaternion<S>,
}

impl<S: RealField + Copy> SO3<S> {
    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            uq: UnitQuaternion::identity(),
        }
    }

    #[inline(always)]
    pub fn from_parts(w: S, x: S, y: S, z: S) -> Self {
        let q = Quaternion::new(w, x, y, z);
        Self {
            uq: UnitQuaternion::from_quaternion(q),
        }
    }
}

impl<S: RealField + Copy> Manifold<S, 4, 3> for SO3<S> {
    type TangentVector = SVector<S, 3>;

    #[inline]
    fn retract(&self, delta: &Self::TangentVector) -> Self {
        let dq = UnitQuaternion::from_scaled_axis(*delta);
        Self { uq: self.uq * dq }
    }

    #[inline]
    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        let relative = self.uq.inverse() * other.uq;
        relative.scaled_axis()
    }

    #[inline]
    fn pushforward_jacobian(&self) -> SMatrix<S, 4, 3> {
        let w = self.uq.w;
        let x = self.uq.i;
        let y = self.uq.j;
        let z = self.uq.k;
        let half = S::one() / (S::one() + S::one());

        let data = [-x, -y, -z, w, -z, y, z, w, -x, -y, x, w];

        SMatrix::<S, 4, 3>::from_row_slice(&data) * half
    }

    #[inline]
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, 4> {
        let w = self.uq.w;
        let x = self.uq.i;
        let y = self.uq.j;
        let z = self.uq.k;

        let tx = tangent.x;
        let ty = tangent.y;
        let tz = tangent.z;
        let half = S::one() / (S::one() + S::one());

        SVector::<S, 4>::new(
            -x * tx - y * ty - z * tz,
            w * tx - z * ty + y * tz,
            z * tx + w * ty - x * tz,
            -y * tx + x * ty + w * tz,
        ) * half
    }

    #[inline(always)]
    fn vector_to_tangent(vec: &SVector<S, 3>) -> Self::TangentVector {
        *vec
    }

    #[inline(always)]
    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, 3> {
        *tangent
    }

    fn from_ambient(vec: &SVector<S, 4>) -> Self {
        // Ambient layout must match `to_ambient`: [w, i, j, k] at indices 0..3.
        // (Reading via `.x/.y/.z/.w` accessors reorders to [idx0, idx1, idx2, idx3]
        // = [w, i, j, k] but with `.w` = idx3, scrambling the scalar part.)
        Self::from_parts(vec[0], vec[1], vec[2], vec[3])
    }

    fn to_ambient(&self) -> SVector<S, 4> {
        SVector::<S, 4>::new(self.uq.w, self.uq.i, self.uq.j, self.uq.k)
    }

    fn parallel_transport(&self, delta: &Self::TangentVector) -> SMatrix<S, 3, 3> {
        // Exact Levi-Civita parallel transport (bi-invariant metric): R(−δ/2),
        // given by the exponential map. Agrees with the right Jacobian to first
        // order; the predict's ±δ composition then yields the exact full adjoint
        // R(−δ).
        let neg_half = -(S::one() / (S::one() + S::one()));
        UnitQuaternion::from_scaled_axis(*delta * neg_half)
            .to_rotation_matrix()
            .into_inner()
    }
}

impl<S: RealField + Copy> InvariantLieGroup<S, 4, 3> for SO3<S> {
    #[inline(always)]
    fn exp(omega: &Self::TangentVector) -> Self {
        Self {
            uq: UnitQuaternion::from_scaled_axis(*omega),
        }
    }

    #[inline(always)]
    fn log(&self) -> Self::TangentVector {
        self.uq.scaled_axis()
    }

    #[inline(always)]
    fn compose(&self, other: &Self) -> Self {
        Self {
            uq: self.uq * other.uq,
        }
    }

    #[inline(always)]
    fn inverse(&self) -> Self {
        Self {
            uq: self.uq.inverse(),
        }
    }

    #[inline(always)]
    fn adjoint(&self) -> SMatrix<S, 3, 3> {
        self.uq.to_rotation_matrix().into_inner()
    }

    #[inline(always)]
    fn small_adjoint(omega: &Self::TangentVector) -> SMatrix<S, 3, 3> {
        let data = [
            S::zero(),
            -omega[2],
            omega[1],
            omega[2],
            S::zero(),
            -omega[0],
            -omega[1],
            omega[0],
            S::zero(),
        ];
        SMatrix::<S, 3, 3>::from_row_slice(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `from_ambient` must invert `to_ambient`. Use a quaternion with four
    // distinct components — a symmetric one (e.g. 0.5,0.5,0.5,0.5) would hide a
    // scalar/vector reordering bug.
    #[test]
    fn from_ambient_inverts_to_ambient() {
        let q = SO3::from_parts(1.0, 2.0, 3.0, 4.0); // normalized internally; w,i,j,k distinct
        let a = q.to_ambient();
        let round = SO3::<f64>::from_ambient(&a);

        // Same rotation: log(q⁻¹ · round) ≈ 0.
        assert!(q.local_lift(&round).norm() < 1e-12);
        // And the ambient vector is reproduced componentwise.
        assert!((a - round.to_ambient()).norm() < 1e-12);
    }

    // `parallel_transport(δ)` is the exact half-angle transport R(−δ/2), and the
    // ±δ composition T(δ)·T(−δ)⁻¹ recovers the exact full adjoint R(−δ) used by
    // the predict step.
    #[test]
    fn parallel_transport_composes_to_full_adjoint() {
        let q = SO3::<f64>::identity();
        let d = SVector::<f64, 3>::new(0.1, -0.2, 0.3);

        let t = q.parallel_transport(&d);
        let r_neg_half = UnitQuaternion::from_scaled_axis(-d * 0.5)
            .to_rotation_matrix()
            .into_inner();
        assert!(
            (t - r_neg_half).norm() < 1e-12,
            "transport should be R(−δ/2)"
        );

        let t_back = q.parallel_transport(&(-d));
        let adjoint = t * t_back.try_inverse().unwrap();
        let r_neg = UnitQuaternion::from_scaled_axis(-d)
            .to_rotation_matrix()
            .into_inner();
        assert!(
            (adjoint - r_neg).norm() < 1e-12,
            "T(δ)·T(−δ)⁻¹ should be R(−δ)"
        );
    }
}
