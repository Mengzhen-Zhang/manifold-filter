use super::Manifold;
use nalgebra::ComplexField;
use nalgebra::RealField;
use nalgebra::{SMatrix, SVector};

#[derive(Clone, Copy, Debug)]
pub struct ComplexRotation<S: RealField> {
    // Ambient storage: x and y components of a unit complex number
    pub z: SVector<S, 2>,
}

impl<S: RealField + Copy> ComplexRotation<S> {
    pub fn identity() -> Self {
        Self {
            z: SVector::<S, 2>::new(S::one(), S::zero()),
        }
    }
}

impl<S: RealField + Copy> Manifold<S, 2, 1> for ComplexRotation<S> {
    type TangentVector = S; // Local 1D angular perturbation dtheta

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        // Multiplicative exponential mapping: z_next = z * exp(i * dtheta)
        let d = *delta;
        let cos_d = d.cos();
        let sin_d = d.sin();
        let x_next = self.z[0] * cos_d - self.z[1] * sin_d;
        let y_next = self.z[1] * cos_d + self.z[0] * sin_d;
        Self {
            z: SVector::<S, 2>::new(x_next, y_next),
        }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        // dtheta = atan2(z2 x z1^*)
        let x_diff = other.z[0] * self.z[0] + other.z[1] * self.z[1];
        let y_diff = other.z[1] * self.z[0] - other.z[0] * self.z[1];
        y_diff.atan2(x_diff)
    }

    fn pushforward_jacobian(&self) -> SMatrix<S, 2, 1> {
        // J_oplus = [-y, x]^T
        SMatrix::<S, 2, 1>::new(-self.z[1], self.z[0])
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, 2> {
        // Matrix-free: J_oplus * dtheta = [-y * dtheta, x * dtheta]^T
        let t = *tangent;
        SVector::<S, 2>::new(-self.z[1] * t, self.z[0] * t)
    }

    fn vector_to_tangent(vec: &SVector<S, 1>) -> Self::TangentVector {
        vec[0]
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, 1> {
        SVector::<S, 1>::new(*tangent)
    }

    fn from_ambient(vec: &SVector<S, 2>) -> Self {
        Self { z: *vec }
    }

    fn to_ambient(&self) -> SVector<S, 2> {
        self.z
    }

    fn parallel_transport(&self, _delta: &Self::TangentVector) -> SMatrix<S, 1, 1> {
        // SO(2) is abelian (ad ≡ 0), so the transport is exactly the identity.
        SMatrix::<S, 1, 1>::identity()
    }
}
