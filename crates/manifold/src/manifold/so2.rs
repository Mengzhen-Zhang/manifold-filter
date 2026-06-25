use nalgebra::{SMatrix, SVector};
use nalgebra::ComplexField;
use nalgebra::RealField;
use super::Manifold;

#[derive(Clone, Debug)]
pub struct ComplexRotation {
    // Ambient storage: x and y components of a unit complex number
    pub z: SVector<f64, 2>, 
}

impl ComplexRotation {
    pub fn identity() -> Self {
        Self { z: SVector::<f64, 2>::new(1.0, 0.0) }
    }
}

impl Manifold<f64, 2, 1> for ComplexRotation {
    type TangentVector = f64; // Local 1D angular perturbation dtheta

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        // Multiplicative exponential mapping: z_next = z * exp(i * dtheta)
        let cos_d = delta.cos();
        let sin_d = delta.sin();
        let x_next = self.z[0] * cos_d - self.z[1] * sin_d;
        let y_next = self.z[1] * cos_d + self.z[0] * sin_d;
        Self { z: SVector::<f64, 2>::new(x_next, y_next) }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        // dtheta = atan2(z2 x z1^*)
        let x_diff = other.z[0] * self.z[0] + other.z[1] * self.z[1];
        let y_diff = other.z[1] * self.z[0] - other.z[0] * self.z[1];
        y_diff.atan2(x_diff)
    }

    fn pushforward_jacobian(&self) -> SMatrix<f64, 2, 1> {
        // J_oplus = [-y, x]^T
        SMatrix::<f64, 2, 1>::new(-self.z[1], self.z[0])
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<f64, 2> {
        // Matrix-free: J_oplus * dtheta = [-y * dtheta, x * dtheta]^T
        SVector::<f64, 2>::new(-self.z[1] * tangent, self.z[0] * tangent)
    }

    fn vector_to_tangent(vec: &SVector<f64, 1>) -> Self::TangentVector {
        vec[0]
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }
}
