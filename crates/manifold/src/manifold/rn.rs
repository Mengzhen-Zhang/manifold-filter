use super::Manifold;
use nalgebra::{RealField, SMatrix, SVector};

#[derive(Clone, Debug)]
pub struct RealLine {
    pub x: SVector<f64, 1>,
}

impl Manifold<f64, 1, 1> for RealLine {
    type TangentVector = f64;

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        Self {
            x: SVector::<f64, 1>::new(self.x[0] + delta),
        }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        other.x[0] - self.x[0]
    }

    fn pushforward_jacobian(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::identity()
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }

    fn vector_to_tangent(vec: &SVector<f64, 1>) -> Self::TangentVector {
        vec[0]
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(*tangent)
    }

    fn from_ambient(vec: &SVector<f64, 1>) -> Self {
        Self { x: *vec }
    }

    fn to_ambient(&self) -> SVector<f64, 1> {
        self.x
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Rn<S: RealField, const N: usize> {
    pub x: SVector<S, N>,
}

impl<S: RealField + Copy, const N: usize> Manifold<S, N, N> for Rn<S, N> {
    type TangentVector = SVector<S, N>;

    fn retract(&self, delta: &Self::TangentVector) -> Self {
        Self { x: self.x + delta }
    }

    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        other.x - self.x
    }

    fn pushforward_jacobian(&self) -> SMatrix<S, N, N> {
        SMatrix::identity()
    }

    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, N> {
        *tangent
    }

    fn vector_to_tangent(vec: &SVector<S, N>) -> Self::TangentVector {
        *vec
    }

    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, N> {
        *tangent
    }

    fn from_ambient(vec: &SVector<S, N>) -> Self {
        Self { x: *vec }
    }

    fn to_ambient(&self) -> SVector<S, N> {
        self.x
    }
}
