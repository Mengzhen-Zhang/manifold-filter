use nalgebra::{SMatrix, SVector, RealField, UnitQuaternion, Quaternion};
use crate::manifold::{Manifold, InvariantLieGroup};

#[derive(Clone, Debug)]
pub struct SO3 {
    /// Ambient Space
    pub uq: UnitQuaternion<f64>,
}

impl SO3 {
    #[inline(always)]
    pub fn identity() -> Self {
	Self {
	    uq: UnitQuaternion::identity(),
	}
    }

    #[inline(always)]
    pub fn from_parts(w: f64, x: f64, y: f64, z: f64) -> Self {
	let q = Quaternion::new(w, x, y, z);
	Self {
	    uq: UnitQuaternion::from_quaternion(q),
	}
    }
}

impl Manifold<f64, 4, 3> for SO3 {
    type TangentVector = SVector<f64, 3>;

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
    fn pushforward_jacobian(&self) -> SMatrix<f64, 4, 3> {
	let w = self.uq.w;
	let x = self.uq.i;
	let y = self.uq.j;
	let z = self.uq.k;

	let data = [
	    -x, -y, -z,
	    w, -z, y,
	    z, w, -x,
	    -y, x, w,
	];

	SMatrix::<f64, 4, 3>::from_row_slice(&data) * 0.5
    }

    #[inline]
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<f64, 4> {
	let w = self.uq.w;
	let x = self.uq.i;
	let y = self.uq.j;
	let z = self.uq.k;

	let tx = tangent.x;
	let ty = tangent.y;
	let tz = tangent.z;

	SVector::<f64, 4>::new(
	    -x * tx - y * ty - z * tz,
             w * tx - z * ty + y * tz,
             z * tx + w * ty - x * tz,
            -y * tx + x * ty + w * tz,
	) * 0.5
    }

    #[inline(always)]
    fn vector_to_tangent(vec: &SVector<f64, 3>) -> Self::TangentVector {
	*vec
    }

    #[inline(always)]
    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<f64, 3> {
	*tangent
    }

    fn from_ambient(vec: &SVector<f64, 4>) -> Self {
	Self::from_parts(vec.w, vec.x, vec.y, vec.z)
    }

    fn to_ambient(&self) -> SVector<f64, 4> {
	SVector::<f64, 4>::new(self.uq.w, self.uq.i, self.uq.j, self.uq.k)
    }
}

impl InvariantLieGroup<f64, 4, 3> for SO3 {
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
	    uq: self.uq * other.uq
	}
    }

    #[inline(always)]
    fn inverse(&self) -> Self {
	Self {
	    uq: self.uq.inverse(),
	}
    }

    #[inline(always)]
    fn adjoint(&self) -> SMatrix<f64, 3, 3> {
	self.uq.to_rotation_matrix().into_inner()
    }

    #[inline(always)]
    fn small_adjoint(omega: &Self::TangentVector) -> SMatrix<f64, 3, 3> {
	let data = [
            0.0, -omega[2],  omega[1],
            omega[2],       0.0, -omega[0],
            -omega[1],  omega[0],       0.0,
        ];
	SMatrix::<f64, 3, 3>::from_row_slice(&data)
    }
}
