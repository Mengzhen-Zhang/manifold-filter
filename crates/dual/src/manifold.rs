use core::{mem::zeroed, ops::Mul};

use crate::{seeded, Dual};
use nalgebra::{Matrix3, Matrix4, RealField, Rotation3, SMatrix, SVector, Vector3};

pub trait Manifold<U: RealField + Copy, const DIM: usize> {
    type Tangent<T>;

    fn box_plus(&self, delta: &Self::Tangent<U>) -> Self;

    fn box_minus(&self, other: &Self) -> Self::Tangent<U>;

    fn tangent_to_vector<T: RealField + Copy>(t: &Self::Tangent<T>) -> SVector<T, DIM>;

    fn vector_to_tangent<T: RealField + Copy>(v: &SVector<T, DIM>) -> Self::Tangent<T>;

    fn vector_transport<T: RealField + Copy>(
        &self,
        v: &Self::Tangent<T>,
        delta: &Self::Tangent<T>,
    ) -> Self::Tangent<T>;

    fn transport_matrix(
        &self,
        v: &Self::Tangent<U>,
        delta: &Self::Tangent<U>,
    ) -> SMatrix<U, DIM, DIM> {
        let mut matrix = SMatrix::<U, DIM, DIM>::zeros();

        let delta_vec: SVector<U, DIM> = Self::tangent_to_vector(delta);
        let v_vec = Self::tangent_to_vector(v);

        for j in 0..DIM {
            // --- seed basis vector e_j ---
            let mut delta_dual = SVector::<Dual<U, DIM>, DIM>::zeros();
            let mut v_dual = SVector::<Dual<U, DIM>, DIM>::zeros();

            for i in 0..DIM {
                delta_dual[i].re = delta_vec[i];
                v_dual[i].re = v_vec[i];
                delta_dual[i].eps[j] = U::one();
            }

            let v_tangent_dual = Self::vector_to_tangent(&v_dual);
            let delta_tangent_dual = Self::vector_to_tangent(&delta_dual);

            let transported = self.vector_transport(&v_tangent_dual, &delta_tangent_dual);

            let transported_vec: SVector<Dual<U, DIM>, DIM> = Self::tangent_to_vector(&transported);

            for i in 0..DIM {
                matrix[(i, j)] = transported_vec[i].eps[j];
            }
        }

        matrix
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TangentSO3<T> {
    pub vec: SVector<T, 3>,
}

impl<T> AsRef<SVector<T, 3>> for TangentSO3<T> {
    fn as_ref(&self) -> &SVector<T, 3> {
        &self.vec
    }
}

impl<T> From<SVector<T, 3>> for TangentSO3<T> {
    fn from(value: SVector<T, 3>) -> Self {
        Self { vec: value }
    }
}

pub trait ProjectiveManifold {
    fn normalize_in_place(&mut self);
}

#[derive(Debug, Clone, Copy)]
pub struct SO3<T: RealField + Copy = f64> {
    pub rot: SMatrix<T, 3, 3>,
}

impl<T: RealField + Copy> From<SMatrix<T, 3, 3>> for SO3<T> {
    fn from(value: SMatrix<T, 3, 3>) -> Self {
        Self { rot: value }
    }
}

impl<T: RealField + Copy> ProjectiveManifold for SO3<T> {
    fn normalize_in_place(&mut self) {
        let mut c1 = self.rot.column(0).into_owned();
        c1 = c1.normalize();
        let c2_raw = self.rot.column(1).into_owned();
        let mut c2 = c2_raw - c1 * c1.dot(&c2_raw);
        c2 = c2.normalize();

        let c3 = c1.cross(&c2);

        self.rot.column_mut(0).copy_from(&c1);
        self.rot.column_mut(1).copy_from(&c2);
        self.rot.column_mut(2).copy_from(&c3);
    }
}

impl<T: RealField + Copy> SO3<T> {
    fn skew(w: &TangentSO3<T>) -> SMatrix<T, 3, 3> {
        let w = w.as_ref();
        let mut out = SMatrix::zeros();
        out[(0, 1)] = -w.z;
        out[(1, 0)] = w.z;
        out[(0, 2)] = w.y;
        out[(2, 0)] = -w.y;
        out[(1, 2)] = -w.x;
        out[(2, 1)] = w.x;
        out
    }

    fn rodrigues(delta: &TangentSO3<T>) -> SMatrix<T, 3, 3> {
        let w = delta.as_ref();
        let theta = w.norm();
        let (sin_t_over_t, one_minus_cos_t_over_t2) = if theta < T::from_f64(1e-6).unwrap() {
            (
                T::one() - (theta * theta) / T::from_f64(6.0).unwrap(),
                T::from_f64(0.5).unwrap() - (theta * theta) / T::from_f64(24.0).unwrap(),
            )
        } else {
            (
                theta.sin() / theta,
                (T::one() - theta.cos()) / (theta * theta),
            )
        };
        let skew = Self::skew(delta);
        let identity = SMatrix::identity();
        let out = identity + skew * sin_t_over_t + (skew * skew) * one_minus_cos_t_over_t2;
        out
    }

    pub fn log_map(&self) -> TangentSO3<T> {
        let m = self.rot;

        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        let half = T::from_f64(0.5).unwrap();
        let eps = T::from_f64(1e-8).unwrap();
        let pi = T::pi();

        // trace(R) = 1 + 2 cos(theta)
        let trace = m.m11 + m.m22 + m.m33;
        let mut cos_theta = (trace - one) * half;

        // Clamp to [-1, 1]
        if cos_theta > one {
            cos_theta = one;
        } else if cos_theta < -one {
            cos_theta = -one;
        }

        let theta = cos_theta.acos();

        // (R - Rᵀ)∨
        let vee = SVector::<T, 3>::new(m.m32 - m.m23, m.m13 - m.m31, m.m21 - m.m12);

        let w = if theta.abs() < eps {
            // First-order approximation
            vee * half
        } else if (pi - theta).abs() < eps {
            // Near π: recover the rotation axis from the diagonal.
            let xx = (m.m11 + one) * half;
            let yy = (m.m22 + one) * half;
            let zz = (m.m33 + one) * half;

            let mut axis = SVector::<T, 3>::zeros();

            if xx >= yy && xx >= zz {
                axis.x = xx.sqrt();
                axis.y = m.m12 / (two * axis.x);
                axis.z = m.m13 / (two * axis.x);
            } else if yy >= zz {
                axis.y = yy.sqrt();
                axis.x = m.m12 / (two * axis.y);
                axis.z = m.m23 / (two * axis.y);
            } else {
                axis.z = zz.sqrt();
                axis.x = m.m13 / (two * axis.z);
                axis.y = m.m23 / (two * axis.z);
            }

            axis *= one / axis.norm();

            axis * theta
        } else {
            // General case
            let scale = theta / (two * theta.sin());
            vee * scale
        };

        TangentSO3::from(w)
    }

    pub fn exp_map(delta: &TangentSO3<T>) -> Self {
        Self {
            rot: Self::rodrigues(delta),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            rot: self.rot.transpose(),
        }
    }

    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            rot: self.rot * rhs.rot,
        }
    }
}

impl<T: RealField + Copy> Mul<&SO3<T>> for SO3<T> {
    type Output = Self;
    fn mul(self, rhs: &SO3<T>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<T: RealField + Copy> Mul<&SO3<T>> for &SO3<T> {
    type Output = SO3<T>;
    fn mul(self, rhs: &SO3<T>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<U: RealField + Copy> Manifold<U, 3> for SO3<U> {
    type Tangent<T> = TangentSO3<T>;

    fn box_plus(&self, delta: &Self::Tangent<U>) -> Self {
        self * &SO3::exp_map(delta)
    }

    fn box_minus(&self, rhs: &Self) -> Self::Tangent<U> {
        (rhs.inverse() * self).log_map()
    }

    fn tangent_to_vector<T: Copy>(t: &Self::Tangent<T>) -> SVector<T, 3> {
        t.vec
    }

    fn vector_to_tangent<T: Copy>(v: &SVector<T, 3>) -> Self::Tangent<T> {
        Self::Tangent { vec: *v }
    }

    fn vector_transport<T: RealField + Copy>(
        &self,
        v: &Self::Tangent<T>,
        delta: &Self::Tangent<T>,
    ) -> Self::Tangent<T> {
        let v = SO3::exp_map(delta).inverse().rot * v.as_ref();
        TangentSO3::from(v)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TangentSE3<T> {
    pub translation: SVector<T, 3>,
    pub rotation: SVector<T, 3>,
}

#[derive(Debug, Clone, Copy)]
pub struct SE3<T: RealField + Copy = f64> {
    pub rotation: SO3<T>,
    pub translation: Vector3<T>,
}

impl<T: RealField + Copy> SE3<T> {
    pub fn compose(&self, rhs: &Self) -> Self {
        let rotation = self.rotation * &rhs.rotation;
        let translation = self.translation + self.rotation.rot * rhs.translation;
        Self {
            rotation,
            translation,
        }
    }

    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = -(rotation.rot * self.translation);
        Self {
            rotation,
            translation,
        }
    }

    fn v_matrix(w: &SVector<T, 3>) -> SMatrix<T, 3, 3> {
        let theta = w.norm();

        let eps = T::from_f64(1e-6).unwrap();
        let I = SMatrix::<T, 3, 3>::identity();

        let wx = SO3::<T>::skew(&TangentSO3::from(*w));

        if theta < eps {
            I + wx * T::from_f64(0.5).unwrap()
        } else {
            let theta2 = theta * theta;
            let theta3 = theta2 * theta;

            let sin = theta.sin();
            let cos = theta.cos();

            let a = (T::one() - cos) / theta2;
            let b = (theta - sin) / theta3;

            I + wx * a + (wx * wx) * b
        }
    }

    pub fn exp_map(delta: &TangentSE3<T>) -> SE3<T> {
        let w = delta.rotation;
        let v = delta.translation;
        let rotation = SO3::exp_map(&TangentSO3::from(w));
        let translation = Self::v_matrix(&w) * v;

        SE3 {
            rotation,
            translation,
        }
    }

    fn v_inv(w: &SVector<T, 3>) -> SMatrix<T, 3, 3> {
        let theta = w.norm();

        let eps = T::from_f64(1e-6).expect("cannot convert from f64");
        let half = T::from_f64(0.5).expect("cannot convert from f64");
        let I = SMatrix::<T, 3, 3>::identity();

        let wx = SO3::<T>::skew(&TangentSO3::from(*w));

        if theta < eps {
            I - wx * half
        } else {
            let theta2 = theta * theta;
            let c = (T::one()
                - (theta * theta.sin()) / (T::from_f64(2.0).unwrap() * (T::one() - theta.cos())))
                / theta2;
            I - wx * half + (wx * wx) * c
        }
    }

    pub fn log_map(&self) -> TangentSE3<T> {
        let w = self.rotation.log_map().vec;
        let v_inv = Self::v_inv(&w);
        let v = v_inv * self.translation;

        TangentSE3 {
            translation: v,
            rotation: w,
        }
    }

    pub fn adjoint(t: &SE3<T>) -> SMatrix<T, 6, 6> {
        let r = t.rotation.rot;
        let p = t.translation;
        let px = SO3::skew(&TangentSO3::from(p));
        let mut ad = SMatrix::<T, 6, 6>::zeros();

        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&(px * r));
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad
    }
}

impl<T: RealField + Copy> Mul<&SE3<T>> for SE3<T> {
    type Output = Self;
    fn mul(self, rhs: &SE3<T>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<T: RealField + Copy> Mul<&SE3<T>> for &SE3<T> {
    type Output = SE3<T>;
    fn mul(self, rhs: &SE3<T>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<U: RealField + Copy> Manifold<U, 6> for SE3<U> {
    type Tangent<T> = TangentSE3<T>;

    fn tangent_to_vector<T: RealField + Copy>(t: &Self::Tangent<T>) -> SVector<T, 6> {
        let mut v = SVector::<T, 6>::zeros();
        v.fixed_rows_mut::<3>(0).copy_from(&t.translation);
        v.fixed_rows_mut::<3>(3).copy_from(&t.rotation);
        v
    }

    fn vector_to_tangent<T: RealField + Copy>(t: &SVector<T, 6>) -> Self::Tangent<T> {
        Self::Tangent {
            rotation: t.fixed_rows::<3>(3).into_owned(),
            translation: t.fixed_rows::<3>(0).into_owned(),
        }
    }

    fn box_plus(&self, delta: &Self::Tangent<U>) -> Self {
        self * &Self::exp_map(delta)
    }

    fn box_minus(&self, other: &Self) -> Self::Tangent<U> {
        (other.inverse() * self).log_map()
    }

    fn vector_transport<T>(
        &self,
        v: &Self::Tangent<T>,
        delta: &Self::Tangent<T>,
    ) -> Self::Tangent<T>
    where
        T: RealField + Copy,
    {
        let t_delta = SE3::exp_map(delta);
        let ad = SE3::adjoint(&t_delta.inverse());
        let vin = SE3::<T>::tangent_to_vector(v);
        let vout = ad * vin;

        SE3::<T>::vector_to_tangent(&vout)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TangentRN<T, const DIM: usize> {
    pub vec: SVector<T, DIM>,
}

impl<U, const N: usize> Manifold<U, N> for SVector<U, N>
where
    U: RealField + Copy,
{
    // For Euclidean space, the tangent type is just itself!
    type Tangent<T> = TangentRN<T, N>;

    fn box_plus(&self, delta: &Self::Tangent<U>) -> Self {
        self + delta.vec
    }

    fn box_minus(&self, other: &Self) -> Self::Tangent<U> {
        TangentRN { vec: self - other }
    }

    fn tangent_to_vector<T: RealField + Copy>(t: &Self::Tangent<T>) -> SVector<T, N> {
        t.vec
    }

    fn vector_to_tangent<T: RealField + Copy>(v: &SVector<T, N>) -> Self::Tangent<T> {
        TangentRN { vec: *v }
    }

    fn vector_transport<T: RealField + Copy>(
        &self,
        v: &Self::Tangent<T>,
        _delta: &Self::Tangent<T>,
    ) -> Self::Tangent<T> {
        // Flat space has no curvature, so parallel transport is simply the identity map!
        *v
    }
}

// #[derive(dual_derive::Manifold)]
// pub struct IntertialState {
//     #[manifold(dim = 6)]
//     pub pose: SE3<f64>,

//     #[manifold(dim = 3)]
//     pub velocity: nalgebra::Vector3<f64>,

//     #[manifold(dim = 3)]
//     pub gyro_bias: nalgebra::Vector3<f64>,
// }
