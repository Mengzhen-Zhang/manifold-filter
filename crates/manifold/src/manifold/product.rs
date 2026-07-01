use super::Manifold;
use nalgebra::{RealField, SMatrix, SVector, Scalar};
use num_traits::Zero;

#[derive(Clone, Debug)]
pub struct ProductSpace<
    M1,
    M2,
    const A1: usize,
    const T1: usize,
    const A2: usize,
    const T2: usize,
    const A_TOTAL: usize,
    const T_TOTAL: usize,
> {
    pub m1: M1,
    pub m2: M2,
}

impl<
        S,
        M1,
        M2,
        const A1: usize,
        const T1: usize,
        const A2: usize,
        const T2: usize,
        const A_TOTAL: usize,
        const T_TOTAL: usize,
    > Manifold<S, A_TOTAL, T_TOTAL> for ProductSpace<M1, M2, A1, T1, A2, T2, A_TOTAL, T_TOTAL>
where
    S: Scalar + Zero,
    M1: Manifold<S, A1, T1>,
    M2: Manifold<S, A2, T2>,
{
    type TangentVector = (M1::TangentVector, M2::TangentVector);

    #[inline]
    fn retract(&self, delta: &Self::TangentVector) -> Self {
        ProductSpace {
            m1: self.m1.retract(&delta.0),
            m2: self.m2.retract(&delta.1),
        }
    }

    #[inline]
    fn local_lift(&self, other: &Self) -> Self::TangentVector {
        (self.m1.local_lift(&other.m1), self.m2.local_lift(&other.m2))
    }

    fn pushforward_jacobian(&self) -> SMatrix<S, A_TOTAL, T_TOTAL> {
        let mut j_product = SMatrix::<S, A_TOTAL, T_TOTAL>::zeros();

        let j1 = self.m1.pushforward_jacobian();
        let j2 = self.m2.pushforward_jacobian();

        // Assign sub-matrices safely using nalgebra's fixed slice views on the stack.
        // These bounds are checked at runtime via assertions, which keeps the type checker happy.
        j_product.fixed_view_mut::<A1, T1>(0, 0).copy_from(&j1);
        j_product.fixed_view_mut::<A2, T2>(A1, T1).copy_from(&j2);

        j_product
    }

    #[inline]
    fn apply_pushforward(&self, tangent: &Self::TangentVector) -> SVector<S, A_TOTAL> {
        let mut out_ambient = SVector::<S, A_TOTAL>::zeros();

        // Compute the independent sub-manifold pushforwards matrix-free!
        let am1 = self.m1.apply_pushforward(&tangent.0);
        let am2 = self.m2.apply_pushforward(&tangent.1);

        // Splice the resulting small vectors directly into the stack output array
        out_ambient.fixed_rows_mut::<A1>(0).copy_from(&am1);
        out_ambient.fixed_rows_mut::<A2>(A1).copy_from(&am2);

        out_ambient
    }

    #[inline]
    fn vector_to_tangent(vec: &SVector<S, T_TOTAL>) -> Self::TangentVector {
        let v1 = vec.fixed_rows::<T1>(0).into_owned();
        let v2 = vec.fixed_rows::<T2>(T1).into_owned();
        (M1::vector_to_tangent(&v1), M2::vector_to_tangent(&v2))
    }

    #[inline]
    fn tangent_to_vector(tangent: &Self::TangentVector) -> SVector<S, T_TOTAL> {
        let mut vec = SVector::<S, T_TOTAL>::zeros();
        vec.fixed_rows_mut::<T1>(0)
            .copy_from(&M1::tangent_to_vector(&tangent.0));
        vec.fixed_rows_mut::<T2>(T1)
            .copy_from(&M2::tangent_to_vector(&tangent.1));
        vec
    }

    #[inline]
    fn from_ambient(vec: &SVector<S, A_TOTAL>) -> Self {
        let v1 = vec.fixed_rows::<A1>(0).into_owned();
        let v2 = vec.fixed_rows::<A2>(A1).into_owned();
        ProductSpace {
            m1: M1::from_ambient(&v1),
            m2: M2::from_ambient(&v2),
        }
    }

    fn to_ambient(&self) -> SVector<S, A_TOTAL> {
        let mut vec = SVector::<S, A_TOTAL>::zeros();
        vec.fixed_rows_mut::<A1>(0).copy_from(&self.m1.to_ambient());
        vec.fixed_rows_mut::<A2>(A1)
            .copy_from(&self.m2.to_ambient());
        vec
    }

    fn parallel_transport(&self, delta: &Self::TangentVector) -> SMatrix<S, T_TOTAL, T_TOTAL>
    where
        S: RealField,
    {
        // The connection on a product is the product of the components' — so the
        // transport is block-diagonal. Delegating lets component overrides (e.g.
        // SO3's exact transport) apply, and avoids a full A_TOTAL x A_TOTAL inverse.
        let mut g = SMatrix::<S, T_TOTAL, T_TOTAL>::zeros();
        g.fixed_view_mut::<T1, T1>(0, 0)
            .copy_from(&self.m1.parallel_transport(&delta.0));
        g.fixed_view_mut::<T2, T2>(T1, T1)
            .copy_from(&self.m2.parallel_transport(&delta.1));
        g
    }
}
