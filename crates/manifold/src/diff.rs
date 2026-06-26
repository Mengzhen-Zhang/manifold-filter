use crate::dual::{self, Dual};
use nalgebra::{RealField, SMatrix, SVector};

pub trait Diff<T, const In: usize, const Out: usize> {
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out>;

    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In>;

    // jacobian vector product
    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out>;
    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>);
}

/// A function written generically over the scalar field. Instantiating it at
/// different scalars yields different modes from a single definition:
/// `S = T` is a plain value evaluation (no AD), `S = Dual<T, In>` produces the
/// full batched Jacobian in one pass, and `S = Dual<T, 1>` gives a single
/// directional derivative.
pub trait DiffFn<const In: usize, const Out: usize> {
    fn eval<S: RealField + Copy>(&self, x: &SVector<S, In>, y: &mut SVector<S, Out>);
}

pub struct AutoDiff<G> {
    pub f: G,
}

impl<G> AutoDiff<G> {
    pub fn new(f: G) -> Self {
	Self { f }
    }
}

impl<T, G, const In: usize, const Out: usize> Diff<T, In, Out> for AutoDiff<G>
where
    T: RealField + Copy,
    G: DiffFn<In, Out>,
{
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out> {
	// Runs on plain reals: no dual numbers, no AD overhead.
	let mut y = SVector::<T, Out>::zeros();
	self.f.eval::<T>(x, &mut y);
	y
    }

    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In> {
	// Seed every input with its unit tangent and evaluate once: a single
	// width-In pass yields the whole Out x In Jacobian.
	let x_dual = dual::seeded(x);
	let mut y_dual = SVector::<Dual<T, In>, Out>::zeros();
	self.f.eval::<Dual<T, In>>(&x_dual, &mut y_dual);
	dual::jacobian(&y_dual)
    }

    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out> {
	// Single direction: a width-1 dual seeded with the tangent `v` gives
	// J*v directly in one cheap pass (cost independent of In).
	let x_dual =
	    SVector::<Dual<T, 1>, In>::from_fn(|i, _| Dual::new(x[i], SVector::<T, 1>::new(v[i])));
	let mut y_dual = SVector::<Dual<T, 1>, Out>::zeros();
	self.f.eval::<Dual<T, 1>>(&x_dual, &mut y_dual);
	y_dual.map(|d| d.eps[0])
    }

    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>) {
	// Value and directional derivative from the same width-1 pass.
	let x_dual =
	    SVector::<Dual<T, 1>, In>::from_fn(|i, _| Dual::new(x[i], SVector::<T, 1>::new(v[i])));
	let mut y_dual = SVector::<Dual<T, 1>, Out>::zeros();
	self.f.eval::<Dual<T, 1>>(&x_dual, &mut y_dual);
	(y_dual.map(|d| d.re), y_dual.map(|d| d.eps[0]))
    }
}


pub struct NormalDiff<T: RealField, const In: usize, const Out: usize> {
    pub f: fn(&SVector<T, In>) -> SVector<T, Out>,
    pub jacobian: fn(&SVector<T, In>) -> SMatrix<T, Out, In>,
}

impl<T: RealField, const In: usize, const Out: usize> NormalDiff<T, In, Out> {
    pub fn new(
	f: fn(&SVector<T, In>) -> SVector<T, Out>,
	jacobian: fn(&SVector<T, In>) -> SMatrix<T, Out, In>,
    ) -> Self {
	Self { f, jacobian }
    }
}


impl<T: RealField, const In: usize, const Out: usize> Diff<T, In, Out> for NormalDiff<T, In, Out> {
    fn eval(&self, x: &SVector<T, In>) -> SVector<T, Out> {
	(self.f)(x)
    }

    fn jacobian(&self, x: &SVector<T, In>) -> SMatrix<T, Out, In> {
	(self.jacobian)(x)
    }

    fn jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> SVector<T, Out> {
	self.jacobian(x) * v
    }

    fn eval_jvp(&self, x: &SVector<T, In>, v: &SVector<T, In>) -> (SVector<T, Out>, SVector<T, Out>) {
	let y = self.eval(x);
	let jvp = self.jvp(x, v);
	(y, jvp)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector2, Matrix2};

    // Builds a generic scalar constant from an f64 literal.
    #[inline]
    fn c<S: RealField>(x: f64) -> S {
        nalgebra::convert(x)
    }

    // -------------------------------------------------------------------------
    // Test Case 1: Simple Linear/Affine Transformation Matrix
    // f(x, y) = [2x + 3y, 4x - y]
    // Expected Jacobian:
    // [2.0,  3.0]
    // [4.0, -1.0]
    // -------------------------------------------------------------------------
    struct LinearFn;
    impl DiffFn<2, 2> for LinearFn {
        fn eval<S: RealField + Copy>(&self, x: &SVector<S, 2>, y: &mut SVector<S, 2>) {
            y[0] = x[0] * c(2.0) + x[1] * c(3.0);
            y[1] = x[0] * c(4.0) - x[1];
        }
    }

    fn linear_test_func_normal(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        Vector2::new(2.0 * x[0] + 3.0 * x[1], 4.0 * x[0] - x[1])
    }

    fn linear_test_jacobian_normal(_x: &SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        Matrix2::new(2.0,  3.0,
                     4.0, -1.0)
    }

    #[test]
    fn test_linear_autodiff() {
        let ad = AutoDiff::new(LinearFn);
        let x = Vector2::new(1.0, 2.0); // Evaluation point
        let v = Vector2::new(10.0, -1.0); // Direction vector

        let expected_eval = Vector2::new(8.0, 2.0);
        let expected_jac = Matrix2::new(
	    2.0,  3.0,
            4.0, -1.0);
        // J * v = [2(10) + 3(-1), 4(10) - 1(-1)] = [17, 41]
        let expected_jvp = Vector2::new(17.0, 41.0);

        assert_eq!(ad.eval(&x), expected_eval);
        assert_eq!(ad.jacobian(&x), expected_jac);
        assert_eq!(ad.jvp(&x, &v), expected_jvp);

        let (val, jvp) = ad.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    #[test]
    fn test_linear_normal() {
        let nd = NormalDiff::new(linear_test_func_normal, linear_test_jacobian_normal);
        let x = Vector2::new(1.0, 2.0);
        let v = Vector2::new(10.0, -1.0);

        let expected_eval = Vector2::new(8.0, 2.0);
        let expected_jac = Matrix2::new(2.0,  3.0,
					4.0, -1.0);
        let expected_jvp = Vector2::new(17.0, 41.0);

        assert_eq!(nd.eval(&x), expected_eval);
        assert_eq!(nd.jacobian(&x), expected_jac);
        assert_eq!(nd.jvp(&x, &v), expected_jvp);

        let (val, jvp) = nd.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    // -------------------------------------------------------------------------
    // Test Case 2: Non-Linear Function (Squaring and Products)
    // f(x, y) = [x^2, x * y]
    // Expected Jacobian:
    // [2x,   0]
    // [y,    x]
    // -------------------------------------------------------------------------
    struct NonlinearFn;
    impl DiffFn<2, 2> for NonlinearFn {
        fn eval<S: RealField + Copy>(&self, x: &SVector<S, 2>, y: &mut SVector<S, 2>) {
            y[0] = x[0] * x[0];       // x^2
            y[1] = x[0] * x[1];       // x * y
        }
    }

    #[test]
    fn test_nonlinear_autodiff() {
        let ad = AutoDiff::new(NonlinearFn);
        let x = Vector2::new(3.0, 4.0);  // Evaluate at x=3, y=4
        let v = Vector2::new(2.0, 5.0);  // Direction vector

        // f(3, 4) = [9, 12)
        let expected_eval = Vector2::new(9.0, 12.0);

        // J = [[6, 0), [4, 3))
        let expected_jac = Matrix2::new(6.0, 0.0,
                                   4.0, 3.0);

        // J * v = [[6, 0), [4, 3)) * [2, 5) = [12, 4 * 2 + 3 * 5) = [12, 23)
        let expected_jvp = Vector2::new(12.0, 23.0);

        assert_eq!(ad.eval(&x), expected_eval);
        assert_eq!(ad.jacobian(&x), expected_jac);
        assert_eq!(ad.jvp(&x, &v), expected_jvp);

        let (val, jvp) = ad.eval_jvp(&x, &v);
        assert_eq!(val, expected_eval);
        assert_eq!(jvp, expected_jvp);
    }

    // -------------------------------------------------------------------------
    // Test Case 3: Polymorphism Verification via Trait Object
    // -------------------------------------------------------------------------
    #[test]
    fn test_polymorphism() {
	let ad = AutoDiff::new(LinearFn);
	let nd = NormalDiff::new(
            linear_test_func_normal,
            linear_test_jacobian_normal,
	);

	let diff_engines: [&dyn Diff<f64, 2, 2>; 2] = [&ad, &nd];

	let x = Vector2::new(2.0, 1.0);
	let v = Vector2::new(1.0, 1.0);

	for engine in diff_engines {
            let res_eval = engine.eval(&x);
            let res_jac = engine.jacobian(&x);
            let res_jvp = engine.jvp(&x, &v);
            let (combined_val, combined_jvp) = engine.eval_jvp(&x, &v);

            assert_eq!(res_eval, Vector2::new(7.0, 7.0));
            assert_eq!(
		res_jac,
		Matrix2::new(
                    2.0,  3.0,
                    4.0, -1.0
		)
            );
            assert_eq!(res_jvp, Vector2::new(5.0, 3.0));
            assert_eq!(combined_val, Vector2::new(7.0, 7.0));
            assert_eq!(combined_jvp, Vector2::new(5.0, 3.0));
	}
    }
}
