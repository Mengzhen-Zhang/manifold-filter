use nalgebra::{RealField, SMatrix, SVector, Scalar};
use num_traits::Zero;
use crate::diff::Diff;
use crate::manifold::Manifold;

/// A unified, stack-allocated trajectory dynamics engine.
/// Completely handles packing, evaluation, and Jacobian extraction for any physical system.
/// TOTAL_DIM stands for dimensions of the arguments of the dynamics function.
#[derive(Clone, Debug)]
pub struct TrajectoryDynamics<S, D, const A_DIM: usize, const T_DIM: usize, const U_DIM: usize, const W_DIM: usize, const TOTAL_DIM: usize>
where
    D: Diff<S, TOTAL_DIM, T_DIM>,
{
    pub diff_engine: D,
    // Empties out extra type tracking parameters at compile time
    _marker: core::marker::PhantomData<S>,
}

impl<S, D, const A_DIM: usize, const T_DIM: usize, const U_DIM: usize, const W_DIM: usize, const TOTAL_DIM: usize> 
    TrajectoryDynamics<S, D, A_DIM, T_DIM, U_DIM, W_DIM, TOTAL_DIM>
where
    S: RealField,
    D: Diff<S, TOTAL_DIM, T_DIM>,
{
    /// Constructor linking a concrete differentiation engine (AutoDiff or NormalDiff)
    #[inline]
    pub fn new(diff_engine: D) -> Self {
        Self {
            diff_engine,
            _marker: core::marker::PhantomData,
        }
    }

    /// Evaluates the continuous-time tangent velocity mapping: f(x, u, w) ∈ T_x M
    #[inline]
    pub fn evaluate_velocity(
        &self, 
        ambient_state: &SVector<S, A_DIM>, 
        control: &SVector<S, U_DIM>,
        noise: &SVector<S, W_DIM>
    ) -> SVector<S, T_DIM> {
        let mut packed_input = SVector::<S, TOTAL_DIM>::zeros();
        packed_input.fixed_rows_mut::<A_DIM>(0).copy_from(ambient_state);
        packed_input.fixed_rows_mut::<U_DIM>(A_DIM).copy_from(control);
        packed_input.fixed_rows_mut::<W_DIM>({A_DIM + U_DIM}).copy_from(noise);
        
        self.diff_engine.eval(&packed_input)
    }

    /// Linearizes the system physics, extracting both the State Error Transition matrix (F_x)
    /// and the Noise Diffusion matrix (G_w) in a single stack-allocated pass.
    #[inline]
    pub fn linearize<M>(&self, state: &M, control: &SVector<S, U_DIM>) -> (SMatrix<S, T_DIM, T_DIM>, SMatrix<S, T_DIM, W_DIM>)
    where
        M: Manifold<S, A_DIM, T_DIM>,
    {
        // 1. Pack state and control. Process noise expectation is nominally zero E[w] = 0
        let mut packed_input = SVector::<S, TOTAL_DIM>::zeros();
        packed_input.fixed_rows_mut::<A_DIM>(0).copy_from(&state.to_ambient());
        packed_input.fixed_rows_mut::<U_DIM>(A_DIM).copy_from(control);

        // 2. Compute the raw combined Jacobian layout via the inner engine: size (T_DIM x (A_DIM + U_DIM + W_DIM))
        let raw_jacobian = self.diff_engine.jacobian(&packed_input);

        // 3. Extract the ambient state Jacobian and push it down onto the local tangent frame
        let j_ambient = raw_jacobian.fixed_columns::<A_DIM>(0);
        let j_oplus = state.pushforward_jacobian();
        let f_tangent = j_ambient * j_oplus; // Continuous-time state transition matrix (T_DIM x T_DIM)

        // 4. Extract the raw noise columns directly (Noise lives in a flat vector space, no pushforward required)
        let g_tangent = raw_jacobian.fixed_columns::<W_DIM>({A_DIM + U_DIM}).into_owned(); // (T_DIM x W_DIM)

        (f_tangent, g_tangent)
    }
}

#[derive(Clone, Debug)]
pub struct TimeVaryingConstraint<S, D, const A_DIM: usize, const T_DIM: usize, const C_DIM: usize> {
    pub diff_engine: D,
    _marker: core::marker::PhantomData<S>,
}

impl<S, D, const A_DIM: usize, const T_DIM: usize, const C_DIM: usize> 
    TimeVaryingConstraint<S, D, A_DIM, T_DIM, C_DIM>
where
    S: RealField,
    D: Diff<S, A_DIM, C_DIM>,
{
    /// Constructor linking an AutoDiff or NormalDiff math engine representation.
    #[inline(always)]
    pub fn new(diff_engine: D) -> Self {
        Self { diff_engine, _marker: core::marker::PhantomData }
    }

    /// Evaluates the raw algebraic constraint residual equation h(x)
    #[inline(always)]
    pub fn evaluate(&self, ambient_state: &SVector<S, A_DIM>) -> SVector<S, C_DIM> {
        self.diff_engine.eval(ambient_state)
    }

    /// Linearizes the constraint equation, projecting the raw ambient Jacobian H_ambient 
    /// directly down into the manifold's local tangent frame: H_tangent = H_ambient * J_oplus
    #[inline(always)]
    pub fn linearize_tangent<M>(&self, state: &M) -> SMatrix<S, C_DIM, T_DIM>
    where
        M: Manifold<S, A_DIM, T_DIM>,
    {
        // 1. Calculate the ambient matrix via the inner differentiation engine (C_DIM x A_DIM)
        let h_ambient = self.diff_engine.jacobian(&state.to_ambient());

        // 2. Extract the local pushforward matrix from the manifold (A_DIM x T_DIM)
        let j_oplus = state.pushforward_jacobian();

        // 3. Project down onto the flat tangent space slice (C_DIM x T_DIM)
        h_ambient * j_oplus
    }
}


#[cfg(test)]
mod tests {
    use super::{TrajectoryDynamics, TimeVaryingConstraint};
    use crate::diff::{NormalDiff, AutoDiff, DiffFn};
    use crate::manifold::{ComplexRotation, RealLine, Manifold};
    use nalgebra::{RealField, SVector, SMatrix};

    const EPS: f64 = 1e-9;

    // Builds a generic scalar constant from an f64 literal.
    #[inline]
    fn c<S: RealField>(x: f64) -> S {
        nalgebra::convert(x)
    }

    // =========================================================================
    // Fixtures: closed-form dynamics / constraint functions used by the engines.
    // NormalDiff needs an explicit value fn AND its analytic Jacobian.
    // =========================================================================

    // Linear velocity on a 1-D system, packed input = [x, u, w] (TOTAL_DIM = 3).
    //   f(x, u, w) = -2x + 3u + w        =>  J = [-2, 3, 1]
    fn lin_vel(p: &SVector<f64, 3>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(-2.0 * p[0] + 3.0 * p[1] + 1.0 * p[2])
    }
    fn lin_vel_jac(_p: &SVector<f64, 3>) -> SMatrix<f64, 1, 3> {
        SMatrix::<f64, 1, 3>::from_row_slice(&[-2.0, 3.0, 1.0])
    }

    // Linear velocity on a 2-D ambient system, packed input = [z0, z1, u, w]
    // (TOTAL_DIM = 4).  f = 2*z0 + 5*z1 + 7*u + 11*w   =>  J = [2, 5, 7, 11]
    fn rot_vel(p: &SVector<f64, 4>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(2.0 * p[0] + 5.0 * p[1] + 7.0 * p[2] + 11.0 * p[3])
    }
    fn rot_vel_jac(_p: &SVector<f64, 4>) -> SMatrix<f64, 1, 4> {
        SMatrix::<f64, 1, 4>::from_row_slice(&[2.0, 5.0, 7.0, 11.0])
    }

    // Nonlinear velocity for the AutoDiff path, packed input = [z0, z1, u, w].
    //   f = z0^2 + 3u + 4w   =>  J = [2*z0, 0, 3, 4]
    struct NonlinearVelAd;
    impl DiffFn<4, 1> for NonlinearVelAd {
        fn eval<S: RealField + Copy>(&self, p: &SVector<S, 4>, y: &mut SVector<S, 1>) {
            y[0] = p[0] * p[0] + p[2] * c(3.0) + p[3] * c(4.0);
        }
    }

    // Unit-norm constraint on the ambient circle:  h(z) = z0^2 + z1^2 - 1
    fn unit_norm(z: &SVector<f64, 2>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(z[0] * z[0] + z[1] * z[1] - 1.0)
    }
    fn unit_norm_jac(z: &SVector<f64, 2>) -> SMatrix<f64, 1, 2> {
        SMatrix::<f64, 1, 2>::from_row_slice(&[2.0 * z[0], 2.0 * z[1]])
    }

    // Coordinate constraint:  h(z) = z0   =>  J = [1, 0]
    fn x_coord(z: &SVector<f64, 2>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(z[0])
    }
    fn x_coord_jac(_z: &SVector<f64, 2>) -> SMatrix<f64, 1, 2> {
        SMatrix::<f64, 1, 2>::from_row_slice(&[1.0, 0.0])
    }

    // Two-row constraint stacking the two above:  h(z) = [z0^2 + z1^2 - 1, z0]
    fn combined(z: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::new(z[0] * z[0] + z[1] * z[1] - 1.0, z[0])
    }
    fn combined_jac(z: &SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::from_row_slice(&[
            2.0 * z[0], 2.0 * z[1],
            1.0,        0.0,
        ])
    }

    // Scalar constraint on the real line:  h(x) = 3x  =>  J = [3]
    fn scale3(x: &SVector<f64, 1>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(3.0 * x[0])
    }
    fn scale3_jac(_x: &SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(3.0)
    }

    // -------------------------------------------------------------------------
    // TrajectoryDynamics::evaluate_velocity — packing of [state | control | noise]
    // -------------------------------------------------------------------------
    #[test]
    fn test_evaluate_velocity_packs_inputs() {
        let sys: TrajectoryDynamics<f64, NormalDiff<f64, 3, 1>, 1, 1, 1, 1, 3> =
            TrajectoryDynamics::new(NormalDiff::new(lin_vel, lin_vel_jac));

        let state = SVector::<f64, 1>::new(4.0);
        let control = SVector::<f64, 1>::new(5.0);
        let noise = SVector::<f64, 1>::new(6.0);

        // -2*4 + 3*5 + 1*6 = 13
        let vel = sys.evaluate_velocity(&state, &control, &noise);
        assert!((vel[0] - 13.0).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // TrajectoryDynamics::linearize on a flat (vector-space) manifold.
    // RealLine pushforward is the identity, so F == ambient state Jacobian and
    // G == the noise columns of the raw Jacobian, untouched.
    // -------------------------------------------------------------------------
    #[test]
    fn test_linearize_real_line() {
        let sys: TrajectoryDynamics<f64, NormalDiff<f64, 3, 1>, 1, 1, 1, 1, 3> =
            TrajectoryDynamics::new(NormalDiff::new(lin_vel, lin_vel_jac));

        let state = RealLine { x: SVector::<f64, 1>::new(4.0) };
        let control = SVector::<f64, 1>::new(5.0);

        let (f, g) = sys.linearize(&state, &control);

        // J_ambient = [-2], J_oplus = [1]  =>  F = -2 ;  G = noise column = [1]
        assert!((f[(0, 0)] - (-2.0)).abs() < EPS);
        assert!((g[(0, 0)] - 1.0).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // TrajectoryDynamics::linearize at the SO(2) identity.
    // J_oplus(identity) = [-y, x]^T = [0, 1]^T, so the rotation's x-column drops
    // out of F and only the y-column survives.
    // -------------------------------------------------------------------------
    #[test]
    fn test_linearize_rotation_identity() {
        let sys: TrajectoryDynamics<f64, NormalDiff<f64, 4, 1>, 2, 1, 1, 1, 4> =
            TrajectoryDynamics::new(NormalDiff::new(rot_vel, rot_vel_jac));

        let state = ComplexRotation::identity(); // z = (1, 0)
        let control = SVector::<f64, 1>::new(0.0);

        let (f, g) = sys.linearize(&state, &control);

        // J_ambient = [2, 5]; J_oplus = [-0, 1]^T  =>  F = 2*0 + 5*1 = 5
        // noise column (index A_DIM + U_DIM = 3) = 11
        assert_eq!(f[(0, 0)], 5.0);
        assert_eq!(g[(0, 0)], 11.0);
    }

    // -------------------------------------------------------------------------
    // Same linearization at a generic rotated state, exercising a nontrivial
    // pushforward projection F = J_ambient * J_oplus.
    // -------------------------------------------------------------------------
    #[test]
    fn test_linearize_rotation_generic() {
        let sys: TrajectoryDynamics<f64, NormalDiff<f64, 4, 1>, 2, 1, 1, 1, 4> =
            TrajectoryDynamics::new(NormalDiff::new(rot_vel, rot_vel_jac));

        // Unit complex number (0.6, 0.8).
        let state = ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) };
        let control = SVector::<f64, 1>::new(0.0);

        let (f, g) = sys.linearize(&state, &control);

        // J_oplus = [-0.8, 0.6]^T  =>  F = 2*(-0.8) + 5*(0.6) = 1.4
        assert!((f[(0, 0)] - 1.4).abs() < EPS);
        assert!((g[(0, 0)] - 11.0).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // TrajectoryDynamics driven by AutoDiff instead of analytic NormalDiff.
    // Verifies the dual-number Jacobian flows through evaluate / linearize and
    // is then pushed onto the SO(2) tangent frame identically.
    // -------------------------------------------------------------------------
    #[test]
    fn test_linearize_autodiff_rotation() {
        let sys: TrajectoryDynamics<f64, AutoDiff<NonlinearVelAd>, 2, 1, 1, 1, 4> =
            TrajectoryDynamics::new(AutoDiff::new(NonlinearVelAd));

        let state = ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) };
        let control = SVector::<f64, 1>::new(2.0);

        // f = z0^2 + 3u + 4w at (0.6, 0.8, 2, 0) = 0.36 + 6 = 6.36
        let vel = sys.evaluate_velocity(&state.to_ambient(), &control, &SVector::<f64, 1>::zeros());
        assert!((vel[0] - 6.36).abs() < EPS);

        let (f, g) = sys.linearize(&state, &control);
        // J_ambient = [2*z0, 0] = [1.2, 0]; J_oplus = [-0.8, 0.6]^T
        //   F = 1.2*(-0.8) + 0*(0.6) = -0.96 ;  noise column = 4
        assert!((f[(0, 0)] - (-0.96)).abs() < EPS);
        assert!((g[(0, 0)] - 4.0).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // TimeVaryingConstraint::evaluate — raw residual, no projection.
    // -------------------------------------------------------------------------
    #[test]
    fn test_constraint_evaluate() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 2, 1>, 2, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(unit_norm, unit_norm_jac));

        // On the circle -> residual 0; off the circle -> nonzero.
        let on = con.evaluate(&SVector::<f64, 2>::new(0.6, 0.8));
        assert!(on[0].abs() < EPS);

        let off = con.evaluate(&SVector::<f64, 2>::new(1.0, 1.0));
        assert!((off[0] - 1.0).abs() < EPS); // 1 + 1 - 1
    }

    // -------------------------------------------------------------------------
    // TimeVaryingConstraint::linearize_tangent for a ROTATION-INVARIANT
    // constraint. The unit-norm residual is unchanged by rotation, so its
    // gradient along the SO(2) tangent direction must vanish.
    // -------------------------------------------------------------------------
    #[test]
    fn test_constraint_linearize_invariant_is_zero() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 2, 1>, 2, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(unit_norm, unit_norm_jac));

        let state = ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) };
        let h_tangent = con.linearize_tangent(&state);

        // H_ambient = [1.2, 1.6]; J_oplus = [-0.8, 0.6]^T
        //   => 1.2*(-0.8) + 1.6*(0.6) = 0
        assert!(h_tangent[(0, 0)].abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // A constraint that is NOT rotation-invariant projects to a nonzero tangent.
    // -------------------------------------------------------------------------
    #[test]
    fn test_constraint_linearize_nonzero() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 2, 1>, 2, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(x_coord, x_coord_jac));

        let state = ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) };
        let h_tangent = con.linearize_tangent(&state);

        // H_ambient = [1, 0]; J_oplus = [-0.8, 0.6]^T  =>  -0.8
        assert!((h_tangent[(0, 0)] - (-0.8)).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // Multi-row constraint (C_DIM = 2): checks the full (C_DIM x T_DIM) shape of
    // the projected Jacobian, combining an invariant row with a coordinate row.
    // -------------------------------------------------------------------------
    #[test]
    fn test_constraint_linearize_multirow() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 2, 2>, 2, 1, 2> =
            TimeVaryingConstraint::new(NormalDiff::new(combined, combined_jac));

        let state = ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) };
        let h_tangent = con.linearize_tangent(&state); // 2 x 1

        // Row 0 (unit-norm, invariant) -> 0 ; Row 1 (x-coord) -> -0.8
        assert!(h_tangent[(0, 0)].abs() < EPS);
        assert!((h_tangent[(1, 0)] - (-0.8)).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // Constraint linearization on a flat manifold: identity pushforward means
    // the tangent Jacobian equals the ambient Jacobian.
    // -------------------------------------------------------------------------
    #[test]
    fn test_constraint_linearize_real_line() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 1, 1>, 1, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(scale3, scale3_jac));

        let state = RealLine { x: SVector::<f64, 1>::new(4.0) };

        let residual = con.evaluate(&state.to_ambient());
        assert!((residual[0] - 12.0).abs() < EPS); // 3 * 4

        let h_tangent = con.linearize_tangent(&state);
        assert!((h_tangent[(0, 0)] - 3.0).abs() < EPS);
    }
}
