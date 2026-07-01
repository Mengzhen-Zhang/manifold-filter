#![no_std]
use nalgebra::{SMatrix, SVector, RealField};
use manifold::manifold::Manifold;
use manifold::manifold::TrajectoryDynamics;
use manifold::manifold::TimeVaryingConstraint;

const TOLERANCE: f64 = 1e-8;

/// Mean-propagation scheme for the [`FilterState::predict`] time update. The
/// covariance step is unaffected by the choice (the error-state transition is
/// dominated by the attitude increment `ω·dt` and the flat p/v blocks, neither
/// of which the scheme changes); only the nominal-state advance differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    /// First-order Euler: one dynamics evaluation, `x ⊕ f(x)·dt`. Cheapest, and
    /// exact for the attitude (constant-ω retract), but rectangle-integrates the
    /// translational state — it grows over-confident under sustained dynamics.
    Euler,
    /// Second-order midpoint (RK2): re-evaluates the dynamics at the half-step,
    /// capturing the within-step state curvature (rotating velocity / attitude)
    /// for one extra dynamics evaluation. See the `integration_fidelity` example.
    Midpoint,
}

/// A unified tracking envelope holding the nominal manifold state
/// and its associated flat local tangent covariance.
#[derive(Clone, Debug)]
pub struct FilterState<S, M, const A_DIM: usize, const T_DIM: usize> {
    /// The nominal state sitting directly on the curved manifold sheet
    pub state: M,
    /// Tangent space uncertainty covariance matrix (T_DIM x T_DIM)
    pub covariance: SMatrix<S, T_DIM, T_DIM>,
}

impl<S, M, const A_DIM: usize, const T_DIM: usize> FilterState<S, M, A_DIM, T_DIM>
where
    S: RealField + Copy,
    M: Manifold<S, A_DIM, T_DIM>,
{
    /// Instantiates a raw state-estimation node on the stack
    #[inline(always)]
    pub fn new(initial_state: M, initial_covariance: SMatrix<S, T_DIM, T_DIM>) -> Self {
        Self {
            state: initial_state,
            covariance: initial_covariance,
        }
    }

    /// Pure EKF time update: steps the nominal state along the manifold and
    /// propagates the tangent covariance. Use `predict_with_constraint` when the
    /// dynamics must stay on a hard constraint surface.
    #[inline]
    pub fn predict<DD, const U_DIM: usize, const W_DIM: usize, const TOTAL_DIM: usize>(
	&mut self,
	dynamics: &TrajectoryDynamics<S, DD, A_DIM, T_DIM, U_DIM, W_DIM, TOTAL_DIM>,
	control: &SVector<S, U_DIM>,
	process_noise_covariance: &SMatrix<S, W_DIM, W_DIM>,
	dt: S,
	integrator: Integrator,
    ) where
	DD: manifold::diff::Diff<S, TOTAL_DIM, T_DIM>,
    {
	let (f_continuous, g_continuous) = dynamics.linearize(&self.state, control);

	let zero_noise = SVector::<S, W_DIM>::zeros();
	// Mean slope over the step. Euler samples the dynamics once at the start;
	// midpoint re-evaluates at the half-step state (RK2), capturing within-step
	// state curvature. Control is held across the step either way — the relevant
	// curvature is state-dependent, not input-dependent (see integration_fidelity).
	let velocity = match integrator {
	    Integrator::Euler => {
		dynamics.evaluate_velocity(&self.state.to_ambient(), control, &zero_noise)
	    }
	    Integrator::Midpoint => {
		let half = S::from_f64(0.5).expect("cannot convert from f64");
		let k1 = dynamics.evaluate_velocity(&self.state.to_ambient(), control, &zero_noise);
		let mid = self.state.retract(&M::vector_to_tangent(&(k1 * (dt * half))));
		dynamics.evaluate_velocity(&mid.to_ambient(), control, &zero_noise)
	    }
	};

	let displacement = velocity * dt;
	let disp_tangent = M::vector_to_tangent(&displacement);
	let neg_disp_tangent = M::vector_to_tangent(&(-displacement));

	// Discrete state transition F = Ad_{Exp(−δ)} + F_cont·dt. The leading term
	// is the prior error transported through the nominal step — the full
	// adjoint, derived from parallel transport at ±δ: Ad_{Exp(−δ)} = T(δ)·T(−δ)⁻¹.
	let t_fwd = self.state.parallel_transport(&disp_tangent);
	let t_bwd = self.state.parallel_transport(&neg_disp_tangent);
	let adjoint = &t_fwd * t_bwd.try_inverse().expect("predict: non-invertible parallel transport");
	let f_discrete = &adjoint + (&f_continuous * dt);
	let g_discrete = &g_continuous * dt;

	self.state = self.state.retract(&disp_tangent);
	self.covariance = {
	    let f = &f_discrete;
	    let cov = &self.covariance;
	    let g = &g_discrete;
	    let n = process_noise_covariance;
	    f * cov * f.transpose() + g * n * g.transpose()
	};
    }

    /// Time update that keeps the state on a hard constraint surface: a `predict`
    /// step followed by an in-loop projection back onto `constraint`. The
    /// constraint is passed directly (not as an `Option`), so its generics are
    /// always inferred at the call site.
    #[inline]
    pub fn predict_with_constraint<DD, DC, const U_DIM: usize, const W_DIM: usize, const TOTAL_DIM: usize, const C_DIM: usize>(
	&mut self,
	dynamics: &TrajectoryDynamics<S, DD, A_DIM, T_DIM, U_DIM, W_DIM, TOTAL_DIM>,
	control: &SVector<S, U_DIM>,
	process_noise_covariance: &SMatrix<S, W_DIM, W_DIM>,
	constraint: &TimeVaryingConstraint<S, DC, A_DIM, T_DIM, C_DIM>,
	dt: S,
	integrator: Integrator,
    ) where
	DD: manifold::diff::Diff<S, TOTAL_DIM, T_DIM>,
	DC: manifold::diff::Diff<S, A_DIM, C_DIM>,
    {
	self.predict(dynamics, control, process_noise_covariance, dt, integrator);
	let tolerance = S::from_f64(TOLERANCE).expect("cannot convert from f64");
	self.project_hard_constraints(constraint, tolerance, 5);
    }

    /// Measurement update from a raw sensor outcome `y`: forms the Euclidean
    /// innovation `y − h(x)` internally and applies the correction. Reach for
    /// `correct_with_innovation` only when the residual is not a plain
    /// subtraction (e.g. a manifold-valued measurement).
    #[inline]
    pub fn correct_measurement<D, const M_DIM: usize>(
        &mut self,
        measurement_engine: &TimeVaryingConstraint<S, D, A_DIM, T_DIM, M_DIM>,
        measurement: &SVector<S, M_DIM>, // The raw sensor outcome y
        r_covariance: &SMatrix<S, M_DIM, M_DIM>, // Sensor measurement noise covariance (R)
    ) where
        D: manifold::diff::Diff<S, A_DIM, M_DIM>,
    {
	let predicted = measurement_engine.evaluate(&self.state.to_ambient());
	self.correct_with_innovation(measurement_engine, &(*measurement - predicted), r_covariance);
    }

    /// Lower-level measurement update from a pre-computed innovation `y − h(x)`.
    #[inline]
    pub fn correct_with_innovation<D, const M_DIM: usize>(
        &mut self,
        measurement_engine: &TimeVaryingConstraint<S, D, A_DIM, T_DIM, M_DIM>,
        innovation: &SVector<S, M_DIM>,
        r_covariance: &SMatrix<S, M_DIM, M_DIM>,
    ) where
        D: manifold::diff::Diff<S, A_DIM, M_DIM>,
    {
	let h_tangent = measurement_engine.linearize_tangent(&self.state);
	let s_matrix = (&h_tangent * &self.covariance * h_tangent.transpose()) + r_covariance;

	self.apply_tangent_update(&h_tangent, innovation, s_matrix);
    }

    #[inline]
    pub fn project_hard_constraints<DC, const C_DIM: usize>(
        &mut self,
        constraint: &TimeVaryingConstraint<S, DC, A_DIM, T_DIM, C_DIM>,
        tolerance: S,
        max_iterations: usize,
    ) where
        DC: manifold::diff::Diff<S, A_DIM, C_DIM>,
    {
        for _ in 0..max_iterations {
            // Evaluate absolute global ambient violation: h(x)
            let residual = constraint.evaluate(&self.state.to_ambient());
            if residual.one_norm() < tolerance {
                break; // Fully satisfied, exit early!
            }

            // Compute combined tangent projector matrix: H = H_ambient * J_oplus
            let h_tangent = constraint.linearize_tangent(&self.state);
            let s_matrix = &h_tangent * &self.covariance * h_tangent.transpose();

            if !self.apply_tangent_update(&h_tangent, &(-residual), s_matrix) {
		break;
	    }
        }
    }

    #[inline(always)]
    fn apply_tangent_update<const DIM: usize>(
        &mut self,
        h_tangent: &SMatrix<S, DIM, T_DIM>,
        innovation_or_residual: &SVector<S, DIM>,
        s_matrix: SMatrix<S, DIM, DIM>,
    ) -> bool {
        let identity_t = SMatrix::<S, T_DIM, T_DIM>::identity();

        if let Some(chol) = s_matrix.cholesky() {
            // 1. Compute optimal Gain matrix: K = P * H^T * S^-1
            let gain = &self.covariance * h_tangent.transpose() * chol.inverse();

            // 2. Map the innovation to a localized tangent space correction vector
            let tangent_correction = &gain * innovation_or_residual;
            let delta = M::vector_to_tangent(&tangent_correction);

            // 3. Parallel transport of the correction, evaluated at the OLD
            //    anchor, to carry the covariance into the new tangent frame below.
            let transport = self.state.parallel_transport(&delta);

            // 4. Step the nominal state mean along the curved manifold surface via retraction
            self.state = self.state.retract(&delta);

            // 5. Reduce uncertainty variance safely (Standard form: P = (I - KH) * P)
            self.covariance = (&identity_t - (&gain * h_tangent)) * &self.covariance;

            // 6. Reframe the covariance into the new tangent space: P = G * P * G^T
            self.covariance = &transport * &self.covariance * transport.transpose();

            // 7. Force matrix symmetry to clean out compounding rounding artifacts
            self.covariance = (&self.covariance + &self.covariance.transpose()) * S::from_f64(0.5).expect("cannot convert from f64");

            true
        } else {
            false // Singularity or numerical breakdown guard
        }
    }
}



#[cfg(test)]
mod tests {
    use super::{FilterState, Integrator};
    use manifold::diff::{AutoDiff, DiffFn, NormalDiff};
    use manifold::manifold::{
        ComplexRotation, ProductSpace, RealLine, TimeVaryingConstraint, TrajectoryDynamics,
    };
    use nalgebra::{RealField, SMatrix, SVector};

    const EPS: f64 = 1e-9;

    // Builds a generic scalar constant from an f64 literal.
    #[inline]
    fn c<S: RealField>(x: f64) -> S {
        nalgebra::convert(x)
    }

    // =========================================================================
    // Fixtures: closed-form dynamics / measurement / constraint functions.
    // NormalDiff needs an explicit value fn AND its analytic Jacobian.
    // =========================================================================

    // 1-D dynamics, packed input = [x, u, w] (TOTAL_DIM = 3).
    //   f(x, u, w) = -2x + 3u + w        =>  J = [-2, 3, 1]
    fn lin_vel(p: &SVector<f64, 3>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(-2.0 * p[0] + 3.0 * p[1] + 1.0 * p[2])
    }
    fn lin_vel_jac(_p: &SVector<f64, 3>) -> SMatrix<f64, 1, 3> {
        SMatrix::<f64, 1, 3>::from_row_slice(&[-2.0, 3.0, 1.0])
    }

    // 2-D ambient dynamics, packed input = [z0, z1, u, w] (TOTAL_DIM = 4).
    //   f = 2*z0 + 5*z1 + 7*u + 11*w     =>  J = [2, 5, 7, 11]
    fn rot_vel(p: &SVector<f64, 4>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(2.0 * p[0] + 5.0 * p[1] + 7.0 * p[2] + 11.0 * p[3])
    }
    fn rot_vel_jac(_p: &SVector<f64, 4>) -> SMatrix<f64, 1, 4> {
        SMatrix::<f64, 1, 4>::from_row_slice(&[2.0, 5.0, 7.0, 11.0])
    }

    // Nonlinear dynamics for the AutoDiff path, packed input = [z0, z1, u, w].
    // The dual-number engine differentiates this; no hand-written Jacobian.
    //   f = z0^2 + 3u + 4w   =>  J = [2*z0, 0, 3, 4]
    struct NonlinearVelAd;
    impl DiffFn<4, 1> for NonlinearVelAd {
        fn eval<S: RealField + Copy>(&self, p: &SVector<S, 4>, y: &mut SVector<S, 1>) {
            y[0] = p[0] * p[0] + p[2] * c(3.0) + p[3] * c(4.0);
        }
    }

    // Scalar measurement / constraint on the real line:  h(x) = 3x  =>  J = [3]
    fn scale3(x: &SVector<f64, 1>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(3.0 * x[0])
    }
    fn scale3_jac(_x: &SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(3.0)
    }

    // Unit-norm constraint on the ambient circle:  h(z) = z0^2 + z1^2 - 1.
    // It is rotation-invariant, so its gradient along the SO(2) tangent vanishes.
    fn unit_norm(z: &SVector<f64, 2>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(z[0] * z[0] + z[1] * z[1] - 1.0)
    }
    fn unit_norm_jac(z: &SVector<f64, 2>) -> SMatrix<f64, 1, 2> {
        SMatrix::<f64, 1, 2>::from_row_slice(&[2.0 * z[0], 2.0 * z[1]])
    }

    // Measurement on the (SO(2) x R) product, ambient = [z0, z1, x] (A_DIM = 3).
    //   h = [z1, x]   =>  J = [[0, 1, 0], [0, 0, 1]]
    fn obs_z1_x(p: &SVector<f64, 3>) -> SVector<f64, 2> {
        SVector::<f64, 2>::new(p[1], p[2])
    }
    fn obs_z1_x_jac(_p: &SVector<f64, 3>) -> SMatrix<f64, 2, 3> {
        SMatrix::<f64, 2, 3>::from_row_slice(&[
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ])
    }

    // Convenience alias for the product manifold used below.
    type So2xR = ProductSpace<ComplexRotation<f64>, RealLine<f64>, 2, 1, 1, 1, 3, 2>;

    // -------------------------------------------------------------------------
    // new — stores the supplied state and covariance verbatim.
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_stores_state_and_covariance() {
        let fs = FilterState::<f64, RealLine<f64>, 1, 1>::new(
            RealLine { x: SVector::<f64, 1>::new(7.0) },
            SMatrix::<f64, 1, 1>::new(0.25),
        );
        assert_eq!(fs.state.x[0], 7.0);
        assert_eq!(fs.covariance[(0, 0)], 0.25);
    }

    // -------------------------------------------------------------------------
    // predict on a flat (RealLine) manifold, no hard constraint.
    // The pushforward is the identity, so the mean and covariance follow the
    // textbook discrete EKF time update exactly.
    // -------------------------------------------------------------------------
    #[test]
    fn test_predict_real_line_no_constraint() {
        let dynamics: TrajectoryDynamics<f64, NormalDiff<f64, 3, 1>, 1, 1, 1, 1, 3> =
            TrajectoryDynamics::new(NormalDiff::new(lin_vel, lin_vel_jac));

        let mut fs = FilterState::<f64, RealLine<f64>, 1, 1>::new(
            RealLine { x: SVector::<f64, 1>::new(4.0) },
            SMatrix::<f64, 1, 1>::new(0.5),
        );

        let control = SVector::<f64, 1>::new(5.0);
        let process_noise = SMatrix::<f64, 1, 1>::new(2.0);
        let dt = 0.1;

        fs.predict(&dynamics, &control, &process_noise, dt, Integrator::Euler);

        // velocity = -2*4 + 3*5 = 7 ;  x_new = 4 + 7*0.1 = 4.7
        assert!((fs.state.x[0] - 4.7).abs() < EPS);

        // F_d = 1 + (-2)*0.1 = 0.8 ;  G_d = 1*0.1 = 0.1
        // P_new = 0.8^2 * 0.5 + 0.1^2 * 2.0 = 0.32 + 0.02 = 0.34
        assert!((fs.covariance[(0, 0)] - 0.34).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // predict on SO(2): the mean steps along the curved sheet via retraction
    // (staying on the unit circle) while the covariance uses the pushed-forward
    // tangent transition F and noise diffusion G.
    // -------------------------------------------------------------------------
    #[test]
    fn test_predict_rotation_stays_on_circle() {
        let dynamics: TrajectoryDynamics<f64, NormalDiff<f64, 4, 1>, 2, 1, 1, 1, 4> =
            TrajectoryDynamics::new(NormalDiff::new(rot_vel, rot_vel_jac));

        let mut fs = FilterState::<f64, ComplexRotation<f64>, 2, 1>::new(
            ComplexRotation::identity(),
            SMatrix::<f64, 1, 1>::new(0.1),
        );

        let control = SVector::<f64, 1>::new(0.0);
        let process_noise = SMatrix::<f64, 1, 1>::new(1.0);
        let dt = 0.01;

        fs.predict(&dynamics, &control, &process_noise, dt, Integrator::Euler);

        // velocity = 2*1 + 5*0 + 7*0 = 2 ;  rotation angle = 2*0.01 = 0.02
        let angle = 0.02_f64;
        assert!((fs.state.z[0] - angle.cos()).abs() < EPS);
        assert!((fs.state.z[1] - angle.sin()).abs() < EPS);
        // The retraction keeps the nominal state on the manifold (unit norm).
        assert!((fs.state.z.norm() - 1.0).abs() < EPS);

        // F = 5, G = 11 at the identity (see manifold linearize tests).
        // F_d = 1 + 5*0.01 = 1.05 ;  G_d = 11*0.01 = 0.11
        // P_new = 1.05^2 * 0.1 + 0.11^2 * 1.0 = 0.110250 + 0.0121 = 0.12235
        assert!((fs.covariance[(0, 0)] - 0.12235).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // predict is generic over the differentiation engine. This drives it with
    // an AutoDiff (dual-number) engine instead of an analytic NormalDiff and
    // checks the dual-number Jacobian flows through linearize -> covariance
    // update identically, matching the same hand-computed closed form.
    // -------------------------------------------------------------------------
    #[test]
    fn test_predict_rotation_autodiff_matches_closed_form() {
        let dynamics: TrajectoryDynamics<f64, AutoDiff<NonlinearVelAd>, 2, 1, 1, 1, 4> =
            TrajectoryDynamics::new(AutoDiff::new(NonlinearVelAd));

        let mut fs = FilterState::<f64, ComplexRotation<f64>, 2, 1>::new(
            ComplexRotation { z: SVector::<f64, 2>::new(0.6, 0.8) },
            SMatrix::<f64, 1, 1>::new(0.2),
        );

        let control = SVector::<f64, 1>::new(2.0);
        let process_noise = SMatrix::<f64, 1, 1>::new(1.0);
        let dt = 0.1;

        fs.predict(&dynamics, &control, &process_noise, dt, Integrator::Euler);

        // velocity = z0^2 + 3u = 0.36 + 6 = 6.36 ;  angle step = 6.36*0.1 = 0.636
        // SO(2) retraction adds the angle step to the starting heading.
        let end_angle = 0.8_f64.atan2(0.6) + 0.636;
        assert!((fs.state.z[0] - end_angle.cos()).abs() < EPS);
        assert!((fs.state.z[1] - end_angle.sin()).abs() < EPS);
        assert!((fs.state.z.norm() - 1.0).abs() < EPS);

        // AutoDiff Jacobian: J_ambient = [2*z0, 0] = [1.2, 0]; J_oplus = [-0.8, 0.6]
        //   F = 1.2*(-0.8) + 0*0.6 = -0.96 ;  noise column G = 4
        // F_d = 1 + (-0.96)*0.1 = 0.904 ;  G_d = 4*0.1 = 0.4
        // P_new = 0.904^2 * 0.2 + 0.4^2 * 1.0 = 0.1634432 + 0.16 = 0.3234432
        assert!((fs.covariance[(0, 0)] - 0.3234432).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // correct_measurement on a flat manifold against the closed-form scalar
    // Kalman update.  Innovation is supplied pre-computed, as documented.
    // -------------------------------------------------------------------------
    #[test]
    fn test_correct_measurement_real_line() {
        let meas: TimeVaryingConstraint<f64, NormalDiff<f64, 1, 1>, 1, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(scale3, scale3_jac));

        let mut fs = FilterState::<f64, RealLine<f64>, 1, 1>::new(
            RealLine { x: SVector::<f64, 1>::new(4.0) },
            SMatrix::<f64, 1, 1>::new(2.0),
        );

        // Raw measurement; h(x) = 3·4 = 12, so the internal innovation is 1.5.
        let measurement = SVector::<f64, 1>::new(13.5);
        let r = SMatrix::<f64, 1, 1>::new(1.0);
        fs.correct_measurement(&meas, &measurement, &r);

        // H = 3 ;  S = 3*2*3 + 1 = 19 ;  K = 2*3/19 = 6/19
        // dx = K * innovation = (6/19)*1.5 = 9/19
        assert!((fs.state.x[0] - (4.0 + 9.0 / 19.0)).abs() < EPS);
        // P_new = (1 - K*H) * P = (1 - 18/19) * 2 = 2/19
        assert!((fs.covariance[(0, 0)] - 2.0 / 19.0).abs() < EPS);
    }

    // correct_measurement(y) must equal correct_with_innovation(y − h(x)).
    #[test]
    fn correct_measurement_matches_innovation_form() {
        let meas: TimeVaryingConstraint<f64, NormalDiff<f64, 1, 1>, 1, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(scale3, scale3_jac));
        let r = SMatrix::<f64, 1, 1>::new(1.0);
        let make = || {
            FilterState::<f64, RealLine<f64>, 1, 1>::new(
                RealLine { x: SVector::<f64, 1>::new(4.0) },
                SMatrix::<f64, 1, 1>::new(2.0),
            )
        };

        let mut a = make();
        a.correct_measurement(&meas, &SVector::<f64, 1>::new(13.5), &r);

        // Same update, innovation form: y − h(x) = 13.5 − 3·4 = 1.5.
        let mut b = make();
        b.correct_with_innovation(&meas, &SVector::<f64, 1>::new(1.5), &r);

        assert!((a.state.x[0] - b.state.x[0]).abs() < EPS);
        assert!((a.covariance[(0, 0)] - b.covariance[(0, 0)]).abs() < EPS);
    }

    // -------------------------------------------------------------------------
    // correct_measurement on a 2-D-tangent product manifold checks the EKF
    // invariants that hold for any valid measurement: the posterior covariance
    // stays symmetric, its diagonal stays positive, and a measurement can only
    // shrink total uncertainty (trace decreases).
    // -------------------------------------------------------------------------
    #[test]
    fn test_correct_measurement_product_invariants() {
        let meas: TimeVaryingConstraint<f64, NormalDiff<f64, 3, 2>, 3, 2, 2> =
            TimeVaryingConstraint::new(NormalDiff::new(obs_z1_x, obs_z1_x_jac));

        let state = So2xR {
            m1: ComplexRotation::identity(),
            m2: RealLine { x: SVector::<f64, 1>::new(0.0) },
        };
        // Symmetric prior with cross-correlation between the two tangent axes.
        let prior = SMatrix::<f64, 2, 2>::new(1.0, 0.2, 0.2, 1.0);
        let mut fs = FilterState::<f64, So2xR, 3, 2>::new(state, prior);

        let prior_trace = fs.covariance.trace();

        // h(x) = [z1, x] = [0, 0] at the identity state, so the measurement is
        // the innovation here.
        let measurement = SVector::<f64, 2>::new(0.3, -0.4);
        let r = SMatrix::<f64, 2, 2>::new(0.5, 0.0, 0.0, 0.5);
        fs.correct_measurement(&meas, &measurement, &r);

        // Posterior covariance is symmetric (the update forces it explicitly).
        assert!((fs.covariance[(0, 1)] - fs.covariance[(1, 0)]).abs() < EPS);
        // Variances remain physically valid.
        assert!(fs.covariance[(0, 0)] > 0.0);
        assert!(fs.covariance[(1, 1)] > 0.0);
        // A measurement reduces total uncertainty.
        assert!(fs.covariance.trace() < prior_trace);

        // Both nominal components moved in response to a nonzero innovation, and
        // the rotation component stays on the manifold.
        assert!((fs.state.m1.z.norm() - 1.0).abs() < EPS);
        assert!(fs.state.m2.x[0].abs() > EPS);
    }

    // -------------------------------------------------------------------------
    // project_hard_constraints drives a violated linear constraint to zero.
    // For h(x) = 3x the Newton/Kalman projection is exact in a single step,
    // collapsing the variance along the fully-constrained direction.
    // -------------------------------------------------------------------------
    #[test]
    fn test_project_hard_constraints_real_line_exact() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 1, 1>, 1, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(scale3, scale3_jac));

        let mut fs = FilterState::<f64, RealLine<f64>, 1, 1>::new(
            RealLine { x: SVector::<f64, 1>::new(4.0) },
            SMatrix::<f64, 1, 1>::new(1.0),
        );

        fs.project_hard_constraints(&con, 1e-10, 5);

        // h(x) = 3x = 0  =>  x = 0 ;  P = (1 - (1/3)*3)*1 = 0
        assert!(fs.state.x[0].abs() < 1e-9);
        assert!(fs.covariance[(0, 0)].abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // predict with a hard constraint: the dynamics push the state off the
    // surface, then the in-loop projection pulls it exactly back.
    // -------------------------------------------------------------------------
    #[test]
    fn test_predict_with_hard_constraint_projects() {
        let dynamics: TrajectoryDynamics<f64, NormalDiff<f64, 3, 1>, 1, 1, 1, 1, 3> =
            TrajectoryDynamics::new(NormalDiff::new(lin_vel, lin_vel_jac));
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 1, 1>, 1, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(scale3, scale3_jac));

        let mut fs = FilterState::<f64, RealLine<f64>, 1, 1>::new(
            RealLine { x: SVector::<f64, 1>::new(4.0) },
            SMatrix::<f64, 1, 1>::new(0.5),
        );

        let control = SVector::<f64, 1>::new(5.0);
        let process_noise = SMatrix::<f64, 1, 1>::new(2.0);
        fs.predict_with_constraint(&dynamics, &control, &process_noise, &con, 0.1, Integrator::Euler);

        // Dynamics move x to 4.7; the hard constraint h(x)=3x=0 then projects it
        // exactly back onto the constraint surface.
        assert!(fs.state.x[0].abs() < 1e-9);
        assert!(fs.covariance[(0, 0)].abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // Singularity guard: a rotation-invariant constraint has a zero tangent
    // gradient, so the innovation covariance S is singular, Cholesky fails, and
    // the update is rejected — leaving the state and covariance untouched even
    // though the ambient residual is nonzero.
    // -------------------------------------------------------------------------
    #[test]
    fn test_project_hard_constraints_singular_guard_is_noop() {
        let con: TimeVaryingConstraint<f64, NormalDiff<f64, 2, 1>, 2, 1, 1> =
            TimeVaryingConstraint::new(NormalDiff::new(unit_norm, unit_norm_jac));

        // Off the unit circle: residual = 2^2 + 0 - 1 = 3 (nonzero) ...
        let mut fs = FilterState::<f64, ComplexRotation<f64>, 2, 1>::new(
            ComplexRotation { z: SVector::<f64, 2>::new(2.0, 0.0) },
            SMatrix::<f64, 1, 1>::new(0.5),
        );

        fs.project_hard_constraints(&con, 1e-10, 5);

        // ... yet the projection cannot act on a rotation-invariant constraint,
        // so nothing changes.
        assert_eq!(fs.state.z[0], 2.0);
        assert_eq!(fs.state.z[1], 0.0);
        assert_eq!(fs.covariance[(0, 0)], 0.5);
    }
}
