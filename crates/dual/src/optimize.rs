#![no_std]

use crate::manifold::Manifold;
use nalgebra::{RealField, SMatrix, SVector};

pub trait CostFunction<
    U: RealField + Copy,
    const RESIDUAL_DIM: usize,
    const TOTAL_PARAMETER_DIM: usize,
>
{
    fn evaluate<T>(&self, parameters: &SVector<T, TOTAL_PARAMETER_DIM>) -> SVector<T, RESIDUAL_DIM>
    where
        T: RealField + Copy;
}

pub trait OptimizationHorizon<
    U: RealField + Copy,
    const STATE_DIM: usize,
    const RESIDUAL_DIM: usize,
>
{
    fn compute_residuals(&self) -> SVector<U, RESIDUAL_DIM>;

    fn linearize(
        &self,
    ) -> (
        SMatrix<U, RESIDUAL_DIM, STATE_DIM>,
        SMatrix<U, RESIDUAL_DIM, RESIDUAL_DIM>,
    );

    fn retract_step(&mut self, delta_x: &SVector<U, STATE_DIM>);
}

pub struct SolverOptions {
    pub max_iterations: usize,
    pub step_tolerance: f64,
}

pub struct SolverSummary<U> {
    pub initial_cost: U,
    pub final_cost: U,
    pub iterations: usize,
}

pub trait StaticNonlinearSolver<
    U: RealField + Copy,
    const STATE_DIM: usize,
    const RESIDUAL_DIM: usize,
>
{
    fn solve<H>(&self, options: &SolverOptions, horizon: &mut H) -> SolverSummary<U>
    where
        H: OptimizationHorizon<U, STATE_DIM, RESIDUAL_DIM>;
}

pub struct StaticLevenbergMarquardt;

impl<U: RealField + Copy, const STATE_DIM: usize, const RESIDUAL_DIM: usize>
    StaticNonlinearSolver<U, STATE_DIM, RESIDUAL_DIM> for StaticLevenbergMarquardt
{
    fn solve<H>(&self, options: &SolverOptions, horizon: &mut H) -> SolverSummary<U>
    where
        H: OptimizationHorizon<U, STATE_DIM, RESIDUAL_DIM>,
    {
        let mut lambda = U::from_f64(1e-3).unwrap();

        // Compile-time sized allocations on the stack!
        let mut best_residuals = horizon.compute_residuals();
        let mut best_cost = best_residuals.norm();
        let initial_cost = best_cost;

        for iter in 0..options.max_iterations {
            // 1. Structural evaluation of Jacobian and Information (Weights) on the stack
            let (jacobian, info) = horizon.linearize();
            let jt = jacobian.transpose();

            // 2. Normal Equation Formulation: H = J^T * W * J
            let mut hessian = &jt * &info * &jacobian;
            let gradient = &jt * &info * &best_residuals;

            // 3. Damping step directly on the diagonal (Levenberg-Marquardt)
            for i in 0..STATE_DIM {
                hessian[(i, i)] += lambda;
            }

            // 4. Solve system with ZERO allocation using nalgebra stack decompositions
            if let Some(cholesky) = hessian.cholesky() {
                let delta_x = cholesky.solve(&(-gradient));

                // 5. Tentative step update using your box_plus primitives
                horizon.retract_step(&delta_x);
                let new_residuals = horizon.compute_residuals();
                let new_cost = new_residuals.norm();

                if new_cost < best_cost {
                    // Step accepted! Cost decreased
                    best_residuals = new_residuals;
                    best_cost = new_cost;
                    lambda /= U::from_f64(10.0).unwrap(); // Expand trust region
                } else {
                    // Step rejected! Revert step by passing negative update
                    horizon.retract_step(&(-delta_x));
                    lambda *= U::from_f64(10.0).unwrap(); // Shrink trust region
                }
            } else {
                break; // Singular matrix, bail out
            }
        }

        SolverSummary {
            initial_cost,
            final_cost: best_cost,
            iterations: options.max_iterations,
        }
    }
}
