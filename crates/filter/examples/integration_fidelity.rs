//! Integration-order fidelity: how well the filter's first-order manifold
//! integrator tracks the true continuous dynamics, and whether the residual
//! injects enough error to make the filter inconsistent.
//!
//! `mc_validation` generates truth with the *same* first-order Euler step the
//! filter uses, so it structurally cannot see integration error. Here truth is a
//! high-fidelity reference (`reference_step`: RK4 translation + exponential-map
//! attitude over 100 substeps) while the filter still runs first-order Euler.
//!
//! Part 1 — deterministic convergence: with clean inputs the global error of
//! first-order Euler vs the reference halves as dt halves (order ≈ 1). Note the
//! manifold retract integrates the *rotation* exactly for constant ω, so the
//! attitude error (only from ω varying within a step) is far below the
//! translational error.
//!
//! Part 2 — NEES against high-fidelity truth, swept over dt, for Euler vs midpoint
//! and a zero-order-hold vs delta-form IMU. Midpoint fixes STEADY dynamics outright
//! (coordinated turn in-band even at 200 ms). The AGGRESSIVE profile needs more, and
//! Part 3 decomposes its NEES by state to find why: two independent residuals — a
//! VELOCITY error (mean integration order, fixed by midpoint) and an ATTITUDE error
//! (the gyro sampled once per step, fixed by an interval-averaged / delta-angle IMU,
//! ∫ω — i.e. within-step ω-variation, not coning). Midpoint + delta-form IMU lands
//! every block on ~3 ⇒ fully consistent. The covariance propagation was never the
//! limitation; the wins were integrating the mean and the gyro across the step.
//!
//! Run:  cargo run --release -p filter --example integration_fidelity

#[path = "support/ins.rs"]
mod ins;
use ins::*;

use filter::Integrator;
use manifold::manifold::{Manifold, SO3};
use nalgebra::{Matrix3, SVector, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;

const Z95: f64 = 1.959964;

// Wilson–Hilferty χ²_k quantile (see mc_validation).
fn chi2_quantile(k: f64, z: f64) -> f64 {
    let h = 2.0 / (9.0 * k);
    k * (1.0 - h + z * h.sqrt()).powi(3)
}

// ---- Part 1: deterministic integration error of the mechanization vs reference.

// One nominal-state advance with the chosen integrator (mirrors predict's mean,
// with the control held across the step — the filter only has one IMU sample).
fn mean_step(s: &Ins, dyn_engine: &Dynamics, u: &SVector<f64, 6>, dt: f64, integ: Integrator) -> Ins {
    let zero = SVector::<f64, 12>::zeros();
    match integ {
        Integrator::Euler => integrate(s, dyn_engine, u, dt),
        Integrator::Midpoint => {
            let k1 = dyn_engine.evaluate_velocity(&s.to_ambient(), u, &zero);
            let mid = s.retract(&Ins::vector_to_tangent(&(k1 * (dt * 0.5))));
            let k2 = dyn_engine.evaluate_velocity(&mid.to_ambient(), u, &zero);
            s.retract(&Ins::vector_to_tangent(&(k2 * dt)))
        }
    }
}

fn integration_error(
    inputs: fn(f64, Vector3<f64>) -> (Vector3<f64>, Vector3<f64>),
    v0: Vector3<f64>,
    dt: f64,
    total_s: f64,
    integ: Integrator,
) -> (f64, f64, f64) {
    let dynamics = dynamics();
    let g = g_vec();
    let init = state(Vector3::zeros(), v0, SO3::identity(), Vector3::zeros(), Vector3::zeros());
    let mut est = init.clone();
    let mut refr = init;
    let n = (total_s / dt).round() as usize;
    for k in 0..n {
        let t = k as f64 * dt;
        // ZOH IMU: true specific force / rate from the *reference* at step start.
        let (a_nav, omega) = inputs(t, vel(&refr));
        let f_b = att(&refr).uq.inverse_transform_vector(&(a_nav - g));
        est = mean_step(&est, &dynamics, &control(f_b, omega), dt, integ);
        refr = reference_step(&refr, t, dt, inputs);
    }
    (
        (pos(&est) - pos(&refr)).norm(),
        (vel(&est) - vel(&refr)).norm(),
        att(&est).local_lift(att(&refr)).norm(),
    )
}

fn part1() {
    println!("--- Part 1: deterministic mean error vs high-fidelity reference (no noise) ---");
    println!("  aggressive 3-D profile, accumulated error at t = 20 s\n");
    println!(
        "  {:>7} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9}",
        "dt[ms]", "E pos[m]", "E vel", "E att", "M pos[m]", "M vel", "M att"
    );
    let all = profiles();
    let prof = &all[3]; // aggressive 3-D
    let total = 20.0;
    for &dt in &[0.04, 0.02, 0.01, 0.005] {
        let (ep, ev, ea) = integration_error(prof.inputs, prof.v0, dt, total, Integrator::Euler);
        let (mp, mv, ma) = integration_error(prof.inputs, prof.v0, dt, total, Integrator::Midpoint);
        println!(
            "  {:>7.0} | {:>9.2e} {:>9.2e} {:>9.2e} | {:>9.2e} {:>9.2e} {:>9.2e}",
            dt * 1e3, ep, ev, ea, mp, mv, ma
        );
    }
    println!("  (E = Euler, M = Midpoint; pos[m], vel[m/s], att[rad] — which state moves?)\n");
}

// ---- Part 2: filter NEES with high-fidelity truth, swept over the step size.

fn nees_vs_reference(prof: &Profile, dt: f64, runs: usize, total_s: f64, integ: Integrator, delta_imu: bool) -> f64 {
    let dynamics = dynamics();
    let meas = position_fix();
    let q_noise = process_noise(dt);
    let r_gps = Matrix3::identity() * SIGMA_GPS * SIGMA_GPS;
    let g = g_vec();
    let n_steps = (total_s / dt).round() as usize;
    let fix_every = (0.5 / dt).round().max(1.0) as usize;
    let burn = (5.0 / dt).round() as usize;

    let mut nees_sum = 0.0;
    let mut count = 0.0;
    for run in 0..runs {
        let mut rng = StdRng::seed_from_u64(run as u64);
        let mut imu = SyntheticImu::new(dt, 10_000 + run as u64);
        let mut truth = state(Vector3::zeros(), prof.v0, SO3::identity(), Vector3::zeros(), Vector3::zeros());
        let mut filter = perturbed_filter(&truth, &mut rng);

        for k in 1..=n_steps {
            let t = (k - 1) as f64 * dt;
            // Advance truth high-fidelity; the filter consumes either a ZOH
            // sample (start of interval) or the interval-averaged delta-form IMU.
            let (next, f_b_avg, omega_avg) = reference_step_with_imu(&truth, t, dt, prof.inputs);
            let (f_b, omega) = if delta_imu {
                (f_b_avg, omega_avg)
            } else {
                let (a_nav, w) = (prof.inputs)(t, vel(&truth));
                (att(&truth).uq.inverse_transform_vector(&(a_nav - g)), w)
            };
            let (accel, gyro) = imu.sample(f_b, omega);

            filter.predict(&dynamics, &control(accel, gyro), &q_noise, dt, integ);
            truth = next;

            if k % fix_every == 0 {
                let gps = pos(&truth) + SIGMA_GPS * randn3(&mut rng);
                filter.correct_measurement(&meas, &gps, &r_gps);
                if k >= burn {
                    let e = nees_error(&truth, &filter.state, &imu);
                    let pinv = filter.covariance.try_inverse().expect("P singular");
                    nees_sum += e.dot(&(pinv * e));
                    count += 1.0;
                }
            }
        }
    }
    nees_sum / count
}

// Per-block marginal normalized error e_bᵀ (P_bb)⁻¹ e_b for [p, v, θ, bₐ, b_g].
// E[·] = 3 per 3-dof block if consistent; the largest block is the culprit.
fn block_nees(prof: &Profile, dt: f64, runs: usize, total_s: f64, integ: Integrator, delta_imu: bool) -> [f64; 5] {
    let dynamics = dynamics();
    let meas = position_fix();
    let q_noise = process_noise(dt);
    let r_gps = Matrix3::identity() * SIGMA_GPS * SIGMA_GPS;
    let g = g_vec();
    let n_steps = (total_s / dt).round() as usize;
    let fix_every = (0.5 / dt).round().max(1.0) as usize;
    let burn = (5.0 / dt).round() as usize;

    let mut sums = [0.0f64; 5];
    let mut count = 0.0;
    for run in 0..runs {
        let mut rng = StdRng::seed_from_u64(run as u64);
        let mut imu = SyntheticImu::new(dt, 10_000 + run as u64);
        let mut truth = state(Vector3::zeros(), prof.v0, SO3::identity(), Vector3::zeros(), Vector3::zeros());
        let mut filter = perturbed_filter(&truth, &mut rng);
        for k in 1..=n_steps {
            let t = (k - 1) as f64 * dt;
            // Advance truth high-fidelity; optionally feed the filter the
            // interval-averaged (delta-angle / delta-v) IMU instead of a ZOH sample.
            let (next, f_b_avg, omega_avg) = reference_step_with_imu(&truth, t, dt, prof.inputs);
            let (f_b, omega) = if delta_imu {
                (f_b_avg, omega_avg)
            } else {
                let (a_nav, w) = (prof.inputs)(t, vel(&truth));
                (att(&truth).uq.inverse_transform_vector(&(a_nav - g)), w)
            };
            let (accel, gyro) = imu.sample(f_b, omega);
            filter.predict(&dynamics, &control(accel, gyro), &q_noise, dt, integ);
            truth = next;
            if k % fix_every == 0 {
                let gps = pos(&truth) + SIGMA_GPS * randn3(&mut rng);
                filter.correct_measurement(&meas, &gps, &r_gps);
                if k >= burn {
                    let e = nees_error(&truth, &filter.state, &imu);
                    for b in 0..5 {
                        let eb = e.fixed_rows::<3>(3 * b).into_owned();
                        let pbb = filter.covariance.fixed_view::<3, 3>(3 * b, 3 * b).into_owned();
                        sums[b] += eb.dot(&(pbb.try_inverse().expect("P_bb singular") * eb));
                    }
                    count += 1.0;
                }
            }
        }
    }
    sums.map(|s| s / count)
}

fn main() {
    println!("=== Integration-order fidelity — first-order vs midpoint mechanization ===\n");
    part1();

    println!("--- Part 2: filter consistency vs high-fidelity truth (NEES) ---");
    let runs = 40usize;
    let total = 30.0;
    let k = 15.0 * runs as f64;
    let (lo, hi) = (chi2_quantile(k, -Z95) / runs as f64, chi2_quantile(k, Z95) / runs as f64);
    println!("  {runs} runs × {total:.0}s, truth = RK4/exp-map reference");
    println!("  NEES 95% band (15 dof): [{lo:.2}, {hi:.2}]   E[η] = 15\n");
    println!("  ZOH = start-of-interval IMU sample;  delta = interval-averaged delta-v/-angle");
    println!("  {:<18} {:>6}  {:>11}  {:>11}  {:>11}", "profile", "dt[ms]", "Euler/ZOH", "Mid/ZOH", "Mid/delta");

    let tag = |n: f64| if n >= lo && n <= hi { format!("{n:.1} ok") } else { format!("{n:.1} HI") };
    let all = profiles();
    for prof in [&all[2], &all[3]] {
        for &dt in &[0.02, 0.05, 0.1, 0.2] {
            let ez = nees_vs_reference(prof, dt, runs, total, Integrator::Euler, false);
            let mz = nees_vs_reference(prof, dt, runs, total, Integrator::Midpoint, false);
            let md = nees_vs_reference(prof, dt, runs, total, Integrator::Midpoint, true);
            println!("  {:<18} {:>6.0}  {:>11}  {:>11}  {:>11}", prof.name, dt * 1e3, tag(ez), tag(mz), tag(md));
        }
    }
    println!();
    println!("--- Part 3: which state block is inconsistent? (aggressive 3-D @ 20ms) ---");
    println!("  {:<12} {:>7} {:>7} {:>7} {:>7} {:>7}", "case", "pos", "vel", "att", "ba", "bg");
    let agg = &profiles()[3];
    let row = |label: &str, b: [f64; 5]| {
        println!("  {:<12} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}", label, b[0], b[1], b[2], b[3], b[4]);
    };
    row("Euler ZOH", block_nees(agg, 0.02, runs, total, Integrator::Euler, false));
    row("Mid ZOH", block_nees(agg, 0.02, runs, total, Integrator::Midpoint, false));
    row("Mid delta", block_nees(agg, 0.02, runs, total, Integrator::Midpoint, true));

    println!("\n  The aggressive residual is two independent errors with two distinct fixes:");
    println!("    • velocity — the mean integration order: midpoint (in predict) fixes it.");
    println!("    • attitude — the gyro sampled once per step: the interval-averaged");
    println!("      (delta-angle) IMU fixes it (∫ω, first-order — so it is ω-variation,");
    println!("      not coning). This is an IMU-input concern, not a filter change.");
    println!("  Midpoint + delta-form IMU lands every block on ~3 (Part 3) ⇒ fully consistent.");
    println!("  The covariance propagation was never the limitation: the error-state F is");
    println!("  fine; what mattered was integrating the mean and the gyro across the step.");
}
