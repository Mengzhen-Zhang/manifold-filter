//! Closing the ESKF gap: a 15-state INS error-state filter on the manifold
//! framework, validated by NEES consistency over a Monte-Carlo ensemble.
//!
//! Unlike `ins_15state` (simplified kinematics), this uses the real strapdown
//! mechanization — `v̇ = R(q)·(a − b_a) + g`, Gauss–Markov biases — shared from
//! `support/ins.rs`. **AutoDiff derives the 15×15 F** that nav-eskf hand-codes;
//! the predict step's full attitude adjoint comes from the `parallel_transport`
//! ±δ composition.
//!
//! A coordinated-turn truth trajectory is flown; a synthetic IMU corrupts the
//! inertial stream with the *same* white-noise densities and Gauss–Markov bias
//! the filter assumes; the filter dead-reckons and folds in a position fix every
//! 0.5 s. The 15-state NEES, averaged over seeds, should sit on E[η] = 15 — the
//! filter knows exactly how wrong it is. For the broader multi-trajectory
//! consistency suite (NEES + NIS, χ² bands), see `mc_validation`.
//!
//! Run:  cargo run --release -p filter --example eskf_nees

#[path = "support/ins.rs"]
mod ins;
use ins::*;

use filter::Integrator;
use manifold::manifold::SO3;
use nalgebra::{SMatrix, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ----- Trajectory: a level coordinated turn -----
const SPEED: f64 = 10.0; // m/s along body-x
const TURN_RATE: f64 = 0.2; // rad/s yaw  ⇒  radius 50 m

fn main() {
    let runs = 40usize;
    let dt = 0.02; // 50 Hz
    let n_steps = 1500; // 30 s
    let fix_every = 25; // 0.5 s
    let burn_in = 5.0; // s

    let dynamics = dynamics();
    let meas = position_fix();
    let q_noise = process_noise(dt);
    let r_gps = SMatrix::<f64, 3, 3>::identity() * SIGMA_GPS * SIGMA_GPS;
    let omega = Vector3::new(0.0, 0.0, TURN_RATE);
    let g = g_vec();

    let n_samples = n_steps / fix_every;
    let mut nees_sum = vec![0.0f64; n_samples];

    for run in 0..runs {
        let mut rng = StdRng::seed_from_u64(run as u64);
        let mut imu = SyntheticImu::new(dt, 10_000 + run as u64);

        // Truth: on the circle, heading +x, level.
        let mut truth = state(
            Vector3::zeros(),
            Vector3::new(SPEED, 0.0, 0.0),
            SO3::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
        );
        let mut filter = perturbed_filter(&truth, &mut rng);

        let mut si = 0;
        for k in 1..=n_steps {
            // True specific force / rate for a constant-speed level turn.
            let r = att(&truth).uq;
            let a_nav = omega.cross(&vel(&truth));
            let f_b = r.inverse_transform_vector(&(a_nav - g));

            let (accel, gyro) = imu.sample(f_b, omega);
            filter.predict(&dynamics, &control(accel, gyro), &q_noise, dt, Integrator::Euler);
            truth = integrate(&truth, &dynamics, &control(f_b, omega), dt);

            if k % fix_every == 0 {
                let gps = pos(&truth) + SIGMA_GPS * randn3(&mut rng);
                filter.correct_measurement(&meas, &gps, &r_gps);

                let e = nees_error(&truth, &filter.state, &imu);
                let pinv = filter.covariance.try_inverse().expect("P singular");
                nees_sum[si] += e.dot(&(pinv * e));
                si += 1;
            }
        }
    }

    // Report: NEES averaged over seeds, then over post-burn-in time.
    let mean_nees: Vec<f64> = nees_sum.iter().map(|s| s / runs as f64).collect();
    let kept: Vec<f64> = mean_nees
        .iter()
        .enumerate()
        .filter(|(i, _)| (((i + 1) * fix_every) as f64 * dt) >= burn_in)
        .map(|(_, &v)| v)
        .collect();
    let mean = kept.iter().sum::<f64>() / kept.len() as f64;

    println!("=== INS ESKF on the manifold framework — NEES validation ===");
    println!(
        "  {} runs, {:.0} s @ {:.0} Hz, fix every {:.1} s",
        runs,
        n_steps as f64 * dt,
        1.0 / dt,
        fix_every as f64 * dt,
    );
    println!("  mean NEES (post {burn_in}s) = {mean:.2}   (expect 15)");
    println!("  per-sample (last) NEES = {:.2}", mean_nees[n_samples - 1]);
    assert!(
        (mean - 15.0).abs() < 3.0,
        "filter inconsistent: mean NEES {mean:.2} not near 15"
    );
    println!("\nOK: the manifold-framework ESKF is consistent (NEES ≈ 15).");
}
