//! Stepping-stone demo: a 15-state INS error-state filter built by *hand-nesting*
//! binary `ProductSpace`s over `Rn` and `SO3`, with no `product_manifold!` macro
//! yet. The point is to prove `FilterState` actually drives a product manifold
//! end to end (predict + measurement update) — and to show, in the flesh, the
//! deep field-access / const-generic bookkeeping the macro will later remove.
//!
//! State:  p∈R³, v∈R³, q∈SO(3), b_a∈R³, b_g∈R³  (ambient 16, tangent 15).
//! The dynamics here are a deliberately simple kinematic model (ṗ=v, v̇=a+w,
//! θ̇=ω+w, biases constant) — enough to exercise every tangent block and the
//! SO(3) retraction, NOT a real strapdown mechanization (that's the ESKF work).
//!
//! Run:  cargo run --release -p filter --example ins_15state

use filter::{FilterState, Integrator};
use manifold::diff::{AutoDiff, DiffFn};
use manifold::manifold::{ProductSpace, Rn, SO3, TimeVaryingConstraint, TrajectoryDynamics};
use nalgebra::{RealField, SMatrix, SVector, Vector3};

// --- the 15-state product, assembled as a left-folded binary tree ------------
// Each level must carry its children's dims AND their sums (stable const
// generics can't compute the totals) — this is the verbosity the macro kills.
type R3 = Rn<f64, 3>;
type Pv = ProductSpace<R3, R3, 3, 3, 3, 3, 6, 6>; //          p × v
type PvQ = ProductSpace<Pv, SO3<f64>, 6, 6, 4, 3, 10, 9>; //  (p×v) × q
type PvQBa = ProductSpace<PvQ, R3, 10, 9, 3, 3, 13, 12>; //   × b_a
type Ins = ProductSpace<PvQBa, R3, 13, 12, 3, 3, 16, 15>; //  × b_g   (16 / 15)

fn make_state() -> Ins {
    let p = R3 { x: Vector3::zeros() };
    let v = R3 { x: Vector3::new(1.0, 0.0, 0.0) }; // 1 m/s along +x
    let q = SO3::identity();
    let ba = R3 { x: Vector3::zeros() };
    let bg = R3 { x: Vector3::zeros() };

    // Typed bindings pin each level's const generics (inference can't).
    let pv: Pv = ProductSpace { m1: p, m2: v };
    let pvq: PvQ = ProductSpace { m1: pv, m2: q };
    let pvqba: PvQBa = ProductSpace { m1: pvq, m2: ba };
    ProductSpace { m1: pvqba, m2: bg }
}

// Block-diagonal P₀ from per-block 1σ: p, v, θ, b_a, b_g.
fn initial_covariance() -> SMatrix<f64, 15, 15> {
    let mut p0 = SMatrix::<f64, 15, 15>::zeros();
    let vars = [1.0_f64, 0.1, 0.02, 0.01, 1e-4].map(|s| s * s);
    for (block, &var) in vars.iter().enumerate() {
        for i in 0..3 {
            p0[(3 * block + i, 3 * block + i)] = var;
        }
    }
    p0
}

// --- dynamics: tangent velocity (15) from packed [ambient(16) | u(6) | w(6)] (28) ---
// u = [a(3), ω(3)],  w = [w_a(3), w_g(3)].  AutoDiff supplies F = ∂(velocity).
struct InsDynamics;
impl DiffFn<28, 15> for InsDynamics {
    fn eval<S: RealField + Copy>(&self, x: &SVector<S, 28>, y: &mut SVector<S, 15>) {
        // ṗ = v   (ambient velocity lives at indices 3..6)
        y[0] = x[3];
        y[1] = x[4];
        y[2] = x[5];
        // v̇ = a + w_a
        y[3] = x[16] + x[22];
        y[4] = x[17] + x[23];
        y[5] = x[18] + x[24];
        // θ̇ = ω + w_g
        y[6] = x[19] + x[25];
        y[7] = x[20] + x[26];
        y[8] = x[21] + x[27];
        // biases: constant (derivative 0)
        for i in 9..15 {
            y[i] = S::zero();
        }
    }
}

// --- measurement: position = ambient[0..3] -----------------------------------
struct PosMeasurement;
impl DiffFn<16, 3> for PosMeasurement {
    fn eval<S: RealField + Copy>(&self, x: &SVector<S, 16>, y: &mut SVector<S, 3>) {
        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
    }
}

// The deep accessors are the other half of the macro's motivation.
fn position(s: &Ins) -> Vector3<f64> {
    s.m1.m1.m1.m1.x
}
fn velocity(s: &Ins) -> Vector3<f64> {
    s.m1.m1.m1.m2.x
}
fn yaw(s: &Ins) -> f64 {
    s.m1.m1.m2.uq.euler_angles().2
}

fn main() {
    let mut fs = FilterState::<f64, Ins, 16, 15>::new(make_state(), initial_covariance());

    let dynamics: TrajectoryDynamics<f64, AutoDiff<InsDynamics>, 16, 15, 6, 6, 28> =
        TrajectoryDynamics::new(AutoDiff::new(InsDynamics));

    // No commanded accel; constant 0.1 rad/s yaw rate.
    let control = SVector::<f64, 6>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]);
    let process_noise = SMatrix::<f64, 6, 6>::identity() * 1e-4;
    let dt = 0.01;

    println!("=== 15-state INS on a hand-nested product manifold ===");
    println!(
        "t=0.00s  pos={:.3?}  vel={:.3?}  yaw={:.4}  trace(P)={:.4}",
        position(&fs.state),
        velocity(&fs.state),
        yaw(&fs.state),
        fs.covariance.trace(),
    );

    // Dead-reckon for 1 s (100 steps): position grows, covariance inflates.
    for _ in 0..100 {
        fs.predict(&dynamics, &control, &process_noise, dt, Integrator::Euler);
    }
    println!(
        "t=1.00s  pos={:.3?}  vel={:.3?}  yaw={:.4}  trace(P)={:.4}   (after predict)",
        position(&fs.state),
        velocity(&fs.state),
        yaw(&fs.state),
        fs.covariance.trace(),
    );

    // Fold in a position fix near truth — covariance must drop.
    let meas: TimeVaryingConstraint<f64, AutoDiff<PosMeasurement>, 16, 15, 3> =
        TimeVaryingConstraint::new(AutoDiff::new(PosMeasurement));
    // A position fix offset slightly from the current estimate.
    let measured = position(&fs.state) + Vector3::new(0.05, -0.02, 0.0);
    let r = SMatrix::<f64, 3, 3>::identity() * 0.25; // (0.5 m)² per axis

    let trace_before = fs.covariance.trace();
    fs.correct_measurement(&meas, &measured, &r);
    println!(
        "t=1.00s  pos={:.3?}  vel={:.3?}  yaw={:.4}  trace(P)={:.4}   (after position fix)",
        position(&fs.state),
        velocity(&fs.state),
        yaw(&fs.state),
        fs.covariance.trace(),
    );

    assert!(
        fs.covariance.trace() < trace_before,
        "a measurement must reduce total uncertainty"
    );
    assert!((fs.state.m1.m1.m2.uq.norm() - 1.0).abs() < 1e-9, "quaternion left the manifold");
    println!("\nOK: filter drives the 15-state product manifold (predict + update consistent).");
}
