#![no_std]

pub mod diff;
pub mod manifold;

// `dual` now lives in its own crate; re-export it so `crate::dual::Dual` and
// `manifold::dual::Dual` continue to resolve.
pub use dual;
