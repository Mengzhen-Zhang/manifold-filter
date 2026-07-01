#![no_std]
//! Forward-mode dual numbers carrying a value (`re`) and `N` partial
//! derivatives (`eps`). `Dual<T, N>` implements `nalgebra::RealField`, so it can
//! be used as the scalar type in any `RealField`-generic computation to obtain
//! exact derivatives by automatic differentiation.
//!
//! With `N` tangent components the derivative seeds propagate as a batch: seed
//! input `i` with the unit direction `eᵢ` ([`Dual::seed`]) and a single
//! evaluation of `f: ℝᴺ → ℝᴹ` yields the whole `M×N` Jacobian ([`jacobian`]).
//! `N` defaults to 1, recovering an ordinary scalar dual number.

pub mod manifold;
pub mod optimize;

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
pub use dual_derive::Manifold;
use nalgebra::{ComplexField, Field, RealField, SMatrix, SVector, Scalar, SimdValue};
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{SubsetOf, SupersetOf};

#[derive(Clone, Copy)]
pub struct Dual<T, const N: usize = 1> {
    /// The value (primal) part.
    pub re: T,
    /// The `N` accumulated partial derivatives (tangent part).
    pub eps: SVector<T, N>,
}

// `SVector: Debug` needs `T: Scalar`, which the derive macro would not add.
impl<T: Scalar, const N: usize> fmt::Debug for Dual<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dual")
            .field("re", &self.re)
            .field("eps", &self.eps)
            .finish()
    }
}

// Ordering and equality act on the value part only (consistent with how the
// selection methods — max/min/clamp — decide).
impl<T: PartialEq, const N: usize> PartialEq for Dual<T, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}

impl<T: PartialOrd, const N: usize> PartialOrd for Dual<T, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<T, const N: usize> Dual<T, N> {
    /// Builds a dual directly from a value and an explicit tangent vector.
    #[inline]
    pub const fn new(re: T, eps: SVector<T, N>) -> Self {
        Self { re, eps }
    }
}

impl<T: RealField + Copy, const N: usize> Dual<T, N> {
    /// A constant: value `re`, zero derivative.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, SVector::zeros())
    }

    /// Alias for [`Dual::from_re`].
    #[inline]
    pub fn constant(re: T) -> Self {
        Self::from_re(re)
    }

    /// Seeds the `i`-th independent variable: value `re`, tangent direction `eᵢ`.
    #[inline]
    pub fn seed(re: T, i: usize) -> Self {
        let mut eps = SVector::<T, N>::zeros();
        eps[i] = T::one();
        Self::new(re, eps)
    }

    /// Constructs a dual, debug-asserting that the tangent (derivative) is
    /// finite. A non-finite tangent means the function was differentiated at a
    /// singular point — `sqrt'(0)` (the ‖·‖ origin), `1/x` at `x = 0`, an
    /// `atan2` at the origin — where the derivative genuinely does not exist.
    ///
    /// Policy: fail loudly at the source op in debug builds rather than
    /// silently fabricate a value (such as a 0 sub-gradient). In release the
    /// assertion compiles out and the non-finite tangent propagates faithfully,
    /// keeping the type `no_std`/embedded-safe. Primal-value finiteness is the
    /// caller's concern, as with any `f64` computation, so only the tangent is
    /// checked here.
    #[inline]
    fn checked(re: T, eps: SVector<T, N>) -> Self {
        debug_assert!(
            eps.iter().all(|e| e.is_finite()),
            "Dual: non-finite derivative — differentiated at a singular point",
        );
        Self::new(re, eps)
    }

    // Applies the chain rule for a unary function: value `val`, local slope `deriv`.
    #[inline]
    fn chain(self, val: T, deriv: T) -> Self {
        Self::checked(val, self.eps * deriv)
    }

    // Power with a dual exponent: d/dt x^n = x^n (n' ln x + n x'/x).
    #[inline]
    fn powf_dual(self, n: Self) -> Self {
        let v = self.re.powf(n.re);
        let eps = (n.eps * self.re.ln() + self.eps * (n.re / self.re)) * v;
        Self::checked(v, eps)
    }

    #[inline]
    pub fn recip(self) -> Self {
        let inv = T::one() / self.re;
        self.chain(inv, -(inv * inv))
    }

    #[inline]
    pub fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }

    #[inline]
    pub fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        self.chain(s, T::one() / (s + s))
    }

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    #[inline]
    pub fn ln(self) -> Self {
        self.chain(self.re.ln(), T::one() / self.re)
    }

    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let nt = T::from_i32(n).expect("i32 -> T conversion failed");
        self.chain(self.re.powi(n), nt * self.re.powi(n - 1))
    }

    #[inline]
    pub fn powf(self, p: T) -> Self {
        self.chain(self.re.powf(p), p * self.re.powf(p - T::one()))
    }

    #[inline]
    pub fn tan(self) -> Self {
        let c = self.re.cos();
        self.chain(self.re.tan(), T::one() / (c * c))
    }

    #[inline]
    pub fn asin(self) -> Self {
        let d = T::one() / (T::one() - self.re * self.re).sqrt();
        self.chain(self.re.asin(), d)
    }

    #[inline]
    pub fn acos(self) -> Self {
        let d = -(T::one() / (T::one() - self.re * self.re).sqrt());
        self.chain(self.re.acos(), d)
    }

    #[inline]
    pub fn atan(self) -> Self {
        let d = T::one() / (T::one() + self.re * self.re);
        self.chain(self.re.atan(), d)
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let re = (self.re * self.re + other.re * other.re).sqrt();
        let eps = (self.eps * self.re + other.eps * other.re) / re;
        Self::checked(re, eps)
    }

    #[inline]
    pub fn abs(&self) -> Self {
        if self.re >= T::zero() {
            *self
        } else {
            -*self
        }
    }

    #[inline]
    pub fn tanh(self) -> Self {
        let t = self.re.tanh();
        self.chain(t, T::one() - t * t)
    }

    #[inline]
    pub fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }

    #[inline]
    pub fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operators (dual ⊗ dual)
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> Add for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.eps + rhs.eps)
    }
}

impl<T: RealField + Copy, const N: usize> Sub for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.eps - rhs.eps)
    }
}

impl<T: RealField + Copy, const N: usize> Mul for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (u*v)' = u'*v + u*v'
        Self::new(self.re * rhs.re, self.eps * rhs.re + rhs.eps * self.re)
    }
}

impl<T: RealField + Copy, const N: usize> Div for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (u/v)' = (u'*v - u*v') / v^2
        let denom = rhs.re * rhs.re;
        Self::checked(
            self.re / rhs.re,
            (self.eps * rhs.re - rhs.eps * self.re) / denom,
        )
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operators (dual ⊗ scalar, dual on the left)
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> Mul<T> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.eps * rhs)
    }
}

impl<T: RealField + Copy, const N: usize> Div<T> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::checked(self.re / rhs, self.eps / rhs)
    }
}

impl<T: RealField + Copy, const N: usize> Add<T> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.re + rhs, self.eps)
    }
}

impl<T: RealField + Copy, const N: usize> Sub<T> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::new(self.re - rhs, self.eps)
    }
}

impl<T: RealField + Copy, const N: usize> Neg for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.eps)
    }
}

impl<T: RealField + Copy, const N: usize> Rem for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // Only exists to satisfy `Num: NumOps`; the modulus boundary is
        // non-differentiable and never sits on an AD path.
        Self::new(self.re % rhs.re, self.eps)
    }
}

// ---------------------------------------------------------------------------
// Scalar-on-the-left operators (concrete scalar types, per the orphan rule)
// ---------------------------------------------------------------------------
impl<const N: usize> Mul<Dual<f64, N>> for f64 {
    type Output = Dual<f64, N>;
    #[inline]
    fn mul(self, rhs: Dual<f64, N>) -> Self::Output {
        Dual::new(self * rhs.re, rhs.eps * self)
    }
}

impl<const N: usize> Mul<Dual<f32, N>> for f32 {
    type Output = Dual<f32, N>;
    #[inline]
    fn mul(self, rhs: Dual<f32, N>) -> Self::Output {
        Dual::new(self * rhs.re, rhs.eps * self)
    }
}

impl<const N: usize> Add<Dual<f64, N>> for f64 {
    type Output = Dual<f64, N>;
    #[inline]
    fn add(self, rhs: Dual<f64, N>) -> Self::Output {
        Dual::new(self + rhs.re, rhs.eps)
    }
}

impl<const N: usize> Add<Dual<f32, N>> for f32 {
    type Output = Dual<f32, N>;
    #[inline]
    fn add(self, rhs: Dual<f32, N>) -> Self::Output {
        Dual::new(self + rhs.re, rhs.eps)
    }
}

impl<const N: usize> Sub<Dual<f64, N>> for f64 {
    type Output = Dual<f64, N>;
    #[inline]
    fn sub(self, rhs: Dual<f64, N>) -> Self::Output {
        // d/dt (c - x) = -x'
        Dual::new(self - rhs.re, -rhs.eps)
    }
}

impl<const N: usize> Sub<Dual<f32, N>> for f32 {
    type Output = Dual<f32, N>;
    #[inline]
    fn sub(self, rhs: Dual<f32, N>) -> Self::Output {
        Dual::new(self - rhs.re, -rhs.eps)
    }
}

impl<const N: usize> Div<Dual<f64, N>> for f64 {
    type Output = Dual<f64, N>;
    #[inline]
    fn div(self, rhs: Dual<f64, N>) -> Self::Output {
        // d/dt (c / x) = -c x' / x^2
        let denom = rhs.re * rhs.re;
        Dual::checked(self / rhs.re, rhs.eps * (-self / denom))
    }
}

impl<const N: usize> Div<Dual<f32, N>> for f32 {
    type Output = Dual<f32, N>;
    #[inline]
    fn div(self, rhs: Dual<f32, N>) -> Self::Output {
        let denom = rhs.re * rhs.re;
        Dual::checked(self / rhs.re, rhs.eps * (-self / denom))
    }
}

// ---------------------------------------------------------------------------
// Compound assignment
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> AddAssign for Dual<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: RealField + Copy, const N: usize> SubAssign for Dual<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: RealField + Copy, const N: usize> MulAssign for Dual<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs; // product rule, via Mul
    }
}

impl<T: RealField + Copy, const N: usize> DivAssign for Dual<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs; // quotient rule, via Div
    }
}

impl<T: RealField + Copy, const N: usize> RemAssign for Dual<T, N> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<T: RealField + Copy, const N: usize> AddAssign<T> for Dual<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T: RealField + Copy, const N: usize> SubAssign<T> for Dual<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T: RealField + Copy, const N: usize> MulAssign<T> for Dual<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: RealField + Copy, const N: usize> DivAssign<T> for Dual<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

// ---------------------------------------------------------------------------
// num-traits scalar layer
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> Zero for Dual<T, N> {
    #[inline]
    fn zero() -> Self {
        Self::from_re(T::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps == SVector::<T, N>::zeros()
    }
}

impl<T: RealField + Copy, const N: usize> One for Dual<T, N> {
    #[inline]
    fn one() -> Self {
        Self::from_re(T::one())
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.eps == SVector::<T, N>::zeros()
    }
}

impl<T: RealField + Copy, const N: usize> Num for Dual<T, N> {
    type FromStrRadixErr = ();
    #[inline]
    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err(()) // never invoked by linear-algebra paths
    }
}

impl<T: RealField + Copy, const N: usize> Signed for Dual<T, N> {
    // |x| = x for x >= 0 else -x; Neg already carries the matching signum*eps.
    #[inline]
    fn abs(&self) -> Self {
        if self.re >= T::zero() {
            *self
        } else {
            -*self
        }
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.re <= other.re {
            Self::zero()
        } else {
            *self - *other
        }
    }

    // signum is piecewise constant => derivative 0.
    #[inline]
    fn signum(&self) -> Self {
        let v = if self.re > T::zero() {
            T::one()
        } else if self.re < T::zero() {
            -T::one()
        } else {
            T::zero()
        };
        Self::from_re(v)
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.re > T::zero()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.re < T::zero()
    }
}

impl<T: RealField + Copy, const N: usize> FromPrimitive for Dual<T, N> {
    // Primitive constants have zero derivative.
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Self::from_re)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Self::from_re)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(Self::from_re)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        T::from_f32(n).map(Self::from_re)
    }
}

impl<T: RealField + Copy, const N: usize> fmt::Display for Dual<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}\u{3b5}", self.re, self.eps.transpose())
    }
}

// ---------------------------------------------------------------------------
// simba: a Dual behaves as a SIMD scalar of width 1 (no vectorization).
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> SimdValue for Dual<T, N> {
    const LANES: usize = 1;
    type Element = Self;
    type SimdBool = bool;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        val
    }
    #[inline]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }
    #[inline]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val;
    }
    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

// ---------------------------------------------------------------------------
// simba subset/superset: a Dual embeds the reals (eps = 0); the real <-> f32/f64
// conversion is delegated to the inner scalar T.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> SubsetOf<Self> for Dual<T, N> {
    #[inline]
    fn to_superset(&self) -> Self {
        *self
    }
    #[inline]
    fn from_superset_unchecked(element: &Self) -> Self {
        *element
    }
    #[inline]
    fn is_in_subset(_element: &Self) -> bool {
        true
    }
}

impl<T: RealField + Copy, const N: usize> SupersetOf<f64> for Dual<T, N> {
    #[inline]
    fn is_in_subset(&self) -> bool {
        <T as SupersetOf<f64>>::is_in_subset(&self.re)
    }
    #[inline]
    fn to_subset_unchecked(&self) -> f64 {
        <T as SupersetOf<f64>>::to_subset_unchecked(&self.re)
    }
    #[inline]
    fn from_subset(element: &f64) -> Self {
        Self::from_re(<T as SupersetOf<f64>>::from_subset(element))
    }
}

impl<T: RealField + Copy, const N: usize> SupersetOf<f32> for Dual<T, N> {
    #[inline]
    fn is_in_subset(&self) -> bool {
        <T as SupersetOf<f32>>::is_in_subset(&self.re)
    }
    #[inline]
    fn to_subset_unchecked(&self) -> f32 {
        <T as SupersetOf<f32>>::to_subset_unchecked(&self.re)
    }
    #[inline]
    fn from_subset(element: &f32) -> Self {
        Self::from_re(<T as SupersetOf<f32>>::from_subset(element))
    }
}

// ---------------------------------------------------------------------------
// approx: tolerance comparisons act on the value part only.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> AbsDiffEq for Dual<T, N> {
    type Epsilon = Self;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self::from_re(T::default_epsilon())
    }
    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.re)
    }
}

impl<T: RealField + Copy, const N: usize> RelativeEq for Dual<T, N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        Self::from_re(T::default_max_relative())
    }
    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.re.relative_eq(&other.re, epsilon.re, max_relative.re)
    }
}

impl<T: RealField + Copy, const N: usize> UlpsEq for Dual<T, N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon.re, max_ulps)
    }
}

impl<T: RealField + Copy, const N: usize> Field for Dual<T, N> {}

// ---------------------------------------------------------------------------
// ComplexField: a Dual is a *real* field element (imaginary part always 0).
// Smooth methods carry the chain rule; step/selection methods act on `re`.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy, const N: usize> ComplexField for Dual<T, N> {
    type RealField = Self;

    #[inline]
    fn from_real(re: Self::RealField) -> Self {
        re
    }
    #[inline]
    fn real(self) -> Self::RealField {
        self
    }
    #[inline]
    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }
    #[inline]
    fn modulus(self) -> Self::RealField {
        Signed::abs(&self)
    }
    #[inline]
    fn modulus_squared(self) -> Self::RealField {
        self * self
    }
    #[inline]
    fn argument(self) -> Self::RealField {
        if self.re >= T::zero() {
            Self::zero()
        } else {
            Self::pi()
        }
    }
    #[inline]
    fn norm1(self) -> Self::RealField {
        Signed::abs(&self)
    }
    #[inline]
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }
    #[inline]
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }

    #[inline]
    fn floor(self) -> Self {
        Self::from_re(self.re.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Self::from_re(self.re.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Self::from_re(self.re.round())
    }
    #[inline]
    fn trunc(self) -> Self {
        Self::from_re(self.re.trunc())
    }
    #[inline]
    fn fract(self) -> Self {
        Self::new(self.re.fract(), self.eps)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    // signum is piecewise constant => derivative 0 (overrides the trait default).
    #[inline]
    fn signum(self) -> Self {
        Self::from_re(self.re.signum())
    }

    #[inline]
    fn abs(self) -> Self::RealField {
        Signed::abs(&self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        Dual::hypot(self, other)
    }
    #[inline]
    fn recip(self) -> Self {
        Dual::recip(self)
    }
    #[inline]
    fn conjugate(self) -> Self {
        self
    }

    #[inline]
    fn sin(self) -> Self {
        Dual::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Dual::cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (Dual::sin(self), Dual::cos(self))
    }
    #[inline]
    fn tan(self) -> Self {
        Dual::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        Dual::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        Dual::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        Dual::atan(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        Dual::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        Dual::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        Dual::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        self.chain(
            self.re.asinh(),
            T::one() / (self.re * self.re + T::one()).sqrt(),
        )
    }
    #[inline]
    fn acosh(self) -> Self {
        self.chain(
            self.re.acosh(),
            T::one() / (self.re * self.re - T::one()).sqrt(),
        )
    }
    #[inline]
    fn atanh(self) -> Self {
        self.chain(self.re.atanh(), T::one() / (T::one() - self.re * self.re))
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        Dual::ln(self) / Dual::ln(base)
    }
    #[inline]
    fn log2(self) -> Self {
        self.chain(self.re.log2(), T::one() / (self.re * T::ln_2()))
    }
    #[inline]
    fn log10(self) -> Self {
        self.chain(self.re.log10(), T::one() / (self.re * T::ln_10()))
    }
    #[inline]
    fn ln(self) -> Self {
        Dual::ln(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), T::one() / (T::one() + self.re))
    }

    #[inline]
    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }
    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        if self.re >= T::zero() {
            Some(Dual::sqrt(self))
        } else {
            None
        }
    }
    #[inline]
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        let three = T::one() + T::one() + T::one();
        self.chain(c, T::one() / (three * c * c))
    }

    #[inline]
    fn exp(self) -> Self {
        Dual::exp(self)
    }
    #[inline]
    fn exp2(self) -> Self {
        let v = self.re.exp2();
        self.chain(v, v * T::ln_2())
    }
    #[inline]
    fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Dual::powi(self, n)
    }
    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        self.powf_dual(n)
    }
    #[inline]
    fn powc(self, n: Self) -> Self {
        self.powf_dual(n)
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.eps.iter().all(|e| e.is_finite())
    }
}

impl<T: RealField + Copy, const N: usize> RealField for Dual<T, N> {
    #[inline]
    fn is_sign_positive(&self) -> bool {
        self.re.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(&self) -> bool {
        self.re.is_sign_negative()
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if sign.re.is_sign_positive() {
            self.abs()
        } else {
            -self.abs()
        }
    }

    // max/min are selectors: decide on the value, return the chosen operand's
    // full dual (its derivative comes along; the other's is discarded).
    #[inline]
    fn max(self, other: Self) -> Self {
        if other > self {
            other
        } else {
            self
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self {
            other
        } else {
            self
        }
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        // self = y, other = x; d atan2(y, x) = (x dy - y dx) / (x^2 + y^2)
        let denom = self.re * self.re + other.re * other.re;
        let re = self.re.atan2(other.re);
        let eps = (self.eps * other.re - other.eps * self.re) / denom;
        Self::checked(re, eps)
    }

    #[inline]
    fn min_value() -> Option<Self> {
        T::min_value().map(Self::from_re)
    }

    #[inline]
    fn max_value() -> Option<Self> {
        T::max_value().map(Self::from_re)
    }

    #[inline]
    fn pi() -> Self {
        Self::from_re(T::pi())
    }
    #[inline]
    fn two_pi() -> Self {
        Self::from_re(T::two_pi())
    }
    #[inline]
    fn frac_pi_2() -> Self {
        Self::from_re(T::frac_pi_2())
    }
    #[inline]
    fn frac_pi_3() -> Self {
        Self::from_re(T::frac_pi_3())
    }
    #[inline]
    fn frac_pi_4() -> Self {
        Self::from_re(T::frac_pi_4())
    }
    #[inline]
    fn frac_pi_6() -> Self {
        Self::from_re(T::frac_pi_6())
    }
    #[inline]
    fn frac_pi_8() -> Self {
        Self::from_re(T::frac_pi_8())
    }
    #[inline]
    fn frac_1_pi() -> Self {
        Self::from_re(T::frac_1_pi())
    }
    #[inline]
    fn frac_2_pi() -> Self {
        Self::from_re(T::frac_2_pi())
    }
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::from_re(T::frac_2_sqrt_pi())
    }
    #[inline]
    fn e() -> Self {
        Self::from_re(T::e())
    }
    #[inline]
    fn log2_e() -> Self {
        Self::from_re(T::log2_e())
    }
    #[inline]
    fn log10_e() -> Self {
        Self::from_re(T::log10_e())
    }
    #[inline]
    fn ln_2() -> Self {
        Self::from_re(T::ln_2())
    }
    #[inline]
    fn ln_10() -> Self {
        Self::from_re(T::ln_10())
    }
}

// ---------------------------------------------------------------------------
// Batched forward-mode helpers: seed all N inputs, evaluate once, read off the
// value vector and the full Jacobian.
// ---------------------------------------------------------------------------

/// Lifts a point into a fully-seeded dual input: variable `i` carries `eᵢ`.
#[inline]
pub fn seeded<T: RealField + Copy, const N: usize>(x: &SVector<T, N>) -> SVector<Dual<T, N>, N> {
    SVector::from_fn(|i, _| Dual::seed(x[i], i))
}

/// The value (primal) part of an evaluated dual output.
#[inline]
pub fn values<T: RealField + Copy, const N: usize, const M: usize>(
    y: &SVector<Dual<T, N>, M>,
) -> SVector<T, M> {
    SVector::from_fn(|i, _| y[i].re)
}

#[inline]
pub fn dual_num<T: RealField + Copy>(re: T, eps: T) -> Dual<T> {
    Dual::new(re, SVector::<T, 1>::new(eps))
}

/// The `M×N` Jacobian of an evaluated dual output (row `r` = ∂yᵣ/∂x).
#[inline]
pub fn jacobian<T: RealField + Copy, const N: usize, const M: usize>(
    y: &SVector<Dual<T, N>, M>,
) -> SMatrix<T, M, N> {
    SMatrix::from_fn(|r, c| y[r].eps[c])
}

pub fn jacobian_mut<F, T, const N: usize, const M: usize>(
    f: &mut F,
    x: &SVector<T, N>,
    y: &mut SVector<T, M>,
    jacobian: &mut SMatrix<T, M, N>,
) where
    T: RealField + Copy,
    F: FnMut(&SVector<Dual<T, N>, N>, &mut SVector<Dual<T, N>, M>),
{
    let x_dual = seeded(x);
    let mut y_dual = SVector::zeros();
    f(&x_dual, &mut y_dual);
    for r in 0..M {
        y[r] = y_dual[r].re;
        for c in 0..N {
            jacobian[(r, c)] = y_dual[r].eps[c];
        }
    }
}

pub fn jvp_mut<F, T, const N: usize, const M: usize>(
    f: &mut F,
    x: &SVector<T, N>,
    v: &SVector<T, N>,
    y: &mut SVector<T, M>,
    jvp: &mut SVector<T, M>,
) where
    T: RealField + Copy,
    F: FnMut(&SVector<Dual<T>, N>, &mut SVector<Dual<T>, M>),
{
    let x_dual = SVector::from_fn(|r, _| dual_num(x[r], v[r]));
    let mut y_dual = SVector::zeros();

    f(&x_dual, &mut y_dual);

    for i in 0..M {
        y[i] = y_dual[i].re;
        jvp[i] = y_dual[i].eps[0];
    }
}

#[cfg(test)]
mod tests {
    use super::{jacobian, seeded, values, Dual};
    use nalgebra::{RealField, SVector};

    const EPS: f64 = 1e-9;

    // Compile-time proof that the full supertrait stack lands: this only type
    // checks if `Dual<f64, N>` actually implements `nalgebra::RealField`.
    fn assert_real_field<S: RealField>() {}

    #[test]
    fn dual_is_a_real_field() {
        assert_real_field::<Dual<f64, 1>>();
        assert_real_field::<Dual<f64, 3>>();
    }

    // A function written generically over `RealField`. Because `S` is generic,
    // every call resolves to the *trait* methods (ComplexField/RealField) —
    // exactly the path nalgebra's own algorithms take, not the inherent ones.
    fn sample<S: RealField + Copy>(x: S) -> S {
        (x * x + S::one()).sqrt() * x.sin() + x.atan2(S::one())
    }

    #[test]
    fn autodiff_through_realfield_trait_methods() {
        let x = 0.7_f64;
        let dual = sample(Dual::<f64, 1>::seed(x, 0));

        assert!((dual.re - sample(x)).abs() < EPS);

        let h = 1e-6;
        let fd = (sample(x + h) - sample(x - h)) / (2.0 * h);
        assert!(
            (dual.eps[0] - fd).abs() < 1e-6,
            "dual.eps = {}, fd = {}",
            dual.eps[0],
            fd
        );
    }

    #[test]
    fn realfield_selectors_route_the_derivative() {
        let a = Dual::<f64, 1>::new(1.0, SVector::<f64, 1>::new(10.0));
        let b = Dual::<f64, 1>::new(2.0, SVector::<f64, 1>::new(20.0));

        let hi = RealField::max(a, b);
        assert_eq!(hi.re, 2.0);
        assert_eq!(hi.eps[0], 20.0); // derivative of the winner, not of `a`

        let lo = RealField::min(a, b);
        assert_eq!(lo.re, 1.0);
        assert_eq!(lo.eps[0], 10.0);
    }

    // The batched win: seed both inputs once, evaluate once, get the whole 2x2
    // Jacobian in a single pass.
    #[test]
    fn batched_jacobian_single_pass() {
        // f(x, y) = [x^2 * y, sin(x) + y]
        // J = [[2xy, x^2], [cos x, 1]]
        fn f<S: RealField + Copy>(v: &SVector<S, 2>) -> SVector<S, 2> {
            SVector::<S, 2>::new(v[0] * v[0] * v[1], v[0].sin() + v[1])
        }

        let x = SVector::<f64, 2>::new(1.3, 0.7);
        let out = f(&seeded(&x));

        let val = values(&out);
        assert!((val[0] - 1.3 * 1.3 * 0.7).abs() < EPS);
        assert!((val[1] - (1.3_f64.sin() + 0.7)).abs() < EPS);

        let j = jacobian(&out);
        assert!((j[(0, 0)] - 2.0 * 1.3 * 0.7).abs() < EPS);
        assert!((j[(0, 1)] - 1.3 * 1.3).abs() < EPS);
        assert!((j[(1, 0)] - 1.3_f64.cos()).abs() < EPS);
        assert!((j[(1, 1)] - 1.0).abs() < EPS);
    }

    // nalgebra's matrix inverse is generic over the scalar, so it differentiates
    // automatically when the scalar is `Dual` — no custom matrix rule needed.
    // We check d(A^-1)/dt against the closed form  -A^-1 (dA/dt) A^-1.
    #[test]
    fn autodiff_through_matrix_inverse() {
        use nalgebra::SMatrix;

        // A(t) = [[2+t, 1], [1, 3+t]],  so dA/dt = I.  Seed t with eps = 1.
        let t = 0.5_f64;
        let d1 = Dual::<f64, 1>::new(1.0, SVector::<f64, 1>::new(0.0)); // constant 1
        let on = |re: f64| Dual::<f64, 1>::new(re, SVector::<f64, 1>::new(1.0)); // depends on t
        let a = SMatrix::<Dual<f64, 1>, 2, 2>::new(on(2.0 + t), d1, d1, on(3.0 + t));

        let inv = a.try_inverse().expect("A is invertible");

        let inv_val = inv.map(|d| d.re); // A^-1
        let inv_der = inv.map(|d| d.eps[0]); // d(A^-1)/dt, via autodiff

        // dA/dt = I  =>  d(A^-1)/dt = -A^-1 (I) A^-1 = -(A^-1)^2
        let expected = -(inv_val * inv_val);

        for i in 0..2 {
            for k in 0..2 {
                assert!(
                    (inv_der[(i, k)] - expected[(i, k)]).abs() < EPS,
                    "mismatch at ({i},{k}): {} vs {}",
                    inv_der[(i, k)],
                    expected[(i, k)]
                );
            }
        }
    }

    // Second derivatives "for free" by nesting: `Dual<f64,1>` is itself a
    // RealField, so `Dual<Dual<f64,1>,1>` type-checks and forward-over-forward
    // yields f''. Distinct types keep the two tangent levels from mixing.
    #[test]
    fn nested_dual_gives_second_derivative() {
        type D1 = Dual<f64, 1>;
        type D2 = Dual<D1, 1>;

        fn f<S: RealField + Copy>(x: S) -> S {
            // x*sin(x): f' = sin x + x cos x,  f'' = 2 cos x - x sin x
            x * x.sin()
        }

        let x0 = 0.7_f64;
        // x = x0 + e1 + e2 : `re` carries (x0 + e1), `eps` carries coeff of e2 (=1).
        let x = D2::new(
            D1::new(x0, SVector::<f64, 1>::new(1.0)),
            SVector::<D1, 1>::new(D1::new(1.0, SVector::<f64, 1>::new(0.0))),
        );

        let y = f(x);

        let value = y.re.re;
        let d1a = y.re.eps[0]; // d/de1
        let d1b = y.eps[0].re; // d/de2
        let d2 = y.eps[0].eps[0]; // d^2/de1 de2

        let (s, c) = (x0.sin(), x0.cos());
        assert!((value - x0 * s).abs() < EPS);
        assert!((d1a - (s + x0 * c)).abs() < EPS); // f'
        assert!((d1b - (s + x0 * c)).abs() < EPS); // f' again (the other tangent)
        assert!((d2 - (2.0 * c - x0 * s)).abs() < EPS); // f''
    }

    // NaN policy: a derivative that does not exist — here ‖·‖ at the origin,
    // where sqrt'(0) = 1/0 — is caught loudly by a debug assertion at the
    // source op, never silently fabricated as a 0 sub-gradient. In release the
    // assertion compiles out and the non-finite tangent propagates faithfully,
    // so this check only applies to debug builds.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-finite derivative")]
    fn singular_derivative_trips_debug_assertion() {
        let _ = Dual::<f64, 1>::seed(0.0, 0).sqrt();
    }
}
