#![no_std]
//! Forward-mode dual numbers: a scalar carrying a value (`re`) and its
//! derivative (`eps`). `Dual<T>` implements `nalgebra::RealField`, so it can be
//! used as the scalar type in any `RealField`-generic computation to obtain
//! exact derivatives by automatic differentiation.

use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{ComplexField, Field, RealField, SimdValue};
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{SubsetOf, SupersetOf};

#[derive(Debug, Clone, Copy)]
pub struct Dual<T> {
    pub re: T,
    pub eps: T,
}

impl<T: PartialOrd> PartialOrd for Dual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
	self.re.partial_cmp(&other.re)
    }
}

impl<T: PartialEq> PartialEq for Dual<T> {
    fn eq(&self, other: &Self) -> bool {
	self.re.eq(&other.re)
    }
}

impl<T> Dual<T> {
    #[inline]
    pub const fn new(re: T, eps: T) -> Self {
	Self { re, eps }
    }

    #[inline]
    pub fn from_re(re: T) -> Self
    where
	T: num_traits::Zero
    {
	Self {re, eps: T::zero() }
    }
}


impl<T: RealField + Copy> Dual<T> {
      #[inline]
      fn chain(self, val: T, deriv: T) -> Self { Self::new(val, deriv *
  self.eps) }
  }


impl<T: RealField> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.eps + rhs.eps)
    }
}

impl<T: RealField> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.eps - rhs.eps)
    }
}

impl<T: RealField + Copy> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Product Rule: (u*v)' = u'*v + u*v'
        Self::new(self.re * rhs.re, self.eps * rhs.re + self.re * rhs.eps)
    }
}

impl<T: RealField + Copy> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Quotient Rule: (u/v)' = (u'*v - u*v') / v^2
        let denom = rhs.re * rhs.re;
        Self::new(self.re / rhs.re, (self.eps * rhs.re - self.re * rhs.eps) / denom)
    }
}

impl<T: RealField + Copy> Mul<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.eps * rhs)
    }
}

impl<T: RealField + Copy> Div<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs, self.eps / rhs)
    }
}

impl<T: RealField> Add<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.re + rhs, self.eps)
    }
}

impl<T: RealField> Sub<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::new(self.re - rhs, self.eps)
    }
}

impl<T: RealField> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
	Self::new(-self.re, -self.eps)
    }
}

impl Mul<Dual<f64>> for f64 {
    type Output = Dual<f64>;
    #[inline]
    fn mul(self, rhs: Dual<f64>) -> Self::Output {
	Dual::<f64>::new(self * rhs.re, self * rhs.eps)
    }
}

impl Mul<Dual<f32>> for f32 {
    type Output = Dual<f32>;
    #[inline]
    fn mul(self, rhs: Dual<f32>) -> Self::Output {
	Dual::<f32>::new(self * rhs.re, self * rhs.eps)
    }
}


impl Div<Dual<f64>> for f64 {
    type Output = Dual<f64>;
    #[inline]
    fn div(self, rhs: Dual<f64>) -> Self::Output {
	Dual::<f64>::new(self / rhs.re, self / rhs.eps)
    }
}

impl Div<Dual<f32>> for f32 {
    type Output = Dual<f32>;
    #[inline]
    fn div(self, rhs: Dual<f32>) -> Self::Output {
	Dual::<f32>::new(self / rhs.re, self / rhs.eps)
    }
}


impl Add<Dual<f64>> for f64 {
    type Output = Dual<f64>;
    #[inline]
    fn add(self, rhs: Dual<f64>) -> Self::Output {
	Dual::<f64>::new(self + rhs.re, rhs.eps)
    }
}

impl Add<Dual<f32>> for f32 {
    type Output = Dual<f32>;
    #[inline]
    fn add(self, rhs: Dual<f32>) -> Self::Output {
	Dual::<f32>::new(self + rhs.re, rhs.eps)
    }
}

impl Sub<Dual<f64>> for f64 {
    type Output = Dual<f64>;
    #[inline]
    fn sub(self, rhs: Dual<f64>) -> Self::Output {
	Dual::<f64>::new(self - rhs.re, rhs.eps)
    }
}

impl Sub<Dual<f32>> for f32 {
    type Output = Dual<f32>;
    #[inline]
    fn sub(self, rhs: Dual<f32>) -> Self::Output {
	Dual::<f32>::new(self - rhs.re, rhs.eps)
    }
}


impl<T: RealField + Copy> Dual<T> {
    #[inline]
    pub fn recip(self) -> Self {
	Self::new(T::one() / self.re, - self.eps / self.re.powi(2))
    }

    #[inline]
    pub fn sin(self) -> Self {
	Self::new(self.re.sin(), self.re.cos() * self.eps)
    }

    #[inline]
    pub fn cos(self) -> Self {
	Self::new(self.re.cos(), -self.re.sin() * self.eps)
    }

    

    #[inline]
    pub fn sqrt(self) -> Self {
	let two = T::from_i8(2).expect("convertion fail");
	
	Self::new(self.re.sqrt(),
		  (self.re.sqrt() * two).recip() * self.eps)
    }
    
    #[inline]
    pub fn exp(self) -> Self {
	Self::new(self.re.exp(), self.re.exp() * self.eps)
    }

    #[inline]
    pub fn ln(self) -> Self {
	Self::new(self.re.ln(), self.eps / self.re)
    }

    #[inline]
    pub fn powi(self, n: i32) -> Self {
	let n_t = T::from_i32(n).expect("convertion fail");
	Self::new(self.re.powi(n), n_t * self.re.powi(n-1) * self.eps)
    }

    #[inline]
    pub fn powf(self, p: T) -> Self {
	Self::new(self.re.powf(p), p * self.re.powf(p-T::one()) * self.eps)
    }

    #[inline]
    pub fn tan(self) -> Self {
	Self::new(self.re.tan(), self.re.cos().powi(2).recip() * self.eps)
    }

    #[inline]
    pub fn asin(self) -> Self {
	let der = (T::one() - self.re.powi(2)).sqrt().recip();
	Self::new(self.re.asin(), der * self.eps)
    }

    #[inline]
    pub fn acos(self) -> Self {
	let der = -(T::one() - self.re.powi(2)).sqrt().recip();
	Self::new(self.re.acos(), der * self.eps)
    }

    #[inline]
    pub fn atan(self) -> Self {
	let der = (T::one() + self.re.powi(2)).recip();
	Self::new(self.re.atan(), der * self.eps)
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
	let x = self;
	let y = other;
	let re = (x.re.powi(2) + y.re.powi(2)).sqrt();
	let eps = (x.re * x.eps + y.re * y.eps) / re;
	Self::new(re, eps)
    }

    #[inline]
    pub fn abs(&self) -> Self {
	Self::new(self.re.abs(), self.re.signum() * self.eps)   
    }
    

    #[inline]
    pub fn tanh(self) -> Self {
	Self::new(self.re.tanh(), (T::one() - self.re.tanh().powi(2)) * self.eps)
    }

    #[inline]
    pub fn sinh(self) -> Self {
	Self::new(self.re.sinh(), self.re.cosh() * self.eps)
    }

    #[inline]
    pub fn cosh(self) -> Self {
	Self::new(self.re.cosh(), self.re.sinh() * self.eps)
    }
}

impl<T: RealField> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) {
	self.re += rhs.re;
	self.eps += rhs.eps;
    }
}

impl<T: RealField> SubAssign for Dual<T> {
    fn sub_assign(&mut self, rhs: Self) {
	self.re -= rhs.re;
	self.eps -= rhs.eps;
    }
}

impl<T: RealField + Copy> MulAssign for Dual<T> {
    fn mul_assign(&mut self, rhs: Self) {
	*self = *self * rhs; // product rule, via Mul
    }
}

impl<T: RealField + Copy> DivAssign for Dual<T> {
    fn div_assign(&mut self, rhs: Self) {
	*self = *self / rhs; // quotient rule, via Div
    }
}

impl<T: RealField> AddAssign<T> for Dual<T> {
    fn add_assign(&mut self, rhs: T) {
	self.re += rhs;
    }
}

impl<T: RealField> SubAssign<T> for Dual<T> {
    fn sub_assign(&mut self, rhs: T) {
	self.re -= rhs;
    }
}

impl<T: RealField + Copy> MulAssign<T> for Dual<T> {
    fn mul_assign(&mut self, rhs: T) {
	self.re *= rhs;
	self.eps *= rhs;
    }
}

impl<T: RealField + Copy> DivAssign<T> for Dual<T> {
    fn div_assign(&mut self, rhs: T) {
	self.re /= rhs;
	self.eps /= rhs;
    }
}

impl<T: RealField> Zero for Dual<T> {
    fn zero() -> Self {
	Self::from_re(T::zero())
    }

    fn is_zero(&self) -> bool {
	self.re.is_zero() && self.eps.is_zero()
    }

    fn set_zero(&mut self) {
        self.re = T::zero();
	self.eps = T::zero();
    }
}

impl<T: RealField + Copy> One for Dual<T> {
    fn one() -> Self {
	Self::from_re(T::one())
    }

    fn set_one(&mut self) {
	*self = Self::one();
    }

    fn is_one(&self) -> bool
    where
        Self: PartialEq, {
	self.re.is_one() && self.eps.is_zero()
    }
}





// ---------------------------------------------------------------------------
// Remainder + its assign form. `Rem` exists only to satisfy `Num: NumOps`; the
// modulus boundary is non-differentiable and never sits on an AD path, so we
// keep the value correct and pass the derivative straight through.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy> Rem for Dual<T> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
	Self::new(self.re % rhs.re, self.eps)
    }
}

impl<T: RealField + Copy> RemAssign for Dual<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
	*self = *self % rhs;
    }
}

// ---------------------------------------------------------------------------
// num-traits scalar layer
// ---------------------------------------------------------------------------
impl<T: RealField + Copy> Num for Dual<T> {
    type FromStrRadixErr = ();
    #[inline]
    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
	Err(()) // never invoked by linear-algebra paths
    }
}

impl<T: RealField + Copy> Signed for Dual<T> {
    // |x| = x for x >= 0 else -x; Neg already carries the matching signum*eps.
    #[inline]
    fn abs(&self) -> Self {
	if self.re >= T::zero() { *self } else { -*self }
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
	if self.re <= other.re { Self::zero() } else { *self - *other }
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
	Self::new(v, T::zero())
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

impl<T: RealField + Copy> FromPrimitive for Dual<T> {
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

impl<T: fmt::Display> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	write!(f, "{} + {}\u{3b5}", self.re, self.eps)
    }
}

// ---------------------------------------------------------------------------
// simba: a Dual behaves as a SIMD scalar of width 1 (no vectorization).
// ---------------------------------------------------------------------------
impl<T: RealField + Copy> SimdValue for Dual<T> {
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
	if cond { self } else { other }
    }
}

// ---------------------------------------------------------------------------
// simba subset/superset: a Dual embeds the reals (eps = 0); the real <-> f32/f64
// conversion is delegated to the inner scalar T.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy> SubsetOf<Self> for Dual<T> {
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

impl<T: RealField + Copy> SupersetOf<f64> for Dual<T> {
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

impl<T: RealField + Copy> SupersetOf<f32> for Dual<T> {
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
impl<T: RealField + Copy> AbsDiffEq for Dual<T> {
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

impl<T: RealField + Copy> RelativeEq for Dual<T> {
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

impl<T: RealField + Copy> UlpsEq for Dual<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
	T::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
	self.re.ulps_eq(&other.re, epsilon.re, max_ulps)
    }
}

impl<T: RealField + Copy> Field for Dual<T> {}

// ---------------------------------------------------------------------------
// ComplexField: a Dual is a *real* field element (imaginary part always 0).
// Smooth methods carry the chain rule; step/selection methods act on `re`.
// ---------------------------------------------------------------------------
impl<T: RealField + Copy> Dual<T> {
    // Power with a dual exponent: d/dt x^n = x^n (n' ln x + n x'/x).
    #[inline]
    fn powf_dual(self, n: Self) -> Self {
	let v = self.re.powf(n.re);
	let d = v * (n.eps * self.re.ln() + n.re * self.eps / self.re);
	Self::new(v, d)
    }
}

impl<T: RealField + Copy> ComplexField for Dual<T> {
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
	if self.re >= T::zero() { Self::zero() } else { Self::pi() }
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
	Self::new(self.re.floor(), T::zero())
    }
    #[inline]
    fn ceil(self) -> Self {
	Self::new(self.re.ceil(), T::zero())
    }
    #[inline]
    fn round(self) -> Self {
	Self::new(self.re.round(), T::zero())
    }
    #[inline]
    fn trunc(self) -> Self {
	Self::new(self.re.trunc(), T::zero())
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
	Self::new(self.re.signum(), T::zero())
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
	self.chain(self.re.asinh(), T::one() / (self.re * self.re + T::one()).sqrt())
    }
    #[inline]
    fn acosh(self) -> Self {
	self.chain(self.re.acosh(), T::one() / (self.re * self.re - T::one()).sqrt())
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
	if self.re >= T::zero() { Some(Dual::sqrt(self)) } else { None }
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
	self.re.is_finite() && self.eps.is_finite()
    }
}

impl<T: RealField + Copy> RealField for Dual<T> {
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
	if other > self {
	    self
	} else {
	    other
	}
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
	let re = self.re.clamp(min.re, max.re);
	let eps = if re == min.re || re == max.re {
	    T::zero()
	} else {
	    self.eps
	};
	Self::new(re, eps)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
	let y = self;
	let x = other;
	let re = y.re.atan2(x.re);
	let eps = (x.re * y.eps - y.re * x.eps) / (x.re.powi(2) + y.re.powi(2));
	Self::new(re, eps)
    }

    #[inline]
    fn min_value() -> Option<Self> {
	T::min_value().map(|v| Self::from_re(v))
    }

    #[inline]
    fn max_value() -> Option<Self> {
	T::max_value().map(|v| Self::from_re(v))
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

#[cfg(test)]
mod tests {
    use super::Dual;
    use nalgebra::RealField;

    const EPS: f64 = 1e-9;

    // Compile-time proof that the full supertrait stack lands: this only type
    // checks if `Dual<f64>` actually implements `nalgebra::RealField`.
    fn assert_real_field<S: RealField>() {}

    #[test]
    fn dual_is_a_real_field() {
        assert_real_field::<Dual<f64>>();
    }

    // A function written generically over `RealField`. Because `S` is generic,
    // every call below resolves to the *trait* methods (ComplexField/RealField),
    // exactly the path nalgebra's own algorithms take — not the inherent ones.
    fn sample<S: RealField + Copy>(x: S) -> S {
        // sqrt(x^2 + 1) * sin(x) + atan2(x, 1)
        (x * x + S::one()).sqrt() * x.sin() + x.atan2(S::one())
    }

    #[test]
    fn autodiff_through_realfield_trait_methods() {
        let x = 0.7_f64;

        // Seed the derivative direction (eps = 1) and push x through `sample`.
        let dual = sample(Dual::new(x, 1.0));

        // Value matches the plain-f64 evaluation.
        assert!((dual.re - sample(x)).abs() < EPS);

        // Derivative matches a central finite difference of the same function.
        let h = 1e-6;
        let fd = (sample(x + h) - sample(x - h)) / (2.0 * h);
        assert!((dual.eps - fd).abs() < 1e-6, "dual.eps = {}, fd = {}", dual.eps, fd);
    }

    #[test]
    fn realfield_selectors_act_on_value_and_route_the_derivative() {
        // max/min are selectors: they return the chosen operand *with its
        // derivative*, deciding on the value part only.
        let a = Dual::new(1.0, 10.0); // value 1, derivative 10
        let b = Dual::new(2.0, 20.0); // value 2, derivative 20

        let hi = RealField::max(a, b);
        assert_eq!(hi.re, 2.0);
        assert_eq!(hi.eps, 20.0); // derivative of the winner, not of `a`

        let lo = RealField::min(a, b);
        assert_eq!(lo.re, 1.0);
        assert_eq!(lo.eps, 10.0);
    }
}
