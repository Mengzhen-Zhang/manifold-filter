use core::ops::{Add, Div, Mul, Sub};
use nalgebra::RealField;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Dual<T> {
    pub re: T,
    pub eps: T,
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
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.eps * rhs)
    }
}

impl<T: RealField + Copy> Div<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self::new(self.re / rhs, self.eps / rhs)
    }
}

impl<T: RealField> Add<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self::new(self.re + rhs, self.eps)
    }
}

impl<T: RealField> Sub<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.re - rhs, self.eps)
    }
}
