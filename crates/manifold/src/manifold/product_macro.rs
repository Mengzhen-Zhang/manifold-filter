//! `product_manifold!` — generate a named-field product manifold from component
//! manifolds, with total dimensions summed automatically (pure `macro_rules!`,
//! no dependencies). The generated struct uses a flat `SVector` tangent and
//! implements `Manifold<f64, ΣAᵢ, ΣTᵢ>`; fields are accessed directly (`s.p`).

/// ```ignore
/// product_manifold! {
///     pub struct InsState {
///         p:  Rn<f64, 3> [3, 3],   // field: Type [AMBIENT_DIM, TANGENT_DIM]
///         v:  Rn<f64, 3> [3, 3],
///         q:  SO3<f64>   [4, 3],
///         ba: Rn<f64, 3> [3, 3],
///         bg: Rn<f64, 3> [3, 3],
///     }
/// }
/// ```
#[macro_export]
macro_rules! product_manifold {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($field:ident : $ty:ty [$a:literal, $t:literal]),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        $vis struct $name {
            $(pub $field: $ty),+
        }

        impl $crate::manifold::Manifold<f64, { $($a +)+ 0 }, { $($t +)+ 0 }> for $name {
            type TangentVector = nalgebra::SVector<f64, { $($t +)+ 0 }>;

            fn retract(&self, delta: &Self::TangentVector) -> Self {
                let this = self;
                $crate::product_manifold!(@retract this delta (0usize); (); $($field : $ty [$a, $t])+)
            }

            fn local_lift(&self, other: &Self) -> Self::TangentVector {
                let this = self;
                let mut out = nalgebra::SVector::<f64, { $($t +)+ 0 }>::zeros();
                $crate::product_manifold!(@lift out this other (0usize); $($field : $ty [$a, $t])+);
                out
            }

            fn pushforward_jacobian(&self) -> nalgebra::SMatrix<f64, { $($a +)+ 0 }, { $($t +)+ 0 }> {
                let this = self;
                let mut out = nalgebra::SMatrix::<f64, { $($a +)+ 0 }, { $($t +)+ 0 }>::zeros();
                $crate::product_manifold!(@push out this (0usize) (0usize); $($field : $ty [$a, $t])+);
                out
            }

            fn apply_pushforward(&self, tangent: &Self::TangentVector) -> nalgebra::SVector<f64, { $($a +)+ 0 }> {
                let this = self;
                let mut out = nalgebra::SVector::<f64, { $($a +)+ 0 }>::zeros();
                $crate::product_manifold!(@apply out this tangent (0usize) (0usize); $($field : $ty [$a, $t])+);
                out
            }

            fn vector_to_tangent(vec: &nalgebra::SVector<f64, { $($t +)+ 0 }>) -> Self::TangentVector { *vec }
            fn tangent_to_vector(tangent: &Self::TangentVector) -> nalgebra::SVector<f64, { $($t +)+ 0 }> { *tangent }

            fn from_ambient(vec: &nalgebra::SVector<f64, { $($a +)+ 0 }>) -> Self {
                $crate::product_manifold!(@from vec (0usize); (); $($field : $ty [$a, $t])+)
            }

            fn to_ambient(&self) -> nalgebra::SVector<f64, { $($a +)+ 0 }> {
                let this = self;
                let mut out = nalgebra::SVector::<f64, { $($a +)+ 0 }>::zeros();
                $crate::product_manifold!(@to out this (0usize); $($field : $ty [$a, $t])+);
                out
            }

            fn parallel_transport(&self, delta: &Self::TangentVector) -> nalgebra::SMatrix<f64, { $($t +)+ 0 }, { $($t +)+ 0 }> {
                let this = self;
                let mut out = nalgebra::SMatrix::<f64, { $($t +)+ 0 }, { $($t +)+ 0 }>::zeros();
                $crate::product_manifold!(@xport out this delta (0usize); $($field : $ty [$a, $t])+);
                out
            }
        }
    };

    // Offsets are passed as parenthesized `tt`s so two can sit adjacent.
    // retract: fold a `Self { field: ... }` literal, slicing the flat delta.
    (@retract $s:tt $d:tt $toff:tt; ($($acc:tt)*); $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $crate::product_manifold!(@retract $s $d ($toff + $t);
            ($($acc)* $field: <$ty as $crate::manifold::Manifold<f64, $a, $t>>::retract(
                &$s.$field,
                &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::vector_to_tangent(
                    &$d.fixed_rows::<$t>($toff).into_owned())),);
            $($rest)*)
    };
    (@retract $s:tt $d:tt $toff:tt; ($($acc:tt)*);) => { Self { $($acc)* } };

    (@lift $out:ident $s:tt $o:tt $toff:tt; $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $out.fixed_rows_mut::<$t>($toff).copy_from(
            &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::tangent_to_vector(
                &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::local_lift(&$s.$field, &$o.$field)));
        $crate::product_manifold!(@lift $out $s $o ($toff + $t); $($rest)*);
    };
    (@lift $out:ident $s:tt $o:tt $toff:tt;) => {};

    (@push $out:ident $s:tt $aoff:tt $toff:tt; $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $out.fixed_view_mut::<$a, $t>($aoff, $toff).copy_from(
            &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::pushforward_jacobian(&$s.$field));
        $crate::product_manifold!(@push $out $s ($aoff + $a) ($toff + $t); $($rest)*);
    };
    (@push $out:ident $s:tt $aoff:tt $toff:tt;) => {};

    (@apply $out:ident $s:tt $tan:tt $aoff:tt $toff:tt; $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $out.fixed_rows_mut::<$a>($aoff).copy_from(
            &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::apply_pushforward(&$s.$field,
                &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::vector_to_tangent(
                    &$tan.fixed_rows::<$t>($toff).into_owned())));
        $crate::product_manifold!(@apply $out $s $tan ($aoff + $a) ($toff + $t); $($rest)*);
    };
    (@apply $out:ident $s:tt $tan:tt $aoff:tt $toff:tt;) => {};

    (@from $v:tt $aoff:tt; ($($acc:tt)*); $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $crate::product_manifold!(@from $v ($aoff + $a);
            ($($acc)* $field: <$ty as $crate::manifold::Manifold<f64, $a, $t>>::from_ambient(
                &$v.fixed_rows::<$a>($aoff).into_owned()),);
            $($rest)*)
    };
    (@from $v:tt $aoff:tt; ($($acc:tt)*);) => { Self { $($acc)* } };

    (@to $out:ident $s:tt $aoff:tt; $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $out.fixed_rows_mut::<$a>($aoff).copy_from(
            &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::to_ambient(&$s.$field));
        $crate::product_manifold!(@to $out $s ($aoff + $a); $($rest)*);
    };
    (@to $out:ident $s:tt $aoff:tt;) => {};

    (@xport $out:ident $s:tt $d:tt $toff:tt; $field:ident : $ty:ty [$a:literal, $t:literal] $($rest:tt)*) => {
        $out.fixed_view_mut::<$t, $t>($toff, $toff).copy_from(
            &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::parallel_transport(&$s.$field,
                &<$ty as $crate::manifold::Manifold<f64, $a, $t>>::vector_to_tangent(
                    &$d.fixed_rows::<$t>($toff).into_owned())));
        $crate::product_manifold!(@xport $out $s $d ($toff + $t); $($rest)*);
    };
    (@xport $out:ident $s:tt $d:tt $toff:tt;) => {};
}

#[cfg(test)]
mod tests {
    use crate::manifold::{Manifold, Rn, SO3};
    use nalgebra::{SVector, Vector3};

    crate::product_manifold! {
        struct TestState {
            pos: Rn<f64, 3> [3, 3],
            rot: SO3<f64>   [4, 3],
        }
    }

    #[test]
    fn macro_product_implements_manifold() {
        let s = TestState {
            pos: Rn {
                x: Vector3::new(1.0, 2.0, 3.0),
            },
            rot: SO3::from_parts(1.0, 2.0, 3.0, 4.0),
        };

        // Dimensions summed: A = 3+4 = 7, T = 3+3 = 6.
        assert_eq!(s.pushforward_jacobian().shape(), (7, 6));

        // from_ambient inverts to_ambient (round trip through the flat vector).
        let a = s.to_ambient();
        let s2 = TestState::from_ambient(&a);
        assert!((a - s2.to_ambient()).norm() < 1e-12);

        // retract by a tangent matches the per-component retractions.
        let delta = SVector::<f64, 6>::from_row_slice(&[0.1, -0.2, 0.3, 0.05, 0.1, -0.05]);
        let moved = s.retract(&delta);
        let pos_expected = s.pos.retract(&delta.fixed_rows::<3>(0).into_owned());
        let rot_expected = s.rot.retract(&delta.fixed_rows::<3>(3).into_owned());
        assert!((moved.pos.x - pos_expected.x).norm() < 1e-12);
        assert!(moved.rot.local_lift(&rot_expected).norm() < 1e-12);

        // local_lift round-trips the retraction.
        assert!((s.local_lift(&moved) - delta).norm() < 1e-9);
    }
}
