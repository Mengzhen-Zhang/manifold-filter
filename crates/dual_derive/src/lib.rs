extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Index, LitInt};

#[proc_macro_derive(Manifold, attributes(manifold))]
pub fn derive_manifold(input: TokenStream) -> TokenStream {
    // Parse the incoming tokens into a syntax tree we can read
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Grab the fields of the struct
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Manifold derive only supports structs with named fields!"),
        },
        _ => panic!("Manifold derive can only be applied to structs!"),
    };

    let mut field_names = Vec::new();
    let mut field_types = Vec::new();
    let mut field_dims = Vec::new();

    // Iterate through fields to extract names, types, and the #[manifold(dim = X)] attribute
    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;

        // Find our attribute: #[manifold(dim = ...)]
        let mut dim_val: Option<LitInt> = None;
        for attr in &field.attrs {
            if attr.path().is_ident("manifold") {
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("dim") {
                        let value = meta.value()?;
                        dim_val = Some(value.parse()?);
                        Ok(())
                    } else {
                        Err(meta.error("unsupported manifold attribute"))
                    }
                });
            }
        }

        let dim = dim_val.expect(
            "Each manifold field must specify an explicit #[manifold(dim = ...)] attribute!",
        );

        field_names.push(field_name);
        field_types.push(field_type);
        field_dims.push(dim);
    }

    // Create syn::Index tokens matching field order positions (0, 1, 2...)
    // This allows exact multi-field tuple unpacking inside quote repetitions
    let tuple_indices: Vec<Index> = (0..field_names.len()).map(Index::from).collect();

    // Generate code blocks for the compile-time slice offsets
    // We compute offsets sequentially: 0, 0 + dim1, 0 + dim1 + dim2, etc.
    let mut offsets = Vec::new();
    let mut current_offset = quote! { 0 };
    for dim in &field_dims {
        offsets.push(current_offset.clone());
        current_offset = quote! { #current_offset + #dim };
    }

    // Generate the final Rust token stream expansion
    let expanded = quote! {
        impl<U> Manifold<U, { #current_offset }> for #name
        where
            U: nalgebra::RealField + Copy,
            #( #field_types: Manifold<U, #field_dims> ),*
        {
            // Map the Tangent associated type to a nested tuple matching field ordering
            type Tangent<T> = ( #( <#field_types as Manifold<U, #field_dims>>::Tangent<T>, )* );

            fn box_plus(&self, delta: &Self::Tangent<U>) -> Self {
                Self {
                    #(
                        // Maps field name with matching tuple indexing position
                        #field_names: self.#field_names.box_plus(&delta.#tuple_indices),
                    )*
                }
            }

            fn box_minus(&self, other: &Self) -> Self::Tangent<U> {
                (
                    #( self.#field_names.box_minus(&other.#field_names), )*
                )
            }

            fn tangent_to_vector<T: nalgebra::RealField + Copy>(t: &Self::Tangent<T>) -> nalgebra::SVector<T, { #current_offset }> {
                let mut out = nalgebra::SVector::<T, { #current_offset }>::zeros();

                // Statically populate the sliced vector using our calculated offsets
                #(
                    let block = <#field_types as Manifold<U, #field_dims>>::tangent_to_vector(&t.#tuple_indices);
                    out.fixed_rows_mut::<#field_dims>(#offsets).copy_from(&block);
                )*

                out
            }

            fn vector_to_tangent<T: nalgebra::RealField + Copy>(v: &nalgebra::SVector<T, { #current_offset }>) -> Self::Tangent<T> {
                (
                    #(
                        {
                            let block = v.fixed_rows::<#field_dims>(#offsets).into_owned();
                            <#field_types as Manifold<U, #field_dims>>::vector_to_tangent(&block)
                        }
                    ,)*
                )
            }

            fn vector_transport<T: nalgebra::RealField + Copy>(
                &self,
                v: &Self::Tangent<T>,
                delta: &Self::Tangent<T>,
            ) -> Self::Tangent<T> {
                (
                    #(
                        self.#field_names.vector_transport(&v.#tuple_indices, &delta.#tuple_indices),
                    )*
                )
            }
        }
    };

    TokenStream::from(expanded)
}
