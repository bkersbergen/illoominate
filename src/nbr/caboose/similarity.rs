use serde::{Deserialize, Serialize};

pub trait Similarity {
    fn from_norms(&self, dot_product: f64, norm_a: f64, norm_b: f64) -> f64;
    fn accumulate_norm(&self, value: f64) -> f64;
    fn finalize_norm(&self, accum: f64) -> f64;
}

pub const JACCARD: Jaccard = Jaccard {};

#[derive(Serialize, Deserialize)]
pub struct Jaccard {}

impl Similarity for Jaccard {
    #[inline(always)]
    fn from_norms(&self, dot_product: f64, norm_a: f64, norm_b: f64) -> f64 {
        dot_product / (norm_a + norm_b - dot_product)
    }

    fn accumulate_norm(&self, _value: f64) -> f64 {
        1.0
    }

    fn finalize_norm(&self, accum: f64) -> f64 {
        accum
    }
}
