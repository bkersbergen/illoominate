use crate::conf::illoominateconfig::HpoConfig;
use std::collections::HashMap;
use std::hash::Hash;

pub type UserId = u32;

#[derive(Debug, Clone, Copy)]
pub struct HyperParams {
    pub m: isize,
    pub r_b: f64,
    pub r_g: f64,
    pub alpha: f64,
    pub k: usize,
}

impl From<&HpoConfig> for HyperParams {
    fn from(hpo: &HpoConfig) -> Self {
        HyperParams {
            m: hpo
                .m
                .unwrap_or_else(|| panic!("m is not set in the HPO configuration"))
                as isize,
            r_b: hpo
                .r_b
                .unwrap_or_else(|| panic!("r_b is not set in the HPO configuration")),
            r_g: hpo
                .r_g
                .unwrap_or_else(|| panic!("r_g is not set in the HPO configuration")),
            alpha: hpo
                .alpha
                .unwrap_or_else(|| panic!("alpha is not set in the HPO configuration")),
            k: hpo
                .k
                .unwrap_or_else(|| panic!("k is not set in the HPO configuration")),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Hash, Ord, PartialOrd)]
pub struct Basket {
    pub id: usize,
    pub items: Vec<usize>,
}

impl Basket {
    pub fn new(id: usize, items: Vec<usize>) -> Self {
        Basket { id, items }
    }
}
#[derive(Clone, Debug)]
pub struct SparseItemVector {
    pub entries: HashMap<usize, f64>,
}

impl Default for SparseItemVector {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseItemVector {
    pub fn new() -> Self {
        SparseItemVector {
            entries: HashMap::new(),
        }
    }

    #[inline]
    pub fn plus_at(&mut self, index: usize, value: f64) {
        let entry = self.entries.entry(index).or_insert(0.0);
        *entry += value;
    }

    #[inline]
    pub fn plus_mult(&mut self, mult: f64, other: &SparseItemVector) {
        for (index, value) in other.entries.iter() {
            self.plus_at(*index, mult * *value);
        }
    }
}
