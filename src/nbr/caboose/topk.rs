use crate::nbr::caboose::types::RowIndex;
use crate::nbr::caboose::types::SimilarRow;
use serde::{Deserialize, Serialize};
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TopK {
    heap: BinaryHeap<SimilarRow>,
    sorted_keys: Vec<RowIndex>,
}

impl TopK {
    pub(crate) fn new(heap: BinaryHeap<SimilarRow>) -> Self {
        let mut keys: Vec<RowIndex> = heap.iter().map(|entry| entry.row).collect();
        keys.sort();

        Self {
            heap,
            sorted_keys: keys,
        }
    }

    pub(crate) fn iter(&self) -> Iter<SimilarRow> {
        self.heap.iter()
    }
}
