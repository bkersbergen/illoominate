use crate::sessrec::vmisknn::Scored;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub type RowIndex = u32;
pub type Score = f32;

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SimilarRow {
    pub row: RowIndex,
    pub similarity: Score,
}

impl SimilarRow {
    pub fn new(row: RowIndex, similarity: Score) -> Self {
        SimilarRow { row, similarity }
    }
}

impl From<&Scored> for SimilarRow {
    fn from(session_score: &Scored) -> Self {
        SimilarRow {
            row: session_score.id as RowIndex,
            similarity: session_score.score as Score,
        }
    }
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(sim_a: &SimilarRow, sim_b: &SimilarRow) -> Ordering {
    match sim_a.similarity.partial_cmp(&sim_b.similarity) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        None => Ordering::Equal,
    }
}

impl Eq for SimilarRow {}

impl Ord for SimilarRow {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for SimilarRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}
