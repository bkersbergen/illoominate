use std::collections::BinaryHeap;

use itertools::Itertools;
use sprs::CsMat;

use crate::nbr::caboose::row_accumulator::RowAccumulator;
use crate::nbr::caboose::similarity::{Similarity, JACCARD};
use crate::nbr::caboose::topk::TopK;
use crate::nbr::caboose::types::SimilarRow;
use crate::nbr::tifuknn::types::UserId;

pub struct SparseTopKIndex {
    pub representations: CsMat<f64>, // TODO required for one bench, fix this
    #[allow(dead_code)]
    pub(crate) representations_transposed: CsMat<f64>,
    pub(crate) topk_per_row: Vec<TopK>,
    #[allow(dead_code)]
    pub(crate) norms: Vec<f64>,
}

impl SparseTopKIndex {
    pub fn neighbors(&self, row: UserId) -> Vec<SimilarRow> {
        if let Some(heap) = self.topk_per_row.get(row as usize) {
            heap.iter().cloned().collect_vec()
        } else {
            Vec::new()
        }
    }

    fn parallel_topk<S: Similarity + Sync>(
        representations: &CsMat<f64>,
        representations_transposed: &CsMat<f64>,
        k: usize,
        norms: &Vec<f64>,
        similarity: &S,
    ) -> Vec<TopK> {
        let (num_rows, _) = representations.shape();

        let data = representations.data();
        let indices = representations.indices();
        let data_t = representations_transposed.data();
        let indices_t = representations_transposed.indices();
        let indptr_sprs = representations.indptr();
        let indptr_t_sprs = representations_transposed.indptr();
        let mut topk_per_row: Vec<TopK> = vec![TopK::new(BinaryHeap::new()); num_rows];

        let row_range = (0..num_rows).collect::<Vec<usize>>();
        // let row_ranges = row_range.par_chunks(1024);  // multi threaded
        let row_ranges = row_range.chunks(1024); // single thread

        let topk_partitioned: Vec<_> = row_ranges
            .map(|range| {
                let indptr = indptr_sprs.raw_storage();
                let indptr_t = indptr_t_sprs.raw_storage();

                // We need to get rid of these allocations and do them only once per thread
                let mut topk_per_row: Vec<TopK> = Vec::with_capacity(range.len());
                let mut accumulator = RowAccumulator::new(num_rows);

                for row in range {
                    let ptr_start = unsafe { *indptr.get_unchecked(*row) };
                    let ptr_end = unsafe { *indptr.get_unchecked(*row + 1) };

                    for ptr in ptr_start..ptr_end {
                        let value = unsafe { *data.get_unchecked(ptr) };

                        let other_ptr_start =
                            unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr)) };
                        let other_ptr_end =
                            unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr) + 1) };

                        for other_ptr in other_ptr_start..other_ptr_end {
                            let index = unsafe { *indices_t.get_unchecked(other_ptr) };
                            let value_t = unsafe { *data_t.get_unchecked(other_ptr) };
                            accumulator.add_to(index, value_t * value);
                        }
                    }

                    let topk = accumulator.topk_and_clear(*row, k, similarity, norms);
                    topk_per_row.push(topk);
                }
                (range, topk_per_row)
            })
            .collect();

        for (range, topk_partition) in topk_partitioned.into_iter() {
            for (index, topk) in range.iter().zip(topk_partition.into_iter()) {
                topk_per_row[*index] = topk;
            }
        }
        topk_per_row
    }

    pub fn new(representations: CsMat<f64>, k: usize) -> Self {
        let (num_rows, _) = representations.shape();

        // eprintln!("--Creating transpose of R...");
        let mut representations_transposed: CsMat<f64> = representations.to_owned();
        representations_transposed.transpose_mut();
        representations_transposed = representations_transposed.to_csr();

        let similarity = JACCARD;

        // eprintln!("--Computing row norms...");
        let norms: Vec<f64> = (0..num_rows)
            .map(|row| {
                let mut norm_accumulator: f64 = 0.0;
                for column_index in representations.indptr().outer_inds_sz(row) {
                    let value = representations.data()[column_index];
                    norm_accumulator += similarity.accumulate_norm(value);
                }
                similarity.finalize_norm(norm_accumulator)
            })
            .collect();
        // eprintln!("num_rows: {:?}", num_rows);

        // eprintln!("--Top-k computation...");
        let topk_per_row = SparseTopKIndex::parallel_topk(
            &representations,
            &representations_transposed,
            k,
            &norms,
            &similarity,
        );

        Self {
            representations,
            representations_transposed,
            topk_per_row,
            norms,
        }
    }
}
