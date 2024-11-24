use crate::nbr::caboose::topk::TopK;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SerializableSparseTopKIndex {
    rep_indptr: Vec<usize>,
    rep_indices: Vec<usize>,
    rep_data: Vec<f64>,
    rep_indptr_t: Vec<usize>,
    rep_indices_t: Vec<usize>,
    rep_data_t: Vec<f64>,
    topk_per_row: Vec<TopK>,
    k: usize,
    norms: Vec<f64>,
}
//
// pub fn serialize_to_file(index: SparseTopKIndex, filename: &str) {
//
//     let (rep_indptr, rep_indices, rep_data) = index.representations.into_raw_storage();
//     let (rep_indptr_t, rep_indices_t, rep_data_t) =
//         index.representations_transposed.into_raw_storage();
//
//     let ser = SerializableSparseTopKIndex {
//         rep_indptr,
//         rep_indices,
//         rep_data,
//         rep_indptr_t,
//         rep_indices_t,
//         rep_data_t,
//         topk_per_row: index.topk_per_row,
//         k: index.k,
//         norms: index.norms
//     };
//
//     let mut f = BufWriter::new(File::create(filename).unwrap());
//     serialize_into(&mut f, &ser).unwrap();
// }
//
// pub fn deserialize_from(num_rows: usize, num_cols: usize, filename: &str) -> SparseTopKIndex {
//     let file = File::open(filename).unwrap();
//     let reader = BufReader::new(file);
//     let ser: SerializableSparseTopKIndex = bincode::deserialize_from(reader).unwrap();
//
//     let representations =
//         CsMat::new((num_rows, num_cols), ser.rep_indptr, ser.rep_indices, ser.rep_data);
//     let representations_transposed =
//         CsMat::new((num_cols, num_rows), ser.rep_indptr_t, ser.rep_indices_t, ser.rep_data_t);
//
//     SparseTopKIndex {
//         representations,
//         representations_transposed,
//         topk_per_row: ser.topk_per_row,
//         k: ser.k,
//         norms: ser.norms,
//         empty: Default::default()
//     }
// }
