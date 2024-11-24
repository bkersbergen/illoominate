// use std::collections::hash_map::Entry;
// use std::collections::{BinaryHeap, HashMap, HashSet};
// use std::hash::Hash;
//
// use average::{Estimate};
// use indicatif::ProgressBar;
// use itertools::{enumerate, Itertools};
// use log::error;
// use rand::prelude::StdRng;
// use rand::SeedableRng;
// use rand::seq::SliceRandom;
// use crate::baselines::ExperimentPayload;
// use crate::importance::{Importance, num_evaluations, RetrievalBasedModel, tmc_utils};
// use crate::importance::candidate_neighbors::CandidateNeighbors;
// use crate::sessrec::metrics::MetricFactory;
// use crate::sessrec::types::{SessionDataset, SessionId};
// use crate::sessrec::vmisknn::{SessionScore, SimilarityComputationNew, VMISKNN};
// use crate::sessrec::vmisknn::topk::TopK;
//
// // pub fn mc_shapley_sparse(experiment_payload: &ExperimentPayload, top_opt_enabled: bool) -> Vec<f64> {
// //     let mut marginal_contribs = Vec::new();
// //     for iteration in 0..experiment_payload.monte_carlo_iterations {
// //         marginal_contribs = one_iteration_dataset(experiment_payload, iteration, top_opt_enabled);
// //     }
// //     marginal_contribs
// // }
//
// #[allow(non_snake_case)]
// pub fn one_iteration_dataset<R: RetrievalBasedModel + Send + Sync>(experiment_payload: &ExperimentPayload, iteration: usize, top_opt_enabled: bool, all_index_model: &R,) -> Vec<f64> {
//     // let max_m = experiment_payload.train_dataset.sessions.keys().len();
//
//     let training_session_ids_permuted: Vec<SessionId> = tmc_utils::permutation(&experiment_payload.train_dataset, experiment_payload.seed, iteration);
//
//     let max_id = *training_session_ids_permuted.iter().max().unwrap_or(&0);
//     // because of truncation stoppage we assign a default importance of 0.0
//     let mut contributions: Vec<f64> = vec![0.0; (max_id + 1) as usize];
//     let metric_binding = experiment_payload.metric_factory.create_importance_metric();
//     let metric = metric_binding.as_ref();
//     experiment_payload.valid_dataset.sessions.iter().for_each(|(_session_id, session)| {
//         let (session_items, _max_timestamp) = session;
//         for pos in 1..session_items.len() {
//             let session = &session_items[0..pos].to_vec();
//             let actual_next_items = session_items[pos..].to_vec();
//             let N_q: HashMap<u32, SessionScore> = all_index_model.retrieve_all(session).iter().map(|x| (x.id, *x)).collect();
//             let mut candidates = CandidateNeighbors::new(experiment_payload.vmis_k);
//             let mut new_score = 0.0;
//             for &key_id in &training_session_ids_permuted {
//                 match &N_q.get(&(key_id as u32)) {
//                     Some(&scored_session) => {
//                         let (topk_changed, _) = candidates.offer(scored_session);
//                         if top_opt_enabled == false || (topk_changed == true && top_opt_enabled == true) {
//                             let neighbors: Vec<_> = candidates.iter().cloned().collect();
//                             let recommended_items = &all_index_model.generate(session, &neighbors);
//                             new_score = metric.compute(&recommended_items, &actual_next_items);
//                         }
//                     }
//                     None => {
//                         // training session idx not in neighbors
//                         // No need to determine closest_neighbors.
//                     }
//                 };
//                 // log::debug!("actual:{:?}", &actual_next_items);
//                 contributions[key_id as usize] += new_score;
//             }
//         }
//     });
//
//     let qty_evaluations: usize = num_evaluations(&experiment_payload.valid_dataset);
//
//     let mut old_score = experiment_payload.random_score;
//     for key_id in training_session_ids_permuted.iter() {
//         let new_score = contributions[*key_id as usize] / (qty_evaluations as f64);
//         let u_q = new_score - old_score;
//         old_score = new_score;
//         contributions[*key_id as usize] = u_q;
//     }
//     log::debug!("----------------------------------------------------------------nested_loops------------------------------------------------------------------------------------");
//     contributions
// }
//
//
