// use std::marker::{Send, Sync};
//
// use rayon::prelude::*;
//
// use crate::baselines::ExperimentPayload;
// use crate::importance::{num_evaluations, RetrievalBasedModel, score_session_contributions_parallel, tmc_utils};
// use crate::importance::candidate_neighbors::CandidateNeighbors;
// use crate::sessrec::types::SessionId;
// use crate::sessrec::vmisknn::{SessionScore, VMISKNN};
//
// pub fn kmc_shapley_agg(experiment_payload: &ExperimentPayload) -> Vec<f64> {
//     let mut marginal_contribs = Vec::new();
//     let max_m = experiment_payload.train_dataset.sessions.len();
//     let k = experiment_payload.vmis_k;
//     let all_index_model = VMISKNN::fit_dataset(&experiment_payload.train_dataset, max_m, k, 1.0);
//     for iteration in 0..experiment_payload.monte_carlo_iterations {
//         marginal_contribs = one_iteration_dataset(experiment_payload, iteration, &all_index_model);
//     }
//     marginal_contribs
// }
//
//
// #[allow(non_snake_case)]
// pub fn one_iteration_dataset<R: RetrievalBasedModel + Send + Sync>(
//     experiment_payload: &ExperimentPayload,
//     iteration: usize,
//     all_index_model: &R,
// ) -> Vec<f64> {
//     let training_session_ids_permuted: Vec<SessionId> = tmc_utils::permutation(&experiment_payload.train_dataset, experiment_payload.seed, iteration);
//
//     let mut permutation_index = vec![0; experiment_payload.train_dataset.sessions.len()];
//     for (idx, &value) in training_session_ids_permuted.iter().enumerate() {
//         permutation_index[value as usize] = idx;
//     }
//     let mut contributions: Vec<f64> = score_session_contributions_parallel(
//         &experiment_payload.valid_dataset,
//         experiment_payload.train_dataset.sessions.len(),
//         |query_session, actual_next_items, contributions| {
//             let mut N_q: Vec<SessionScore> = all_index_model.retrieve_all(query_session);
//             N_q.sort_by_key(|session_score| permutation_index[session_score.id as usize]);
//
//             let mut candidate_neighbors = CandidateNeighbors::new(all_index_model.k());
//             let mut prev_score = 0.0;
//
//             let metric_binding = experiment_payload.metric_factory.create_importance_metric();
//             let metric = metric_binding.as_ref();
//
//             let mut agg = all_index_model.create_preaggregate();
//
//             for similar_session in N_q {
//                 let (topk_updated, dropped_out) = candidate_neighbors.offer(similar_session);
//                 if topk_updated {
//
//                     &all_index_model.add_to_preaggregate(&mut agg, &query_session, &similar_session);
//
//                     if dropped_out.is_some() {
//                         &all_index_model.remove_from_preaggregate(&mut agg, &query_session, &dropped_out.unwrap());
//                     }
//
//                     //let neighbors: Vec<_> = candidate_neighbors.iter().collect();
//
//                     let recommended_items = &all_index_model.generate_from_preaggregate(query_session, &agg);
//
//                     let metric_result = metric.compute(&recommended_items, &actual_next_items);
//
//                     let new_score = metric_result - prev_score;
//                     prev_score = metric_result;
//                     contributions[similar_session.id as usize] += new_score
//                 }
//             }
//         },
//     );
//
//     let qty_evaluations = num_evaluations(&experiment_payload.valid_dataset);
//
//     // Parallelize the normalization of contributions
//     contributions.par_iter_mut().for_each(|contribution| {
//         *contribution /= qty_evaluations as f64;
//     });
//
//     // subtract the random score from the first session in the permuted list
//     contributions[training_session_ids_permuted[0] as usize] -= experiment_payload.random_score;
//     contributions
// }
//
