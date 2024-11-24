// use std::collections::{HashMap, HashSet};
// use indicatif::{ProgressBar, ProgressStyle};
// use itertools::enumerate;
// use crate::baselines::ExperimentPayload;
// use crate::importance::{evaluate_dataset, tmc_utils};
// use crate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
// use crate::sessrec::metrics::product_info::ProductInfo;
// use crate::sessrec::types::{SessionDataset, SessionId};
// use crate::sessrec::vmisknn::VMISKNN;
//
// pub fn tmc_shapley(
//     experiment_payload: &ExperimentPayload, truncation_enabled: bool
// ) -> Vec<f64> {
//     let (bootstrap_mean_score, bootstrap__tolerance) = if truncation_enabled {
//         let model = VMISKNN::fit_dataset(&experiment_payload.train_dataset, experiment_payload.vmis_m, experiment_payload.vmis_k, 1.0);
//         tmc_utils::mean_tolerance_score_dataset(&model, &experiment_payload.metric_factory, &experiment_payload.valid_dataset)
//     } else {
//         (0.0, 0.0)
//     };
//     let mut marginal_contribs = Vec::new();
//     for iteration in 0..experiment_payload.monte_carlo_iterations {
//         marginal_contribs = one_iteration_dataset(experiment_payload, truncation_enabled, bootstrap_mean_score, bootstrap__tolerance, iteration);
//     }
//     marginal_contribs
// }
//
// #[allow(non_snake_case)]
// pub fn one_iteration_dataset(
//     experiment_payload: &ExperimentPayload,
//     truncation_enabled: bool,
//     bootstrap_mean_score: f64,
//     bootstrap__tolerance: f64,
//     iteration: usize,
// ) -> Vec<f64> {
//     let mut X_batch = SessionDataset::new(Vec::with_capacity(experiment_payload.train_dataset.sessions.len()));
//     let permutation: Vec<SessionId> = tmc_utils::permutation(&experiment_payload.train_dataset, experiment_payload.seed, iteration);
//
//     let max_id = *permutation.iter().max().unwrap_or(&0);
//     // because of truncation stoppage we assign a default importance of 0.0
//     let mut contributions: Vec<f64> = vec![0.0; (max_id + 1) as usize];
//
//     let mut old_score = experiment_payload.random_score;
//
//     let mut truncation_counter = 0;
//     let bar = ProgressBar::new(permutation.len() as u64);
//     bar.set_style(
//         ProgressStyle::default_bar()
//             .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap(),
//     );
//     for (n, &session_idx) in enumerate(&permutation) {
//
//         let payload = experiment_payload.train_dataset.sessions.get(&session_idx).unwrap();
//
//         X_batch.sessions.insert(session_idx, payload.clone());
//
//         let model = VMISKNN::fit_dataset(&X_batch, experiment_payload.vmis_m, experiment_payload.vmis_k, 1.0);
//         let evaluation_metrics: Vec<(String, f64)>  = evaluate_dataset(&model, &experiment_payload.metric_factory, &experiment_payload.valid_dataset);
//         let (_metric_name, mrr_result) = evaluation_metrics.get(0).unwrap();
//         let new_score = mrr_result;
//         let u_q = new_score - old_score;
//         old_score = *new_score;
//         contributions[session_idx as usize] = u_q;
//         if truncation_enabled {
//             if (new_score - bootstrap_mean_score).abs() <= bootstrap__tolerance * bootstrap_mean_score {
//                 truncation_counter += 1;
//                 if truncation_counter > 5 {
//                     log::info!("Truncation stoppage at iteration: {} / {}", n, &contributions.len());
//                     break;
//                 }
//             } else {
//                 truncation_counter = 0
//             }
//         }
//         bar.inc(1);
//     }
//     contributions
// }
