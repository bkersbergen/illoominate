// use std::collections::{HashMap, HashSet};
// use indicatif::ProgressBar;
// use itertools::Itertools;
// use crate::importance::evaluate_dataset;
// use crate::sessrec::metrics::{MetricConfig, MetricFactory};
// use crate::sessrec::metrics::product_info::ProductInfo;
// use crate::sessrec::types::{SessionDataset, SessionId};
// use crate::sessrec::vmisknn::VMISKNN;
//
// pub fn vmis_loo_importance(
//     train: &SessionDataset,
//     valid: &SessionDataset,
//     k: usize,
//     m: usize
// ) -> HashMap<SessionId, f64> {
//
//     let model = VMISKNN::fit_dataset(train, m, k, 1.0);
//
//     let product_info = ProductInfo::new(HashSet::new());
//     let config = MetricConfig {
//         importance_metric: crate::sessrec::metrics::MetricType::MRR,
//         evaluation_metrics: vec![crate::sessrec::metrics::MetricType::MRR],
//         length: 21,
//         mrr_alpha: 0.8,
//     };
//     let metric_factory = MetricFactory::new(&config, &product_info);
//
//     let evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(&model, &metric_factory, &valid);
//     let (metric_name, baseline_value) = evaluation_metrics.get(0).unwrap();
//     log::debug!("_metric_name: {:?}", metric_name);
//     log::debug!("baseline_value: {:?}", baseline_value);
//
//     let mut train_clone = SessionDataset::clone(train);
//
//     let mut values_leave_one_out = HashMap::with_capacity(train_clone.sessions.len());
//
//     // Get a separate vector containing copies of the keys
//     let keys: Vec<_> = train_clone.sessions.keys().cloned().sorted().collect();
//
//     log::debug!("Determine LOO importance values");
//     let pb = ProgressBar::new(keys.len() as u64);
//     for session_id in keys.into_iter() {
//
//         let removed = train_clone.sessions.remove(&session_id).unwrap();
//
//         let model = VMISKNN::fit_dataset(&train_clone, m, k, 1.0);
//
//         let evaluation_metrics: Vec<(String, f64)>  = evaluate_dataset(&model, &metric_factory, &valid);
//         let (_metric_name, metric_value) = evaluation_metrics.get(0).unwrap();
//         let contribution = baseline_value - metric_value;
//         // log::debug!("without {} : {:.4}", session_id, contribution);
//         values_leave_one_out.insert(session_id, contribution);
//
//         train_clone.sessions.insert(session_id, removed);
//         pb.inc(1);
//     }
//     pb.finish();
//     log::debug!("LOO values calculated!");
//
//     values_leave_one_out
// }
//
