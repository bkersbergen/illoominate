// use std::cmp::max;
// use std::collections::{HashMap, HashSet};
// use std::env;
// use std::fs::OpenOptions;
// use std::fs::File;
// use std::io::Write;
// use chrono::{Local};
// use env_logger::Builder;
// use log::LevelFilter;
// use rand::seq::SliceRandom;
// use rand::seq::IteratorRandom;
// use illoominate::importance::{Dataset, evaluate_dataset, Importance};
// use illoominate::importance::k_mc_shapley::KMcShapley;
// use illoominate::sessrec::vmisknn::VMISKNN;
// use illoominate::sessrec::io;
// use illoominate::sessrec::types::{SessionId, SessionDataset, ItemId};
// use illoominate::importance::k_loo::KLoo;
// use rand::SeedableRng;
// use rand::rngs::StdRng;
// use illoominate::nbr::tifuknn::types::HyperParams;
// use illoominate::sessrec::io::read_sustainable_products_info;
// use illoominate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
// use illoominate::sessrec::metrics::product_info::ProductInfo;
//
//
// fn main() {
//     // Initialize the logger with timezone
//     Builder::new()
//         .filter_level(LevelFilter::Info)
//         .format(|buf, record| {
//             writeln!(
//                 buf,
//                 "[{} {}] {} {}",
//                 Local::now().format("%Y-%m-%dT%H:%M:%S"),
//                 record.module_path().unwrap_or_else(|| "-".into()),
//                 record.level(),
//                 record.args()
//             )
//         })
//         .init();
//
//
//     let data_location = env::var("DATA_LOCATION").unwrap();
//     // let data_location = "result";
//
//     log::info!("using DATA_LOCATION: {data_location}");
//     // let data_path = "experiments/datasets/retailrocket";
//     // let data_path = "experiments/datasets/ecom1m/bolcom-clicks-1m_train_full 20231010_05_05";
//
//     let config = MetricConfig {
//         importance_metric: MetricType::MRR,
//         evaluation_metrics: vec![MetricType::MRR],
//         length: 20,
//         mrr_alpha: 0.8,
//     };
//     run_importance_experiment_for(&data_location, &config);
//     //
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::SustainabilityCoverage,
//     //     evaluation_metrics: vec![MetricType::SustainabilityCoverage, MetricType::MRR, MetricType::ResponsibleMrr],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     // run_importance_experiment_for(&data_path, &config);
//     //
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::ResponsibleMrr,
//     //     evaluation_metrics: vec![MetricType::ResponsibleMrr, MetricType::MRR, MetricType::SustainabilityCoverage],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     // run_importance_experiment_for(&data_path, &config);
//
// }
//
//
// fn run_importance_experiment_for(data_path: &str, config: &MetricConfig) {
//     let sustainable_products: HashSet<ItemId> = if config.evaluation_metrics.contains(&MetricType::ResponsibleMrr) || config.evaluation_metrics.contains(&MetricType::SustainabilityCoverage) {
//         read_sustainable_products_info(&format!("{}/__sustainable_mapped_items.csv.csv", data_path))
//     } else {
//         HashSet::new()
//     };
//     // // let sustainable_products: HashSet<ItemId> = read_sustainable_products_info(&format!("{}/__sustainable_mapped_items.csv.csv", data_path));
//     // let sustainable_products: HashSet<ItemId> = HashSet::new();
//     let product_info = ProductInfo::new(sustainable_products);
//     let metric_factory = MetricFactory::new(&config, &product_info);
//
//     let importance_metric_binding = metric_factory.create_importance_metric();
//
//     let importance_metric_friendly = importance_metric_binding.as_ref().get_name().to_lowercase();
//     let mut output_file_shapley_importances =
//         create_file(&format!("{}/__removal_impact_shapley_importance_{}_eval_{}.csv", data_path, importance_metric_friendly, importance_metric_friendly));
//     let mut output_file_loo_importances =
//         create_file(&format!("{}/__removal_impact_loo_importance_{}_eval_{}.csv", data_path, importance_metric_friendly, importance_metric_friendly));
//
//     let mut output_files_evaluation_metric_results: Vec<File> = metric_factory
//         .create_evaluation_metrics()
//         .iter_mut()
//         .map(|metric| {
//             let metric_friendly_name = metric.get_name().to_lowercase();
//             create_file(&format!(
//                 "{}/__removal_impact_results_importance_{}_eval_{}.csv",
//                 data_path, importance_metric_friendly, metric_friendly_name
//             ))
//         })
//         .collect();
//
//     let train = SessionDataset::new(io::read_data(&format!("{}/train.csv", data_path)));
//     let valid = SessionDataset::new(io::read_data(&format!("{}/valid.csv", data_path)));
//     let test = SessionDataset::new(io::read_data(&format!("{}/test.csv", data_path)));
//
//     let k = 50;
//     let m = 500;
//     let qty_impact_resolution = 250;
//     let seed = 1313;
//
//     let model = VMISKNN::fit_dataset(&train, m, k, 1.0);
//     let loo_values = KLoo::new().compute_importance(&model, &metric_factory, &train, &valid);
//
//     let mut loo_indices_sorted_by_importance: Vec<(u32, f64)> = loo_values.clone()
//         .into_iter().collect();
//     loo_indices_sorted_by_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//
//     for seed in [seed, 18102022, 12345, 1311, 128] {
//         log::info!("determine removal impact with seed: {}", seed);
//
//         for (index, importance) in &loo_indices_sorted_by_importance {
//             let data_to_append = format!("{},{},{:.10}", seed, index, importance);
//             writeln!(output_file_loo_importances, "{}", data_to_append)
//                 .expect("Failed to write to file");
//         }
//         let important_first_loo = positive_by_importance(&loo_values);
//         evaluate_removal_impact("important_first_loo", &metric_factory, train.clone(), &valid, &test,
//                                 &important_first_loo, m, k, seed, qty_impact_resolution,
//                                 &mut output_files_evaluation_metric_results);
//
//         let least_first_loo = negative_by_importance_reverse(&loo_values);
//         evaluate_removal_impact("least_first_loo", &metric_factory, train.clone(), &valid, &test,
//                                 &least_first_loo, m, k, seed, qty_impact_resolution,
//                                 &mut output_files_evaluation_metric_results);
//
//         let error = 0.1;
//         let num_iterations = 100;
//
//         // let model = VMISKNN::fit_dataset(&train, m, k, 1.0);
//
//         let tmc_shapley = KMcShapley::new(error, num_iterations, seed);
//
//         let shapley_values = tmc_shapley.compute_importance(&model, &metric_factory, &train, &valid);
//
//         let mut indices_sorted_by_importance: Vec<(u32, f64)> = shapley_values.clone()
//             .into_iter().collect();
//         indices_sorted_by_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//
//         for (index, importance) in &indices_sorted_by_importance {
//             let data_to_append = format!("{},{},{:.10}", seed, index, importance);
//             writeln!(output_file_shapley_importances, "{}", data_to_append)
//                 .expect("Failed to write to file");
//         }
//
//
//         let important_first_shapley = positive_by_importance(&shapley_values);
//         evaluate_removal_impact("important_first_shapley", &metric_factory, train.clone(), &valid, &test,
//                                 &important_first_shapley, m, k, seed, qty_impact_resolution,
//                                 &mut output_files_evaluation_metric_results);
//
//
//         let least_first_shapley = negative_by_importance_reverse(&shapley_values);
//         evaluate_removal_impact("least_first_shapley", &metric_factory, train.clone(), &valid, &test,
//                                 &least_first_shapley, m, k, seed, qty_impact_resolution,
//                                 &mut output_files_evaluation_metric_results);
//
//         let num_random_sessions_to_remove =
//             *[important_first_shapley.len(),
//                 least_first_shapley.len(),
//                 important_first_loo.len(),
//                 least_first_loo.len(),
//             ].iter().max().unwrap();
//
//         let mut rng = StdRng::seed_from_u64((seed) as u64);
//         let mut random_sessions_to_remove: Vec<u32> = shapley_values.clone()
//             .into_iter()
//             .choose_multiple(&mut rng, num_random_sessions_to_remove)
//             .into_iter()
//             .map(|(index, _)| index)
//             .collect();
//
//         random_sessions_to_remove.shuffle(&mut rng);
//
//         evaluate_removal_impact("random", &metric_factory, train.clone(), &valid, &test,
//                                 &random_sessions_to_remove, m, k, seed, qty_impact_resolution,
//                                 &mut output_files_evaluation_metric_results);
//
//     }
//
//
// }
//
// fn evaluate_removal_impact<D: Dataset>(
//     exp: &str,
//     metric_factory: &MetricFactory,
//     mut train: SessionDataset,
//     valid: &SessionDataset,
//     test: &SessionDataset,
//     sessions_to_remove: &[SessionId],
//     m: usize,
//     k: usize,
//     seed: usize,
//     qty_impact_resolution: usize,
//     output_files_evaluation_metric_results: &mut Vec<File>,
// ) {
//
//     assert!(train.sessions.len() >= sessions_to_remove.len());
//
//     let retrained_model = VMISKNN::fit_dataset(&train, m, k, 1.0);
//
//     let validation_evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(&retrained_model, &metric_factory, &valid);
//     let test_evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(&retrained_model, &metric_factory, &test);
//
//     for ((output_file, (_valid_metric_name, valid_metric_score)), (_test_metric_name, test_metric_score)) in output_files_evaluation_metric_results.iter_mut().zip(validation_evaluation_metrics.iter()).zip(test_evaluation_metrics.iter()) {
//         let data_to_append = format!(
//             "{},{},{:.4},{:.4},{}",
//             exp,
//             seed,
//             valid_metric_score,
//             test_metric_score,
//             0
//         );
//         log::info!("{}", data_to_append);
//         writeln!(output_file, "{}", data_to_append)
//             .expect("Results Failed to write to file");
//     }
//
//     let mut num_sessions_removed = 0;
//
//     let impact_interval = max(1, (sessions_to_remove.len() as f64 / qty_impact_resolution as f64) as usize);
//     for session_index in sessions_to_remove {
//
//         // Remove SessionId from the Training data
//         if let Some(_) = train.sessions.remove(&(*session_index as SessionId)) {
//             num_sessions_removed += 1;
//             let compute_impact = (train.sessions.len() == 1) || (num_sessions_removed % impact_interval == 0);
//             if compute_impact {
//                 // evaluate on validation data and write output
//                 let retrained_model = VMISKNN::fit_dataset(&train, m, k, 1.0);
//
//                 let validation_evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(&retrained_model, &metric_factory, &valid);
//                 let test_evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(&retrained_model, &metric_factory, &test);
//
//                 for ((output_file, (_valid_metric_name, valid_metric_score)), (_test_metric_name, test_metric_score)) in output_files_evaluation_metric_results.iter_mut().zip(validation_evaluation_metrics.iter()).zip(test_evaluation_metrics.iter()) {
//                     let data_to_append = format!(
//                         "{},{},{:.4},{:.4},{}",
//                         exp,
//                         seed,
//                         valid_metric_score,
//                         test_metric_score,
//                         num_sessions_removed
//                     );
//                     log::info!("{}", data_to_append);
//                     writeln!(output_file, "{}", data_to_append)
//                         .expect("Results Failed to write to file");
//                 }
//
//             }
//
//         };
//     }
// }
//
// fn positive_by_importance(original_importances: &HashMap<u32, f64>) -> Vec<u32> {
//     let mut importances: Vec<(u32, f64)> = original_importances.clone().into_iter().collect();
//     importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//     importances.into_iter()
//         .filter(|(_, importance)| *importance > 0.0)
//         .map(|(session_id, _)| session_id)
//         .collect()
// }
//
// fn negative_by_importance_reverse(original_importances: &HashMap<u32, f64>) -> Vec<u32> {
//     let mut importances: Vec<(u32, f64)> = original_importances.clone().into_iter().collect();
//     importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//     importances.into_iter()
//         .filter(|(_, importance)| *importance < 0.0)
//         .map(|(session_id, _)| session_id)
//         .collect()
// }
//
// fn create_file(name: &str) -> File {
//     OpenOptions::new()
//         .create_new(true)
//         .write(true)
//         .append(false)
//         .open(name)
//         .expect(&format!("Failed to create file: {}", name))
// }

fn main() {}
