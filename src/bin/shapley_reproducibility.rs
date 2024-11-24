// #![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]
//
// use std::collections::{HashMap, HashSet};
// use std::env;
// use std::time::{Duration, Instant};
// use std::fs::File;
// use std::io::{BufWriter, Write};
// use illoominate::importance::Importance;
//
// use illoominate::baselines::tmcshapley::tmc_shapley;
// use illoominate::sessrec::io;
// use illoominate::sessrec::types::{SessionDataset, SessionId};
// use env_logger;
// use log::logger;
// use rayon::{current_num_threads, ThreadPoolBuilder};
// use illoominate::importance::k_mc_shapley::KMcShapley;
// use illoominate::sessrec::io::read_data;
// use illoominate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
// use illoominate::sessrec::metrics::product_info::ProductInfo;
// use illoominate::sessrec::vmisknn::VMISKNN;
//
// const TRAIN_DATA_FILE: &str = "data/bolcom100k/train.csv";
// const VALID_DATA_FILE: &str = "data/bolcom100k/valid.csv";
//
// fn main() {
//     env::set_var("RUST_LOG", "debug");
//     env_logger::init();
//
//
//     let pool_single_thead = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
//
//     let pool_multi_thead = ThreadPoolBuilder::new().num_threads(0).build().unwrap();
//     println!("rayon multithreaded pool using {} threads", current_num_threads());
//
//
//     // for (k, m) in [(500, 500), (10, 100), (2, 5)] {
//     //     for seed in [42, 18102022, 12345] {
//     for (k, m) in [(50, 500)] {
//         for seed in [42] {
//
//             log::info!("\n\n----[SEED: {}, k: {}, m: {}]----------------------------", seed, k, m);
//
//             let model_and_data = create_model_metric_train_valid(k, m);
//
//             let error = 0.1;
//             let num_iterations = 1;
//
//             let start_time = Instant::now();
//             let mc_results =
//                 tmc_shapley(&model_and_data.train_dataset, &model_and_data.valid_dataset, false, k, m, error, num_iterations, seed);
//             let mc_duration = Instant::now() - start_time;
//
//             let start_time = Instant::now();
//             let tmc_results =
//                 tmc_shapley(&model_and_data.train_dataset, &model_and_data.valid_dataset, true, k, m, error, num_iterations, seed);
//             let tmc_duration = Instant::now() - start_time;
//
//             let mut kmc_st_results= HashMap::new();
//             let mut kmc_st_duration = Duration::new(0,0);
//             pool_single_thead.install(|| {
//                 let start_time = Instant::now();
//                 kmc_st_results =
//                     KMcShapley::new(error, num_iterations, seed).compute_importance(&model_and_data.model, &model_and_data.metric_factory, &model_and_data.train_dataset, &model_and_data.valid_dataset);
//                 kmc_st_duration = Instant::now() - start_time;
//             });
//
//             let mut kmc_mt_results= HashMap::new();
//             let mut kmc_mt_duration = Duration::new(0,0);
//             pool_multi_thead.install(|| {
//                 let start_time = Instant::now();
//                 kmc_mt_results =
//                     KMcShapley::new(error, num_iterations, seed).compute_importance(&model_and_data.model, &model_and_data.metric_factory, &model_and_data.train_dataset, &model_and_data.valid_dataset);
//                 kmc_mt_duration = Instant::now() - start_time;
//             });
//
//             log::info!("Duration MC-Shapley version: {:?}", mc_duration);
//             log::info!("Duration TMC-Shapley version: {:?}", tmc_duration);
//             log::info!("Duration K-MC-Shapley (ST): {:?}", kmc_st_duration);
//             log::info!("Duration K-MC-Shapley (MT): {:?}", kmc_mt_duration);
//
//             let mc_sum_of_importances: f64 = mc_results.values().sum();
//             let tmc_sum_of_importances: f64 = tmc_results.values().sum();
//             let kmc_st_sum_of_importances: f64 = kmc_st_results.values().sum();
//             let kmc_mt_sum_of_importances: f64 = kmc_mt_results.values().sum();
//
//             log::debug!("MC-Shapley_sum_of_importances: {:.5}", mc_sum_of_importances);
//             log::debug!("TMC-Shapley_sum_of_importances: {:.5}", tmc_sum_of_importances);
//             log::debug!("K-MC-Shapley_st_sum_of_importances: {:.5}", kmc_st_sum_of_importances);
//             log::debug!("K-MC-Shapley_mt_sum_of_importances: {:.5}", kmc_mt_sum_of_importances);
//
//             log::info!("Comparing MC-Shapley with MC-Shapley");
//             print_results(&mc_results, &mc_results);
//             log::info!("Comparing MC-Shapley with TMC-Shapley");
//             print_results(&mc_results, &tmc_results);
//             log::info!("Comparing MC-Shapley with K-MC-Shapley Single Threaded");
//             print_results(&mc_results, &kmc_st_results);
//             log::info!("Comparing MC-Shapley with K-MC-Shapley Multi Threaded");
//             print_results(&mc_results, &kmc_mt_results);
//             write_to_disk(&mc_results, "shapley_reproducibility_mc.csv");
//             write_to_disk(&tmc_results, "shapley_reproducibility_tmc.csv");
//             write_to_disk(&kmc_st_results, "shapley_reproducibility_kmc_st.csv");
//             write_to_disk(&kmc_mt_results, "shapley_reproducibility_kmc_mt.csv");
//         }
//     }
// }
//
// fn print_results(baseline: &HashMap<SessionId, f64>, alternative: &HashMap<SessionId, f64>) {
//     // Convert the hash map into a vector of key-value pairs and sort it by keys
//     let mut baseline_vec: Vec<_> = baseline.iter().collect();
//     baseline_vec.sort_by_key(|&(k, _)| k);
//
//     let mut differences = Vec::new();
//
//     for (key, baseline_value) in baseline_vec {
//         let alternative_value = alternative.get(key).unwrap_or_else(|| &0.0);
//         let difference = (baseline_value - alternative_value).abs();
//         differences.push(difference);
//     }
//
//     // Sort the differences to compute percentiles
//     differences.sort_by(|a, b| a.partial_cmp(b).unwrap());
//
//     let qty_total = differences.len();
//
//     let p10 = percentile(&differences, qty_total, 10);
//     let p50 = percentile(&differences, qty_total, 50);
//     let p90 = percentile(&differences, qty_total, 90);
//     let p99 = percentile(&differences, qty_total, 99);
//     let p100 = percentile(&differences, qty_total, 100);
//
//     println!("Percentiles of differences:");
//     println!("p10: {}", p10);
//     println!("p50 (median): {}", p50);
//     println!("p90: {}", p90);
//     println!("p99: {}", p99);
//     println!("p100 (max): {}", p100);
// }
//
// fn percentile(differences: &Vec<f64>, qty_total: usize, percentile: usize) -> f64 {
//     if qty_total == 0 {
//         return 0.0;
//     }
//     let rank = (percentile * (qty_total - 1)) as f64 / 100.0;
//     let lower = rank.floor() as usize;
//     let upper = rank.ceil() as usize;
//
//     if lower == upper {
//         differences[lower]
//     } else {
//         differences[lower] * (upper as f64 - rank) + differences[upper] * (rank - lower as f64)
//     }
// }
//
//
// pub fn write_to_disk(importances: &HashMap<SessionId, f64>, output_filename: &str) {
//     // Open a file in write mode
//     let file = match File::create(output_filename) {
//         Ok(file) => file,
//         Err(err) => {
//             eprintln!("Error: Unable to create file '{}': {}", output_filename, err);
//             return;
//         }
//     };
//     let mut writer = BufWriter::new(file);
//
//     // Write each key-value pair to the CSV file
//     for (key, value) in importances {
//         if let Err(err) = writeln!(writer, "{},{}", key, value) {
//             eprintln!("Error: Unable to write to file '{}': {}", output_filename, err);
//             return;
//         }
//     }
//
//     println!("CSV file '{}' exported successfully!", output_filename);
// }
//
//
//
// fn create_model_metric_train_valid(k: usize, m: usize) -> ModelAndData {
//     let train_dataset = SessionDataset::new(read_data(&TRAIN_DATA_FILE.to_string()));
//     let valid_dataset = SessionDataset::new(read_data(&VALID_DATA_FILE.to_string()));
//
//     // Box the config and product_info to allocate them on the heap
//     let config = Box::new(MetricConfig {
//         importance_metric: MetricType::MRR,
//         evaluation_metrics: vec![MetricType::MRR],
//         length: 21,
//         mrr_alpha: 0.8,
//     });
//
//     let product_info = Box::new(ProductInfo::new(HashSet::new()));
//
//     // Leak the boxed values to convert them into `'static` references
//     let metric_factory = MetricFactory::new(Box::leak(config), Box::leak(product_info));
//
//     let model = VMISKNN::fit_dataset(&train_dataset, m, k, 1.0);
//
//     ModelAndData {
//         model,
//         metric_factory,
//         train_dataset,
//         valid_dataset,
//     }
// }
//
// struct ModelAndData {
//     model: VMISKNN,
//     metric_factory: MetricFactory<'static>,
//     train_dataset: SessionDataset,
//     valid_dataset: SessionDataset,
// }
//
fn main() {}
