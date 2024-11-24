// use std::collections::{HashMap, HashSet};
// use criterion::{Criterion, criterion_group, criterion_main};
// use illoominate::baselines::loo::vmis_loo_importance;
// use illoominate::baselines::tmcshapley;
// use illoominate::importance::Importance;
// use illoominate::importance::k_loo::KLoo;
// use illoominate::importance::k_mc_shapley::KMcShapley;
// use illoominate::sessrec::io::read_data;
// use illoominate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
// use illoominate::sessrec::metrics::product_info::ProductInfo;
// use illoominate::sessrec::types::{SessionDataset, SessionId};
// use illoominate::sessrec::vmisknn::VMISKNN;
//
// criterion_main!(benches);
// criterion_group!{
//     name = benches;
//     config = Criterion::default()
//     .significance_level(0.1)
//     .output_directory("benchmark_results".as_ref())
//     .sample_size(10)
//     .measurement_time(std::time::Duration::from_secs(30));
//     targets = mc_shapley_benchmark, tmc_shapley_benchmark
// }
//
// // const TRAIN_DATA_FILE: &str = "experiments/datasets/retailrocket/train.csv";
// // const VALID_DATA_FILE: &str = "experiments/datasets/retailrocket/valid.csv";
//
// const TRAIN_DATA_FILE: &str = "data/bolcom100k/train.csv";
// const VALID_DATA_FILE: &str = "data/bolcom100k/valid.csv";
//
// const K: usize = 50;
// const M: usize = 500;
//
// // const NUM_ITERATIONS: usize = 100;
//
// // const SEED: usize = 42;
// // This benchmark would take Warning: Unable to complete 10 samples in 30.0s. You may wish to increase target time to 31244.1s.
// // Benchmarking MC Shapley (baseline): Collecting 10 samples in estimated  31244 s (10 iterations)
// // For MC Shapley alone.
//
//
// fn mc_shapley_benchmark(c: &mut Criterion) {
//     let data = create_model_metric_train_valid();
//     c.bench_function("MC Shapley (baseline)", |b| b.iter(|| tmc_shapley_compute(&data, false)));
// }
//
//
// fn tmc_shapley_benchmark(c: &mut Criterion) {
//     let data = create_model_metric_train_valid();
//     c.bench_function("TMC Shapley (baseline)", |b| b.iter(|| tmc_shapley_compute(&data, true)));
// }
//
// fn tmc_shapley_compute(data: &ModelAndData, truncation_enabled: bool) -> HashMap<SessionId, f64> {
//     let error = 0.1;
//     let iterations = 10;
//     tmcshapley::tmc_shapley(&data.train_dataset, &data.valid_dataset, truncation_enabled, K, M, error, iterations, 1)
// }
// //
// // fn k_mc_benchmark(c: &mut Criterion) {
// //     let data = create_model_metric_train_valid();
// //     c.bench_function("K-MC-Shapley (ours)", |b| b.iter(|| k_mc_shapley_compute(&data)));
// // }
// //
// // fn k_mc_shapley_compute(data: &ModelAndData) -> HashMap<SessionId, f64> {
// //     let k_mc_shapley = KMcShapley::new(ERROR, NUM_ITERATIONS, SEED);
// //     k_mc_shapley.compute_importance(&data.model, &data.metric_factory, &data.train_dataset, &data.valid_dataset)
// // }
//
// fn create_model_metric_train_valid() -> ModelAndData {
//     let training_data_path = TRAIN_DATA_FILE.to_string();
//     let valid_data_file = VALID_DATA_FILE.to_string();
//
//     let train = read_data(&training_data_path);
//     let valid = read_data(&valid_data_file);
//
//     let train_dataset = SessionDataset::new(train);
//     let valid_dataset = SessionDataset::new(valid);
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
//     let model = VMISKNN::fit_dataset(&train_dataset, M, K, 1.0);
//
//     ModelAndData {
//         model,
//         metric_factory,
//         train_dataset,
//         valid_dataset,
//     }
// }
// struct ModelAndData {
//     model: VMISKNN,
//     metric_factory: MetricFactory<'static>,
//     train_dataset: SessionDataset,
//     valid_dataset: SessionDataset,
// }
//
//
// // use std::collections::{BinaryHeap, HashMap};
// // use std::env;
// // use std::time::Instant;
// // use Illoominate::io2;
// // use Illoominate::io2::{ItemId, SessionDataSet, TrainingSessionId};
// // use itertools::enumerate;
// // use rand::prelude::SliceRandom;
// // use Illoominate::importance::{Importance, localtmcshapley, localtmcshapley_opt, tmcshapley};
// // use Illoominate::importance::localtmcshapley::LocalTmcShapley;
// //
// // use Illoominate::importance::tmcshapley::TmcShapley;
// // use Illoominate::metrics::mrr::Mrr;
// // use Illoominate::sessrec::vmisknn::{SessionScore, SimilarityComputationNew, VMISKNN};
// // use Illoominate::stats::DataDictionary;
// //
// // fn main() {
// //     let n_iterations = 10;
// //
// //     let (train, valid) = create_train_test();
// //     let tolerance = 0.0;
// //     let mean_score = 0.0;
// //     let random_score = 0.1;
// //     let k = 500;
// //     let m= 500;
// //
// //     for i in 0..n_iterations {
// //         let start_time = Instant::now();
// //         let alternative = tmcshapley::one_iteration_dataset(&train, &valid, tolerance, mean_score, random_score, k, m)
// //         let alt_duration = Instant::now() - start_time;
// //         let start = Instant::now();
// //
// //
// //
// //
// //
// //     }
// //
// // }
// //
// //
// //
// // fn create_train_test() -> (SessionDataSet, SessionDataSet) {
// //     let training_data_path = "/Users/bkersbergen/datasets/bolcom/prepared/bolcom-clicks-1m_train_full.txt";
// //     let valid_data_file = "/Users/bkersbergen/datasets/bolcom/prepared/bolcom-clicks-1m_test.txt";
// //     // let training_data_path = "/Users/bkersbergen/datasets/bolcom/prepared/train.txt";
// //     // let valid_data_file = "/Users/bkersbergen/datasets/bolcom/prepared/test.txt";
// //
// //     let training_datapoints = io2::read_data(&training_data_path.to_string());
// //     let validation_datapoints = io2::read_data(&valid_data_file.to_string());
// //
// //     let data_dictionary = DataDictionary::fit(&vec![&training_datapoints, &validation_datapoints]);
// //     let mut indexed_datapoints_iter = data_dictionary.transform(&vec![&training_datapoints, &validation_datapoints]).into_iter();
// //
// //     let train_dataset = SessionDataSet::new(indexed_datapoints_iter.next().unwrap_or_default());
// //     let valid_dataset = SessionDataSet::new(indexed_datapoints_iter.next().unwrap_or_default());
// //     (train_dataset, valid_dataset)
// // }
// //
// // fn shapley_baseline() -> Vec<f64> {
// //     let (train, valid) = create_train_test();
// //     let tolerance = 0.0;
// //     let mean_score = 0.0;
// //     let random_score = 0.1;
// //     let k = 500;
// //     let m= 500;
// //     tmcshapley::one_iteration_dataset(&train, &valid, tolerance, mean_score, random_score, k, m)
// // }
// //
// // fn shapley_illoominate_no_opt() -> Vec<f64> {
// //     let (train, valid) = create_train_test();
// //     let tolerance = 0.0;
// //     let mean_score = 0.0;
// //     let random_score = 0.1;
// //     let k = 500;
// //     let m= 500;
// //     localtmcshapley::one_iteration_dataset(&train, &valid, tolerance, mean_score, random_score, k, m)
// // }
// //
// // fn shapley_illoominate_opt() -> Vec<f64> {
// //     let (train, valid) = create_train_test();
// //     let tolerance = 0.0;
// //     let mean_score = 0.0;
// //     let random_score = 0.1;
// //     let k = 500;
// //     let m= 500;
// //     localtmcshapley_opt::one_iteration_dataset(&train, &valid, tolerance, mean_score, random_score, k, m)
// // }
// //
// //
