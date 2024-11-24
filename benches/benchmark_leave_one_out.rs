// use std::collections::{HashMap, HashSet};
// use criterion::{Criterion, criterion_group, criterion_main};
// use illoominate::baselines::loo::vmis_loo_importance;
// use illoominate::baselines::tmcshapley::tmc_shapley;
// use illoominate::importance::Importance;
// use illoominate::importance::k_loo::KLoo;
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
//     targets = k_loo_benchmark, loo_benchmark, mc_shapley_benchmark
// }
//
// const TRAIN_DATA_FILE: &str = "experiments/datasets/retailrocket/train.csv";
// const VALID_DATA_FILE: &str = "experiments/datasets/retailrocket/valid.csv";
//
// const K: usize = 50;
// const M: usize = 500;
//
// fn mc_shapley_benchmark(c: &mut Criterion) {
//     let data = create_model_metric_train_valid();
//     c.bench_function("MC-Shapley (baseline)", |b| b.iter(|| mc_shapley_compute(&data)));
// }
//
// fn mc_shapley_compute(data: &ModelAndData) -> HashMap<SessionId, f64> {
//     tmc_shapley(&data.train_dataset, &data.valid_dataset, K, M)
// }
//
// fn loo_benchmark(c: &mut Criterion) {
//     let data = create_model_metric_train_valid();
//     c.bench_function("LOO (baseline)", |b| b.iter(|| loo_compute(&data)));
// }
//
// fn loo_compute(data: &ModelAndData) -> HashMap<SessionId, f64> {
//     vmis_loo_importance(&data.train_dataset, &data.valid_dataset, K, M)
// }
//
// fn k_loo_benchmark(c: &mut Criterion) {
//     let data = create_model_metric_train_valid();
//     c.bench_function("K-LOO (ours)", |b| b.iter(|| k_loo_compute(&data)));
// }
//
// fn k_loo_compute(data: &ModelAndData) -> HashMap<SessionId, f64> {
//     KLoo::new().compute_importance(&data.model, &data.metric_factory, &data.train_dataset, &data.valid_dataset)
// }
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
//     // No need to store _config and _product_info separately anymore
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
