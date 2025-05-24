#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

use std::collections::{HashMap, HashSet};
use std::env;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{BufWriter, Write};
use illoominate::importance::{evaluate_dataset, Importance};

use illoominate::baselines::tmcshapley::{tmc_shapley, ExperimentPayload};
use illoominate::importance::mc_utils::random_score_dataset;
use illoominate::sessrec::io;
use illoominate::sessrec::types::{Interaction, SessionDataset, SessionId};
use env_logger;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::{current_num_threads, ThreadPoolBuilder};
use illoominate::importance::k_mc_shapley::KMcShapley;
use illoominate::metrics::{MetricConfig, MetricFactory, MetricType};
use illoominate::metrics::product_info::ProductInfo;
use illoominate::sessrec::io::read_data;
use illoominate::sessrec::vmisknn::VMISKNN;

const TRAIN_DATA_FILE: &str = "data/tafeng/processed/train.csv";
const VALID_DATA_FILE: &str = "data/tafeng/processed/valid.csv";

// --- NEW PARAMETERS FOR FAIRER COMPARISON ---
const TMC_MONTE_CARLO_TOTAL_ITERATIONS: usize = 5;
const KMC_PERMUTATIONS_PER_CHECK: usize = 10;

fn main() {
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    let pool_single_thread = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool_multi_thread = ThreadPoolBuilder::new().num_threads(0).build().unwrap(); // Uses all available threads

    log::info!("================================================================================");
    log::info!("          ILLOOMINATE SHAPLEY VALUE REPRODUCIBILITY BENCHMARK          ");
    log::info!("================================================================================");
    log::info!("Rayon multi-threaded pool configured to use {} threads.", current_num_threads());


    for (k_param, m_param) in [(50, 500)] { // Example parameters, adjust as needed
        for seed_val in [42usize] { // Example seeds, adjust as needed
            log::info!("\n================================================================================");
            log::info!(" STARTING BENCHMARK RUN - SEED: {}, k: {}, m: {}", seed_val, k_param, m_param);
            log::info!("================================================================================");

            let kmc_error_threshold = 0.1; // Define KMC error threshold

            log::info!("[Configuration]");
            log::info!("  Train Data File: {}", TRAIN_DATA_FILE);
            log::info!("  Validation Data File: {}", VALID_DATA_FILE);
            log::info!("  VMIS-kNN k: {}", k_param);
            log::info!("  VMIS-kNN m: {}", m_param);
            log::info!("  Seed: {}", seed_val);
            log::info!("  TMC Total Iterations: {}", TMC_MONTE_CARLO_TOTAL_ITERATIONS);
            log::info!("  KMC Permutations per Check: {}", KMC_PERMUTATIONS_PER_CHECK);
            log::info!("  KMC Error Threshold: {}", kmc_error_threshold);
            log::info!("--------------------------------------------------------------------------------");

            log::info!("[Setup] Creating model, metric factory, and datasets...");
            let model_and_data = create_model_metric_train_valid(k_param, m_param);
            log::info!("[Setup] Finished creating model and data.");

            let (calculated_random_score, _) = random_score_dataset(
                &model_and_data.model,
                &model_and_data.metric_factory,
                &model_and_data.valid_dataset
            );
            log::info!("[Setup] Calculated random_score for utility function: {:.5}", calculated_random_score);
            let eval_results = evaluate_dataset(&model_and_data.model,
                                                &model_and_data.metric_factory,
                                                &model_and_data.valid_dataset
            );
            for (name, score) in eval_results {
                log::info!("[Setup] Full model score on validation set for {}: {:.5}", name, score);
            }
            log::info!("--------------------------------------------------------------------------------");


            // --- TMC-Shapley (Vanilla, non-truncated) ---
            log::info!("[Running] Vanilla TMC-Shapley (non-truncated, {} iterations)...", TMC_MONTE_CARLO_TOTAL_ITERATIONS);
            let tmc_payload_mc = ExperimentPayload {
                train_dataset: model_and_data.train_dataset.clone(),
                valid_dataset: model_and_data.valid_dataset.clone(),
                metric_factory: model_and_data.metric_factory.clone(),
                vmis_m: m_param,
                vmis_k: k_param,
                monte_carlo_iterations: TMC_MONTE_CARLO_TOTAL_ITERATIONS,
                seed: seed_val,
                random_score: calculated_random_score,
            };
            let start_time_mc = Instant::now();
            let mc_results_vec = tmc_shapley(&tmc_payload_mc, false);
            log::debug!("[Debug] TMC (non-trunc) raw vector: {} elements, sum = {:.5}", mc_results_vec.len(), mc_results_vec.iter().sum::<f64>());
            log::debug!("[Debug] TMC (non-trunc) non-zero entries: {}", mc_results_vec.iter().filter(|x| **x != 0.0).count());
            let mc_duration = start_time_mc.elapsed();
            let mc_results = vec_to_hashmap_filtered(mc_results_vec);
            log::info!("[Finished] Vanilla TMC-Shapley (non-truncated). Duration: {:?}", mc_duration);

            // --- TMC-Shapley (Vanilla, truncated) ---
            log::info!("[Running] Vanilla TMC-Shapley (truncated, {} iterations)...", TMC_MONTE_CARLO_TOTAL_ITERATIONS);
            let tmc_payload_tmc = ExperimentPayload {
                train_dataset: model_and_data.train_dataset.clone(),
                valid_dataset: model_and_data.valid_dataset.clone(),
                metric_factory: model_and_data.metric_factory.clone(),
                vmis_m: m_param,
                vmis_k: k_param,
                monte_carlo_iterations: TMC_MONTE_CARLO_TOTAL_ITERATIONS,
                seed: seed_val,
                random_score: calculated_random_score,
            };
            let start_time_tmc = Instant::now();
            let tmc_results_vec = tmc_shapley(&tmc_payload_tmc, true);
            log::debug!("[Debug] TMC (trunc) raw vector: {} elements, sum = {:.5}", tmc_results_vec.len(), tmc_results_vec.iter().sum::<f64>());
            log::debug!("[Debug] TMC (trunc) non-zero entries: {}", tmc_results_vec.iter().filter(|x| **x != 0.0).count());
            let tmc_duration = start_time_tmc.elapsed();
            let tmc_results = vec_to_hashmap_filtered(tmc_results_vec);
            log::info!("[Finished] Vanilla TMC-Shapley (truncated). Duration: {:?}", tmc_duration);

            // --- KMC-Shapley (Single Threaded) ---
            log::info!("[Running] K-MC-Shapley (Single Threaded, {} perm/check)...", KMC_PERMUTATIONS_PER_CHECK);
            let mut kmc_st_results= HashMap::new();
            let mut kmc_st_duration = Duration::new(0,0);
            pool_single_thread.install(|| {
                let kmc_algorithm = KMcShapley::new(kmc_error_threshold, KMC_PERMUTATIONS_PER_CHECK, seed_val);
                let start_time = Instant::now();
                kmc_st_results = kmc_algorithm.compute_importance(
                    &model_and_data.model,
                    &model_and_data.metric_factory,
                    &model_and_data.train_dataset,
                    &model_and_data.valid_dataset
                );
                kmc_st_duration = start_time.elapsed();
            });
            log::info!("[Finished] K-MC-Shapley (Single Threaded). Duration: {:?}", kmc_st_duration);

            // --- KMC-Shapley (Multi Threaded) ---
            log::info!("[Running] K-MC-Shapley (Multi Threaded, {} perm/check)...", KMC_PERMUTATIONS_PER_CHECK);
            let mut kmc_mt_results= HashMap::new();
            let mut kmc_mt_duration = Duration::new(0,0);
            pool_multi_thread.install(|| {
                let kmc_algorithm = KMcShapley::new(kmc_error_threshold, KMC_PERMUTATIONS_PER_CHECK, seed_val);
                let start_time = Instant::now();
                kmc_mt_results = kmc_algorithm.compute_importance(
                    &model_and_data.model,
                    &model_and_data.metric_factory,
                    &model_and_data.train_dataset,
                    &model_and_data.valid_dataset
                );
                kmc_mt_duration = start_time.elapsed();
            });
            log::info!("[Finished] K-MC-Shapley (Multi Threaded). Duration: {:?}", kmc_mt_duration);
            log::info!("--------------------------------------------------------------------------------");

            log::info!("[Duration Summary]");
            log::info!("  Vanilla TMC-Shapley (non-truncated, {} iterations): {:?}", TMC_MONTE_CARLO_TOTAL_ITERATIONS, mc_duration);
            log::info!("  Vanilla TMC-Shapley (truncated,    {} iterations): {:?}", TMC_MONTE_CARLO_TOTAL_ITERATIONS, tmc_duration);
            log::info!("  K-MC-Shapley (Single Threaded, {} perm/check):   {:?}", KMC_PERMUTATIONS_PER_CHECK, kmc_st_duration);
            log::info!("  K-MC-Shapley (Multi Threaded,  {} perm/check):    {:?}", KMC_PERMUTATIONS_PER_CHECK, kmc_mt_duration);
            log::info!("--------------------------------------------------------------------------------");

            let mc_sum_of_importances: f64 = mc_results.values().sum();
            let tmc_sum_of_importances: f64 = tmc_results.values().sum();
            let kmc_st_sum_of_importances: f64 = kmc_st_results.values().sum();
            let kmc_mt_sum_of_importances: f64 = kmc_mt_results.values().sum();

            log::debug!("[Sum of Importances - Sanity Check]");
            log::debug!("  Vanilla TMC (non-trunc) sum: {:.5}", mc_sum_of_importances);
            log::debug!("  Vanilla TMC (trunc)     sum: {:.5}", tmc_sum_of_importances);
            log::debug!("  K-MC-Shapley (ST)       sum: {:.5}", kmc_st_sum_of_importances);
            log::debug!("  K-MC-Shapley (MT)       sum: {:.5}", kmc_mt_sum_of_importances);
            log::debug!("--------------------------------------------------------------------------------");


            log::info!("\n================================================================================");
            log::info!(" RESULT COMPARISON (vs Vanilla TMC non-truncated)");
            log::info!("================================================================================");

            log::info!("\n--- [Comparison] Vanilla TMC (non-trunc) vs. Vanilla TMC (non-trunc) ---");
            print_results("Vanilla TMC (non-trunc)", &mc_results, "Vanilla TMC (non-trunc)", &mc_results);

            log::info!("\n--- [Comparison] Vanilla TMC (non-trunc) vs. Vanilla TMC (trunc) ---");
            print_results("Vanilla TMC (non-trunc)", &mc_results, "Vanilla TMC (trunc)", &tmc_results);

            log::info!("\n--- [Comparison] Vanilla TMC (non-trunc) vs. K-MC-Shapley (ST) ---");
            print_results("Vanilla TMC (non-trunc)", &mc_results, "K-MC-Shapley (ST)", &kmc_st_results);

            log::info!("\n--- [Comparison] Vanilla TMC (non-trunc) vs. K-MC-Shapley (MT) ---");
            print_results("Vanilla TMC (non-trunc)", &mc_results, "K-MC-Shapley (MT)", &kmc_mt_results);

            log::info!("\n================================================================================");
            log::info!(" BENCHMARK RUN FINISHED - SEED: {}, k: {}, m: {}", seed_val, k_param, m_param);
            log::info!("================================================================================\n\n");
        }
    }
    log::info!("================================================================================");
    log::info!("                 ALL BENCHMARK RUNS COMPLETED                 ");
    log::info!("================================================================================");
}

fn vec_to_hashmap_filtered(scores_vec: Vec<f64>) -> HashMap<SessionId, f64> {
    let mut filtered_map = HashMap::new();
    let mut dropped = 0;
    for (id, score) in scores_vec.into_iter().enumerate().map(|(id, score)| (id as SessionId, score)) {
        if score != 0.0 {
            filtered_map.insert(id, score);
        } else {
            dropped += 1;
        }
    }
    log::debug!("[vec_to_hashmap_filtered] Dropped {} zero-valued scores", dropped);
    filtered_map
}

fn print_results(baseline_name: &str, baseline: &HashMap<SessionId, f64>,
                 alternative_name: &str, alternative: &HashMap<SessionId, f64>) {
    let mut baseline_vec: Vec<_> = baseline.iter().collect();
    baseline_vec.sort_by_key(|&(k, _)| k);

    let mut differences = Vec::new();
    let mut count_alternative_missing = 0;
    let mut count_baseline_missing_in_alt = 0;
    let mut common_keys_count = 0;

    for (&key, baseline_value) in baseline_vec.iter() {
        if let Some(alternative_value) = alternative.get(&key) {
            let difference = (*baseline_value - *alternative_value).abs();
            differences.push(difference);
            common_keys_count += 1;
        } else {
            // Baseline key not found in alternative. Consider its absolute value as difference.
            differences.push(baseline_value.abs());
            count_alternative_missing += 1;
        }
    }

    for &key_alt in alternative.keys() {
        if !baseline.contains_key(&key_alt) {
            // Alternative key not found in baseline. Consider its absolute value as difference.
            // This might double count if we also add baseline_value.abs() above for missing keys.
            // For a fair comparison of values, we should only compare common keys or be very clear.
            // For now, let's just count how many are unique to alternative.
            count_baseline_missing_in_alt += 1;
        }
    }

    log::info!("  Comparing {} ({} entries) with {} ({} entries)",
        baseline_name, baseline.len(), alternative_name, alternative.len());

    if count_alternative_missing > 0 {
        log::debug!("    {} keys from baseline ({}) were missing in alternative ({}). Their values contribute to differences as absolute values.",
            count_alternative_missing, baseline_name, alternative_name);
    }
    if count_baseline_missing_in_alt > 0 {
        log::debug!("    {} keys from alternative ({}) were present only in alternative results (not included in percentile diffs).",
            count_baseline_missing_in_alt, alternative_name);
    }
    log::info!("    Number of common keys (used for percentile calculation): {}", common_keys_count);


    if differences.is_empty() {
        log::info!("    No common keys with non-zero values or no data to compare between {} and {}.", baseline_name, alternative_name);
        return;
    }

    differences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qty_total_for_percentiles = differences.len(); // Use the length of the differences vector

    log::info!("    Difference Percentiles (based on {} comparable values):", qty_total_for_percentiles);
    log::info!("      p10: {:.6e}", percentile(&differences, qty_total_for_percentiles, 10));
    log::info!("      p50 (median): {:.6e}", percentile(&differences, qty_total_for_percentiles, 50));
    log::info!("      p90: {:.6e}", percentile(&differences, qty_total_for_percentiles, 90));
    log::info!("      p99: {:.6e}", percentile(&differences, qty_total_for_percentiles, 99));
    log::info!("      p100 (max): {:.6e}", percentile(&differences, qty_total_for_percentiles, 100));
}


fn percentile(sorted_differences: &Vec<f64>, qty_total: usize, p: usize) -> f64 {
    if qty_total == 0 { return 0.0; }
    let p_clamped = p.clamp(0, 100); // Ensure p is between 0 and 100

    if p_clamped == 0 { return if sorted_differences.is_empty() { 0.0 } else { sorted_differences[0] }; }
    // For p100, we want the last element. Index is qty_total - 1.
    if p_clamped == 100 { return if sorted_differences.is_empty() { 0.0 } else { sorted_differences[qty_total - 1] }; }

    // Calculate rank: (P / 100) * (N - 1)
    // This is 0-indexed rank.
    let rank = (p_clamped as f64 / 100.0) * (qty_total.saturating_sub(1)) as f64;

    let lower_idx = rank.floor() as usize;
    let upper_idx = rank.ceil() as usize;

    // Ensure indices are within bounds
    let lower_idx_clamped = lower_idx.min(qty_total.saturating_sub(1));
    let upper_idx_clamped = upper_idx.min(qty_total.saturating_sub(1));


    if lower_idx_clamped == upper_idx_clamped { // Exact rank or at the boundaries
        sorted_differences[lower_idx_clamped]
    } else { // Interpolate
        let fraction = rank - lower_idx as f64;
        sorted_differences[lower_idx_clamped] * (1.0 - fraction) + sorted_differences[upper_idx_clamped] * fraction
    }
}

pub fn write_to_disk(importances: &HashMap<SessionId, f64>, output_filename: &str) {
    log::info!("[Output] Writing results to {}...", output_filename);
    let file = match File::create(output_filename) {
        Ok(file) => file,
        Err(err) => {
            log::error!("Error: Unable to create file '{}': {}", output_filename, err);
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    let mut sorted_importances: Vec<_> = importances.iter().collect();
    sorted_importances.sort_by_key(|&(k, _)| k);

    for (key, value) in sorted_importances {
        if let Err(err) = writeln!(writer, "{},{}", key, value) {
            log::error!("Error: Unable to write to file '{}': {}", output_filename, err);
            return;
        }
    }
    log::info!("[Output] CSV file '{}' exported successfully!", output_filename);
}

fn create_model_metric_train_valid(k: usize, m: usize) -> ModelAndData {
    let train_data = read_data(&TRAIN_DATA_FILE.to_string());
    let mut valid_data = read_data(&VALID_DATA_FILE.to_string());

    let mut rng = thread_rng();
    valid_data.shuffle(&mut rng);
    println!("ONLY SAMPLING 500 FROM VALIDATION DATA");
    let valid_subset_interactions: Vec<Interaction> = valid_data.into_iter().take(500).collect();

    let train_dataset = SessionDataset::new(train_data);
    let valid_dataset = SessionDataset::new(valid_subset_interactions);
    log::info!("Train dataset size: {}", train_dataset.sessions.len());
    log::info!("Valid dataset size (after sampling): {}", valid_dataset.sessions.len());

    let config = Box::new(MetricConfig {
        importance_metric: MetricType::MRR,
        evaluation_metrics: vec![MetricType::MRR],
        length: 20, // Standard length for MRR@k
        alpha: 0.0, // MRR doesn't typically use alpha, but if your MetricConfig needs it.
    });

    let product_info = Box::new(ProductInfo::new(HashSet::new())); // Assuming ProductInfo is Copy or cheap to clone
    let metric_factory = MetricFactory::new(Box::leak(config), *product_info); // Deref to pass ProductInfo by value

    let model = VMISKNN::fit_dataset(&train_dataset, m, k, 1.0);

    ModelAndData {
        model,
        metric_factory,
        train_dataset,
        valid_dataset,
    }
}

struct ModelAndData {
    model: VMISKNN,
    metric_factory: MetricFactory<'static>, // MetricFactory needs to be Cloneable
    train_dataset: SessionDataset,
    valid_dataset: SessionDataset,
}
