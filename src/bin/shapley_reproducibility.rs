#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

use std::collections::{HashMap, HashSet};
use std::env;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{BufWriter, Write};
use illoominate::importance::Importance;

// Ensure this points to the tmc_shapley function in /src/baselines/tmcshapley.rs
// and that ExperimentPayload is accessible from there.
use illoominate::baselines::tmcshapley::{tmc_shapley, ExperimentPayload};
use illoominate::importance::mc_utils::random_score_dataset; // For calculating random_score
use illoominate::sessrec::io;
use illoominate::sessrec::types::{SessionDataset, SessionId};
use env_logger;
// log::logger is not typically used directly, log macros (info!, debug!, etc.) are preferred.
// use log::logger; 
use rayon::{current_num_threads, ThreadPoolBuilder};
use illoominate::importance::k_mc_shapley::KMcShapley;
use illoominate::metrics::{MetricConfig, MetricFactory, MetricType};
use illoominate::metrics::product_info::ProductInfo;
use illoominate::sessrec::io::read_data;
use illoominate::sessrec::vmisknn::VMISKNN;

const TRAIN_DATA_FILE: &str = "data/instacart/processed/train.csv";
const VALID_DATA_FILE: &str = "data/instacart/processed/valid.csv";

fn main() {
    env::set_var("RUST_LOG", "info"); // Set to info or debug as needed
    env_logger::init();

    let pool_single_thread = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool_multi_thread = ThreadPoolBuilder::new().num_threads(0).build().unwrap(); // 0 uses default number of threads
    log::info!("Rayon multi-threaded pool using {} threads", current_num_threads());

    // Loop parameters for the benchmark
    // Using a single iteration for k, m, and seed for a focused benchmark run.
    // You can expand these arrays if you want to test more configurations.
    for (k_param, m_param) in [(50, 500)] { // k and m parameters for VMISKNN
        for seed_val in [42usize] { // Seed for reproducibility

            log::info!("\n\n----[BENCHMARK RUN - SEED: {}, k: {}, m: {}]----------------------------", seed_val, k_param, m_param);

            let model_and_data = create_model_metric_train_valid(k_param, m_param);

            // Parameters for Shapley calculations
            let kmc_error_threshold = 0.1; // Convergence error for KMC-Shapley
            // num_iterations for KMC is iterations per convergence check.
            // monte_carlo_iterations for TMC is total permutations.
            // For benchmarking a single iteration's mechanism, 1 is fine for TMC.
            // For KMC, its internal loop will run until convergence or max_iterations.
            let num_shapley_iterations = 1;

            // Calculate random_score needed for ExperimentPayload
            let (calculated_random_score, _) = random_score_dataset(
                &model_and_data.model,
                &model_and_data.metric_factory,
                &model_and_data.valid_dataset
            );
            log::info!("Calculated random_score for benchmark: {:.5}", calculated_random_score);

            // --- TMC-Shapley (Vanilla, non-truncated) ---
            let tmc_payload_mc = ExperimentPayload {
                train_dataset: model_and_data.train_dataset.clone(),
                valid_dataset: model_and_data.valid_dataset.clone(),
                metric_factory: model_and_data.metric_factory.clone(),
                vmis_m: m_param,
                vmis_k: k_param,
                monte_carlo_iterations: num_shapley_iterations, // For TMC, this is total permutations
                seed: seed_val, 
                random_score: calculated_random_score,
            };
            let start_time_mc = Instant::now();
            let mc_results_vec = tmc_shapley(&tmc_payload_mc, false); // truncation_enabled = false
            let mc_duration = start_time_mc.elapsed();
            let mc_results = vec_to_hashmap_filtered(mc_results_vec);

            // --- TMC-Shapley (Vanilla, truncated) ---
            let tmc_payload_tmc = ExperimentPayload { // Re-create or clone if mutable aspects are involved, here it's fine
                train_dataset: model_and_data.train_dataset.clone(),
                valid_dataset: model_and_data.valid_dataset.clone(),
                metric_factory: model_and_data.metric_factory.clone(),
                vmis_m: m_param,
                vmis_k: k_param,
                monte_carlo_iterations: num_shapley_iterations,
                seed: seed_val,
                random_score: calculated_random_score,
            };
            let start_time_tmc = Instant::now();
            let tmc_results_vec = tmc_shapley(&tmc_payload_tmc, true); // truncation_enabled = true
            let tmc_duration = start_time_tmc.elapsed();
            let tmc_results = vec_to_hashmap_filtered(tmc_results_vec);

            // --- KMC-Shapley (Single Threaded) ---
            let mut kmc_st_results= HashMap::new();
            let mut kmc_st_duration = Duration::new(0,0);
            pool_single_thread.install(|| {
                // For KMC, num_shapley_iterations is 'iterations per convergence check'
                let kmc_algorithm = KMcShapley::new(kmc_error_threshold, num_shapley_iterations, seed_val);
                let start_time = Instant::now();
                // Assuming KMcShapley::compute_importance returns HashMap<u32, f64>
                // as per the provided src/importance/k_mc_shapley.rs
                kmc_st_results = kmc_algorithm.compute_importance(
                    &model_and_data.model,
                    &model_and_data.metric_factory,
                    &model_and_data.train_dataset,
                    &model_and_data.valid_dataset
                );
                kmc_st_duration = start_time.elapsed();
            });

            // --- KMC-Shapley (Multi Threaded) ---
            let mut kmc_mt_results= HashMap::new();
            let mut kmc_mt_duration = Duration::new(0,0);
            pool_multi_thread.install(|| {
                let kmc_algorithm = KMcShapley::new(kmc_error_threshold, num_shapley_iterations, seed_val);
                let start_time = Instant::now();
                kmc_mt_results = kmc_algorithm.compute_importance(
                    &model_and_data.model,
                    &model_and_data.metric_factory,
                    &model_and_data.train_dataset,
                    &model_and_data.valid_dataset
                );
                kmc_mt_duration = start_time.elapsed();
            });

            log::info!("Duration Vanilla TMC-Shapley (non-truncated): {:?}", mc_duration);
            log::info!("Duration Vanilla TMC-Shapley (truncated):    {:?}", tmc_duration);
            log::info!("Duration K-MC-Shapley (Single Threaded):   {:?}", kmc_st_duration);
            log::info!("Duration K-MC-Shapley (Multi Threaded):    {:?}", kmc_mt_duration);

            // Sum of importances (optional, for sanity check)
            let mc_sum_of_importances: f64 = mc_results.values().sum();
            let tmc_sum_of_importances: f64 = tmc_results.values().sum();
            let kmc_st_sum_of_importances: f64 = kmc_st_results.values().sum();
            let kmc_mt_sum_of_importances: f64 = kmc_mt_results.values().sum();

            log::debug!("Vanilla TMC (non-trunc) sum_of_importances: {:.5}", mc_sum_of_importances);
            log::debug!("Vanilla TMC (trunc) sum_of_importances:     {:.5}", tmc_sum_of_importances);
            log::debug!("K-MC-Shapley (ST) sum_of_importances:       {:.5}", kmc_st_sum_of_importances);
            log::debug!("K-MC-Shapley (MT) sum_of_importances:       {:.5}", kmc_mt_sum_of_importances);

            // Comparisons (optional for benchmark, but good for reproducibility check)
            log::info!("--- Result Comparison (vs Vanilla TMC non-truncated) ---");
            log::info!("Comparing Vanilla TMC (non-trunc) with itself:");
            print_results(&mc_results, &mc_results);
            log::info!("Comparing Vanilla TMC (non-trunc) with Vanilla TMC (trunc):");
            print_results(&mc_results, &tmc_results);
            log::info!("Comparing Vanilla TMC (non-trunc) with K-MC-Shapley (ST):");
            print_results(&mc_results, &kmc_st_results);
            log::info!("Comparing Vanilla TMC (non-trunc) with K-MC-Shapley (MT):");
            print_results(&mc_results, &kmc_mt_results);

            // Writing to disk (optional for benchmark)
            // write_to_disk(&mc_results, "shapley_reproducibility_mc.csv");
            // write_to_disk(&tmc_results, "shapley_reproducibility_tmc.csv");
            // write_to_disk(&kmc_st_results, "shapley_reproducibility_kmc_st.csv");
            // write_to_disk(&kmc_mt_results, "shapley_reproducibility_kmc_mt.csv");
        }
    }
}

// Helper function to convert Vec<f64> (indexed by SessionId) to HashMap<SessionId, f64>
// It also filters out zero scores, which can be common for non-contributing data points.
fn vec_to_hashmap_filtered(scores_vec: Vec<f64>) -> HashMap<SessionId, f64> {
    scores_vec
        .into_iter()
        .enumerate()
        .map(|(id, score)| (id as SessionId, score)) // Assuming SessionId is u32 or similar to usize
        .filter(|(_, score)| *score != 0.0) // Optional: keep only non-zero contributions
        .collect()
}


fn print_results(baseline: &HashMap<SessionId, f64>, alternative: &HashMap<SessionId, f64>) {
    let mut baseline_vec: Vec<_> = baseline.iter().collect();
    baseline_vec.sort_by_key(|&(k, _)| k);

    let mut differences = Vec::new();
    let mut count_alternative_missing = 0;
    let mut count_baseline_missing_in_alt = 0;


    for (&key, baseline_value) in baseline_vec.iter() {
        if let Some(alternative_value) = alternative.get(&key) {
            let difference = (*baseline_value - *alternative_value).abs();
            differences.push(difference);
        } else {
            // Key from baseline is not in alternative. Difference is |baseline_value - 0.0|
            differences.push(baseline_value.abs());
            count_alternative_missing +=1;
        }
    }

    // Check for keys in alternative not in baseline (if comprehensive comparison is needed)
    for &key_alt in alternative.keys() {
        if !baseline.contains_key(&key_alt) {
            // Key from alternative is not in baseline. Difference is |0.0 - alternative_value|
            // This part is tricky if baseline is considered the ground truth for existing keys.
            // For now, focusing on differences for common keys or keys present in baseline.
            count_baseline_missing_in_alt +=1;
        }
    }
    if count_alternative_missing > 0 {
        log::debug!("{} keys from baseline were missing in alternative results.", count_alternative_missing);
    }
    if count_baseline_missing_in_alt > 0 {
        log::debug!("{} keys from alternative were missing in baseline results.", count_baseline_missing_in_alt);
    }


    if differences.is_empty() {
        log::info!("No common keys with non-zero values or no data to compare.");
        return;
    }

    differences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let qty_total = differences.len();

    log::info!("Difference Percentiles (based on {} comparable data points):", qty_total);
    log::info!("  p10: {:.6}", percentile(&differences, qty_total, 10));
    log::info!("  p50 (median): {:.6}", percentile(&differences, qty_total, 50));
    log::info!("  p90: {:.6}", percentile(&differences, qty_total, 90));
    log::info!("  p99: {:.6}", percentile(&differences, qty_total, 99));
    log::info!("  p100 (max): {:.6}", percentile(&differences, qty_total, 100));
}

fn percentile(sorted_differences: &Vec<f64>, qty_total: usize, p: usize) -> f64 {
    if qty_total == 0 {
        return 0.0;
    }
    // Ensure p is within bounds for indexing after rank calculation
    let p_clamped = p.clamp(0, 100);


    if p_clamped == 0 {
        // Check if sorted_differences is empty before accessing index 0
        return if sorted_differences.is_empty() { 0.0 } else { sorted_differences[0] };
    }
    if p_clamped == 100 {
        return if sorted_differences.is_empty() { 0.0 } else { sorted_differences[qty_total - 1] };
    }

    // Using (qty_total - 1) for 0-indexed rank calculation.
    // Ensure qty_total > 0 before this calculation to avoid underflow with qty_total - 1 if qty_total is 0.
    // This is covered by the initial qty_total == 0 check.
    let rank = (p_clamped as f64 / 100.0) * (qty_total - 1) as f64;

    let lower_idx = rank.floor() as usize;
    let upper_idx = rank.ceil() as usize;

    // Ensure indices are within bounds
    if upper_idx >= qty_total { // Should only happen if lower_idx is also qty_total -1 due to clamping p to 100
        return sorted_differences[qty_total - 1];
    }
    if lower_idx >= qty_total { // Should not happen if p_clamped <= 100 and rank calc is correct
        return sorted_differences[qty_total - 1];
    }


    if lower_idx == upper_idx {
        sorted_differences[lower_idx]
    } else {
        // Linear interpolation
        let fraction = rank - lower_idx as f64;
        sorted_differences[lower_idx] * (1.0 - fraction) + sorted_differences[upper_idx] * fraction
    }
}


pub fn write_to_disk(importances: &HashMap<SessionId, f64>, output_filename: &str) {
    let file = match File::create(output_filename) {
        Ok(file) => file,
        Err(err) => {
            log::error!("Error: Unable to create file '{}': {}", output_filename, err);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // For consistent output, sort by key before writing
    let mut sorted_importances: Vec<_> = importances.iter().collect();
    sorted_importances.sort_by_key(|&(k, _)| k);

    for (key, value) in sorted_importances {
        if let Err(err) = writeln!(writer, "{},{}", key, value) {
            log::error!("Error: Unable to write to file '{}': {}", output_filename, err);
            return; // Stop writing on first error
        }
    }
    log::info!("CSV file '{}' exported successfully!", output_filename);
}

fn create_model_metric_train_valid(k: usize, m: usize) -> ModelAndData {
    let train_dataset = SessionDataset::new(read_data(&TRAIN_DATA_FILE.to_string()));
    let valid_dataset = SessionDataset::new(read_data(&VALID_DATA_FILE.to_string()));

    let config = Box::new(MetricConfig {
        importance_metric: MetricType::MRR,
        evaluation_metrics: vec![MetricType::MRR],
        length: 21,
        alpha: 0.0,
    });

    // ProductInfo can be empty if sustainability metrics are not the focus here
    let product_info = Box::new(ProductInfo::new(HashSet::new()));

    // Leak the boxed values to convert them into `'static` references
    // This is generally okay for a benchmark/main binary but be mindful in library code.
    let metric_factory = MetricFactory::new(Box::leak(config), *product_info); // Deref product_info

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
    metric_factory: MetricFactory<'static>, // MetricFactory now holds 'static refs
    train_dataset: SessionDataset,
    valid_dataset: SessionDataset,
}