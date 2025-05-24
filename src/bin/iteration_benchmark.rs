// src/bin/iteration_benchmark.rs

use std::collections::{HashMap, HashSet};
use std::time::{Instant, Duration};
use std::env;
use std::io::Write; // For logging

use chrono::Local;
use env_logger::Builder;
use log::LevelFilter;
use rand::SeedableRng; // For StdRng if needed by KMC setup, though benchmark seed is passed
use rand::seq::SliceRandom;
use illoominate::baselines::ExperimentPayload;
// If used by KMC setup

// Illominate crate imports
use illoominate::conf::illoominateconfig::{IlloominateConfig, create_metric_config};
use illoominate::metrics::{MetricFactory, MetricConfig, MetricType, product_info::ProductInfo};
use illoominate::sessrec::types::{SessionDataset, ItemId};
use illoominate::sessrec::vmisknn::VMISKNN;
use illoominate::nbr::tifuknn::TIFUKNN;
use illoominate::nbr::tifuknn::types::HyperParams;
use illoominate::nbr::types::NextBasketDataset;
use illoominate::sessrec::io::{self as sessrec_io, read_sustainable_products_info}; // Aliased to avoid conflict if other 'io' is used
use illoominate::nbr::removal_impact::split_train_eval;

use illoominate::importance::k_mc_shapley::one_iteration_dataset as kmc_one_iteration_dataset;
use illoominate::importance::mc_utils::random_score_dataset;
use illoominate::importance::Dataset; // Trait needed for KMC's one_iteration_dataset

// --- Helper Structs and Functions (mirrored or adapted from removal_impact.rs) ---
// In a larger refactor, these would ideally be in a shared library module.

fn init_logging() {
    Builder::new()
        .filter_level(LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {}] {} {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.module_path().unwrap_or("-"),
                record.level(),
                record.args()
            )
        })
        .init();
}


fn read_datasets(data_path: &str, app_config: &IlloominateConfig) -> Datasets {
    match app_config.model.name.to_lowercase().as_str() {
        "vmis" => {
            let session_train = SessionDataset::new(sessrec_io::read_data(&format!("{}/train.csv", data_path)));
            let session_valid = SessionDataset::new(sessrec_io::read_data(&format!("{}/valid.csv", data_path)));
            let session_test = SessionDataset::new(sessrec_io::read_data(&format!("{}/test.csv", data_path)));
            Datasets {
                session_datasets: Some((session_train, session_valid, session_test)),
                basket_datasets: None,
            }
        },
        "tifu" => {
            let baskets_filename = "baskets.csv";
            let validation_ratio = app_config.nbr
                .as_ref()
                .and_then(|sbr| sbr.validation_ratio)
                .unwrap_or(0.5);

            log::info!("Using validation ratio {:.2}", validation_ratio);

            let basket_csv_path = format!("{}/{}", data_path, baskets_filename);
            let all_baskets_by_user: NextBasketDataset = read_baskets_file(&basket_csv_path);
            let (basket_train, basket_valid, basket_test) =
                split_train_eval(all_baskets_by_user, validation_ratio);
            Datasets {
                session_datasets: None,
                basket_datasets: Some((basket_train, basket_valid, basket_test)),
            }
        },
        invalid => panic!("Unknown model type: {}", invalid),
    }
}

pub fn create_metric_factory<'a>(
    data_path: &'a str,
    metric_config: &'a MetricConfig,
) -> MetricFactory<'a> {
    let sustainable_products: HashSet<ItemId> = if metric_config
        .evaluation_metrics
        .contains(&MetricType::SustainableMrr)
        || metric_config
        .evaluation_metrics
        .contains(&MetricType::SustainabilityCoverageTerm)
    {
        read_sustainable_products_info(&format!("{}/__sustainable_mapped_items.csv", data_path)) // Adjusted filename
    } else {
        HashSet::new()
    };

    let product_info = ProductInfo::new(sustainable_products);
    MetricFactory::new(metric_config, product_info.clone())
}


// --- The Benchmark Function (Moved from removal_impact.rs) ---
fn run_one_iteration_comparison(
    _data_path: &str, // Kept for signature consistency, though not directly used if datasets are pre-loaded
    app_config: &IlloominateConfig,
    metric_factory: &MetricFactory,
    datasets_struct: &Datasets,
    seed_for_benchmark: u64,
    num_benchmark_runs: usize,
) {
    log::info!("\n\n--- Starting One-Iteration Performance Comparison ({} runs each) ---", num_benchmark_runs);

    let is_vmis_model = match app_config.model.name.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        _ => {
            log::error!("Unknown model type for one-iteration benchmark. Skipping.");
            return;
        }
    };

    // --- TMC Vanilla Benchmark (Only for VMIS, as tmcshapley.rs is VMIS-specific) ---
    if is_vmis_model {
        if let Some((session_train, session_valid, _)) = &datasets_struct.session_datasets {
            let tmc_random_score = 0.0; // Placeholder for TMC context

            let tmc_payload = ExperimentPayload {
                train_dataset: session_train, // This is a reference, not ownership transfer
                valid_dataset: session_valid,
                metric_factory,
                vmis_m: app_config.hpo.m.unwrap_or(500),
                vmis_k: app_config.hpo.k.unwrap_or(50),
                monte_carlo_iterations: num_benchmark_runs, // Part of payload, not directly used by one_iteration
                seed: seed_for_benchmark as usize,
                random_score: tmc_random_score,
            };

            // Disable truncation for a raw speed comparison of the iteration logic
            let tmc_durations = run_tmc_one_iteration_benchmark(&tmc_payload, num_benchmark_runs, false);
            if !tmc_durations.is_empty() {
                let total_duration: Duration = tmc_durations.iter().sum();
                log::info!("TMC Vanilla `one_iteration_dataset` (VMIS) average time: {:?}", total_duration / num_benchmark_runs as u32);
                log::info!("TMC Vanilla `one_iteration_dataset` (VMIS) individual times: {:?}", tmc_durations);
            }
        } else {
            log::warn!("Session datasets not available for TMC Vanilla benchmark, skipping.");
        }
    } else {
        log::info!("Skipping TMC Vanilla benchmark as model is not VMIS.");
    }

    // --- KMC-Shapley `one_iteration_dataset` Benchmark ---
    let mut kmc_durations = Vec::with_capacity(num_benchmark_runs);
    log::info!("Starting KMC `one_iteration_dataset` benchmark for {} model and {} runs...", app_config.model.name, num_benchmark_runs);

    if is_vmis_model {
        if let Some((session_train_kmc, session_valid_kmc, _)) = &datasets_struct.session_datasets {
            let model_kmc = VMISKNN::fit_dataset(
                session_train_kmc,
                app_config.hpo.m.unwrap_or(500),
                app_config.hpo.k.unwrap_or(50),
                1.0,
            );
            let (kmc_random_score, _) = random_score_dataset(&model_kmc, metric_factory, session_valid_kmc);
            log::info!("KMC Benchmark (VMIS): Calculated random_score: {}", kmc_random_score);

            for i in 0..num_benchmark_runs {
                let start_time = Instant::now();
                let _contributions = kmc_one_iteration_dataset(
                    &model_kmc,
                    metric_factory,
                    session_train_kmc,
                    session_valid_kmc,
                    kmc_random_score,
                    seed_for_benchmark as usize, // KMC's function expects usize seed
                    i,                           // Iteration number for permutation
                );
                kmc_durations.push(start_time.elapsed());
            }
        } else {
            log::warn!("Session datasets not available for KMC (VMIS) benchmark, skipping.");
        }
    } else { // TIFU case for KMC
        if let Some((basket_train_kmc, basket_valid_kmc, _)) = &datasets_struct.basket_datasets {
            let tifu_hyperparameters = HyperParams::from(&app_config.hpo);
            let model_kmc = TIFUKNN::new(basket_train_kmc, &tifu_hyperparameters);
            let (kmc_random_score, _) = random_score_dataset(&model_kmc, metric_factory, basket_valid_kmc);
            log::info!("KMC Benchmark (TIFU): Calculated random_score: {}", kmc_random_score);

            for i in 0..num_benchmark_runs {
                let start_time = Instant::now();
                let _contributions = kmc_one_iteration_dataset(
                    &model_kmc,
                    metric_factory,
                    basket_train_kmc,
                    basket_valid_kmc,
                    kmc_random_score,
                    seed_for_benchmark as usize,
                    i,
                );
                kmc_durations.push(start_time.elapsed());
            }
        } else {
            log::warn!("Basket datasets not available for KMC (TIFU) benchmark, skipping.");
        }
    }

    if !kmc_durations.is_empty() {
        for (i, duration) in kmc_durations.iter().enumerate() {
            log::info!("KMC `one_iteration_dataset` ({}) run {}/{} took {:?}", app_config.model.name, i + 1, num_benchmark_runs, duration);
        }
        let total_duration: Duration = kmc_durations.iter().sum();
        log::info!("KMC `one_iteration_dataset` ({}) average time: {:?}", app_config.model.name, total_duration / num_benchmark_runs as u32);
        log::info!("KMC `one_iteration_dataset` ({}) individual times: {:?}", app_config.model.name, kmc_durations);
    } else {
        log::info!("KMC benchmark ({}) did not produce timing results (check data/model setup).", app_config.model.name);
    }
    log::info!("--- One-Iteration Performance Comparison Finished ---");
}


fn main() {
    init_logging();

    let data_location = match env::var("DATA_LOCATION") {
        Ok(val) => val.trim().to_string(),
        Err(_) => {
            log::error!("Environment variable DATA_LOCATION is not set. Exiting.");
            std::process::exit(1);
        }
    };

    let config_filename = match env::var("CONFIG_FILENAME") {
        Ok(val) => val.trim().to_string(),
        Err(_) => {
            log::error!("Environment variable CONFIG_FILENAME is not set. Exiting.");
            std::process::exit(1);
        }
    };

    log::info!("DATA_LOCATION: {}", data_location);
    log::info!("CONFIG_FILENAME: {}", config_filename);

    let app_config = IlloominateConfig::load(&format!("{}/{}", data_location, config_filename))
        .expect("Failed to load config file");
    log::info!("Loaded app_config: {:?}", app_config);

    let metric_config = create_metric_config(&app_config);
    let metric_factory = create_metric_factory(&data_location, &metric_config);
    let datasets_struct = read_datasets(&data_location, &app_config);

    let num_benchmark_runs = 10; 

    let benchmark_seed = 42;
    
    run_one_iteration_comparison(
        &data_location,
        &app_config,
        &metric_factory,
        &datasets_struct,
        benchmark_seed,
        num_benchmark_runs,
    );
}