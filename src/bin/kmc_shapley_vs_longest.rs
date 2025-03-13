
use std::collections::{HashMap, HashSet};
use std::{env};
use std::cmp::max;
use std::fs::File;
use chrono::Local;
use env_logger::Builder;
use log::LevelFilter;
use illoominate::conf::illoominateconfig::{create_metric_config, IlloominateConfig};
use illoominate::metrics::{MetricConfig, MetricFactory, MetricType};
use std::io::Write;
use std::process::exit;
use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;
use illoominate::importance::k_loo::KLoo;
use illoominate::importance::k_mc_shapley::KMcShapley;
use illoominate::sessrec::io;
use illoominate::sessrec::types::{ItemId, SessionDataset};
use illoominate::sessrec::vmisknn::VMISKNN;
use rand::seq::IteratorRandom;
use illoominate::importance::{Dataset, Importance};
use illoominate::nbr::removal_impact::{split_train_eval, tifu_evaluate_removal_impact};
use illoominate::nbr::tifuknn::io::read_baskets_file;
use illoominate::nbr::tifuknn::TIFUKNN;
use illoominate::nbr::tifuknn::types::{HyperParams};
use illoominate::nbr::types::NextBasketDataset;
use illoominate::sessrec::io::read_sustainable_products_info;
use illoominate::metrics::product_info::ProductInfo;
use illoominate::sessrec::removal_impact::{create_file, vmis_evaluate_removal_impact};

fn main() {
    // Experiment: KMC-Shapley vs. Removing Longest Sessions
    // Objective: Compare Shapley-based data pruning against removing long sessions.
    // Methodology: See full details in docs/experiments/kmc_shapley_vs_longest.md
    init_logging();
    let data_location = match env::var("DATA_LOCATION") {
        Ok(val) => val.trim().to_string(), // Convert to String to take ownership
        Err(_) => {
            log::error!("Environment variable DATA_LOCATION is not set");
            std::process::exit(1); // Exit with an error code
        }
    };

    let config_filename = match env::var("CONFIG_FILENAME") {
        Ok(val) => val.trim().to_string(), // Convert to String to take ownership
        Err(_) => {
            log::error!("Environment variable CONFIG_FILENAME is not set");
            std::process::exit(1); // Exit with an error code
        }
    };

    log::info!("DATA_LOCATION: {data_location}");
    log::info!("CONFIG_FILENAME: {config_filename}");

    let app_config = IlloominateConfig::load(format!("{}/{}", data_location, config_filename).as_str()).expect("Failed to load config file");
    log::info!("{:?}", app_config);

    let metric_config = create_metric_config(&app_config);

    run_experiment(&data_location, &app_config, &metric_config);
}



fn run_experiment(data_path: &str, app_config: &IlloominateConfig, metric_config: &MetricConfig) {
    let metric_factory = create_metric_factory(data_path, metric_config);
    let is_vmis = match app_config.model.name.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let datasets = read_datasets(data_path, app_config);

    let (mut loo_outputs, mut shapley_outputs, mut longest_session_outputs, mut metric_outputs) = create_output_files(data_path, &metric_factory);


    let qty_impact_resolution = 250;


    // for seed in [1313, 18102022, 12345] {
    for seed in [1313] {
        log::info!("determine removal impact with seed: {}", seed);

        let shapley_error = 2.0;
        let shapley_num_iterations = 5;

        let k_loo_algorithm = KLoo::new();
        let kmc_shapley_algorithm = KMcShapley::new(shapley_error, shapley_num_iterations, seed);

        let (loo_importances, shap_values, longest_sessions) = if is_vmis {
            if let Some((session_train, session_valid, _session_test)) = &datasets.session_datasets {
                let model:VMISKNN = VMISKNN::fit_dataset(session_train, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), 1.0);

                let loo_importances = k_loo_algorithm.compute_importance(&model, &metric_factory, session_train, session_valid);
                let shap_values = kmc_shapley_algorithm.compute_importance(&model, &metric_factory, session_train, session_valid);
                let longest_sessions = get_longest_sessions(session_train);
                (loo_importances, shap_values, longest_sessions)
            } else {
                panic!("unable to read training data for SBR")
            }
        } else if let Some((basket_train, basket_valid, _basket_test)) = &datasets.basket_datasets {
            panic!("Only VMIS model supported for longest session removal");
            log::error!("storing training data for SBR");
        } else { panic!() };

        write_importance(&loo_importances, &mut loo_outputs, seed);
        write_importance(&shap_values, &mut shapley_outputs, seed);
        write_importance(&longest_sessions, &mut longest_session_outputs, seed);

        /////////////// START DATA REMOVAL EXPERIMENTS

        let important_first_loo = positive_by_importance(&loo_importances);
        let least_first_loo = negative_by_importance_reverse(&loo_importances);
        let important_first_shapley = positive_by_importance(&shap_values);
        let least_first_shapley = negative_by_importance_reverse(&shap_values);
        let long_first = get_keys_sorted_by_value_desc(&shap_values);
        let short_first: Vec<u32> = long_first.iter().rev().cloned().collect();

        let num_random_sessions_to_remove =
            *[loo_importances.keys().len(),
                shap_values.keys().len(),
            ].iter().max().unwrap();

        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut random_keys_to_remove: Vec<u32> = shap_values.clone()
            .into_iter()
            .choose_multiple(&mut rng, num_random_sessions_to_remove)
            .into_iter()
            .map(|(index, _)| index)
            .collect();

        random_keys_to_remove.shuffle(&mut rng);


        if is_vmis {
            if let Some((session_train, session_valid, session_test)) = &datasets.session_datasets {
                let evaluation_interval = max(
                    1,
                    (num_random_sessions_to_remove as f64 / qty_impact_resolution as f64) as usize,
                );
                let max_qty_evaluations: usize = num_random_sessions_to_remove;

                vmis_evaluate_removal_impact("important_first_loo", &metric_factory, session_train, session_valid, session_test,
                                             &important_first_loo, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);

                vmis_evaluate_removal_impact("least_first_loo", &metric_factory, session_train, session_valid, session_test,
                                             &least_first_loo, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);

                vmis_evaluate_removal_impact("important_first_shapley", &metric_factory, session_train, session_valid, session_test,
                                             &important_first_shapley, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);


                vmis_evaluate_removal_impact("least_first_shapley", &metric_factory, session_train, session_valid, session_test,
                                             &least_first_shapley, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);


                vmis_evaluate_removal_impact("random", &metric_factory, session_train, session_valid, session_test,
                                             &random_keys_to_remove, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);


                vmis_evaluate_removal_impact("long_first", &metric_factory, session_train, session_valid, session_test,
                                             &long_first, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);
                vmis_evaluate_removal_impact("short_first", &metric_factory, session_train, session_valid, session_test,
                                             &short_first, app_config.hpo.m.unwrap(), app_config.hpo.k.unwrap(), seed, evaluation_interval,
                                             max_qty_evaluations, &mut metric_outputs);
            } else { panic!() }
        } else if let Some((basket_train, basket_valid, basket_test)) = &datasets.basket_datasets {
            let tifu_hyperparameters = HyperParams::from(&app_config.hpo);
            tifu_evaluate_removal_impact("important_first_loo", &metric_factory, basket_train, basket_valid, basket_test,
                                         &important_first_loo, &tifu_hyperparameters, seed, qty_impact_resolution,
                                         &mut metric_outputs);

            tifu_evaluate_removal_impact("least_first_loo", &metric_factory, basket_train, basket_valid, basket_test,
                                         &least_first_loo, &tifu_hyperparameters, seed, qty_impact_resolution,
                                         &mut metric_outputs);

            tifu_evaluate_removal_impact("important_first_shapley", &metric_factory, basket_train, basket_valid, basket_test,
                                         &important_first_shapley, &tifu_hyperparameters, seed, qty_impact_resolution,
                                         &mut metric_outputs);

            tifu_evaluate_removal_impact("least_first_shapley", &metric_factory, basket_train, basket_valid, basket_test,
                                         &least_first_shapley, &tifu_hyperparameters, seed, qty_impact_resolution,
                                         &mut metric_outputs);

            tifu_evaluate_removal_impact("random", &metric_factory, basket_train, basket_valid, basket_test,
                                         &random_keys_to_remove, &tifu_hyperparameters, seed, qty_impact_resolution,
                                         &mut metric_outputs);
        } else { panic!() };

        /////////////// END DATA REMOVAL EXPERIMENTS
    }
}

fn write_importance(importances: &HashMap<u32, f64>, output_file: &mut File, used_seed_value: usize) {
    let mut indices_sorted_by_importance: Vec<(u32, f64)> = importances.clone()
        .into_iter().collect();
    indices_sorted_by_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (index, importance) in &indices_sorted_by_importance {
        let data_to_append = format!("{},{},{:.10}", used_seed_value, index, importance);
        writeln!(output_file, "{}", data_to_append)
            .expect("Failed to write to file");
    }
}

fn get_keys_sorted_by_value_desc(map: &HashMap<u32, f64>) -> Vec<u32> {
    let mut sorted_keys: Vec<(f64, u32)> = map.iter().map(|(&key, &value)| (value, key)).collect();
    sorted_keys.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    sorted_keys.into_iter().map(|(_, key)| key).collect()
}

fn get_keys_sorted_by_value_asc(map: &HashMap<u32, f64>) -> Vec<u32> {
    let mut sorted_keys: Vec<(f64, u32)> = map.iter().map(|(&key, &value)| (value, key)).collect();
    sorted_keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    sorted_keys.into_iter().map(|(_, key)| key).collect()
}

pub fn positive_by_importance(original_importances: &HashMap<u32, f64>) -> Vec<u32> {
    // orders by score in descending order and filter out positive scores.
    let mut importances: Vec<(u32, f64)> = original_importances.clone().into_iter().collect();
    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    importances.into_iter()
        .filter(|(_, importance)| *importance > 0.0)
        .map(|(session_id, _)| session_id)
        .collect()
}

pub fn negative_by_importance_reverse(original_importances: &HashMap<u32, f64>) -> Vec<u32> {
    // orders by score in ascending order and filters out only negative scores
    let mut importances: Vec<(u32, f64)> = original_importances.clone().into_iter().collect();
    importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    importances.into_iter()
        .filter(|(_, importance)| *importance < 0.0)
        .map(|(session_id, _)| session_id)
        .collect()
}

fn create_output_files(data_path: &str, metric_factory: &MetricFactory) -> (File, File, File, Vec<File>){
    let overwrite_output = true;

    let importance_metric_binding = metric_factory.create_importance_metric();

    let importance_metric_friendly = importance_metric_binding.as_ref().get_name().to_lowercase();
    let output_file_loo_importances =
        create_file(&format!("{}/__removal_impact_loo_importance_{}_eval_{}.csv", data_path, importance_metric_friendly, importance_metric_friendly), overwrite_output);
    let output_file_shapley_importances =
        create_file(&format!("{}/__removal_impact_shapley_importance_{}_eval_{}.csv", data_path, importance_metric_friendly, importance_metric_friendly), overwrite_output);
    let output_file_longest_session =
        create_file(&format!("{}/__removal_impact_longest_session_{}_eval_{}.csv", data_path, importance_metric_friendly, importance_metric_friendly), overwrite_output);


    let output_files_evaluation_metric_results: Vec<File> = metric_factory
        .create_evaluation_metrics()
        .iter_mut()
        .map(|metric| {
            let metric_friendly_name = metric.get_name().to_lowercase();
            create_file(&format!(
                "{}/__removal_impact_results_importance_{}_eval_{}.csv",
                data_path, importance_metric_friendly, metric_friendly_name
            ), overwrite_output)
        })
        .collect();

    (output_file_loo_importances, output_file_shapley_importances, output_file_longest_session, output_files_evaluation_metric_results)
}

fn get_longest_sessions(session_train: &SessionDataset) -> HashMap<u32, f64> {
    let session_lengths: HashMap<u32, f64> = session_train.sessions.iter()
        .map(|(session_id, (items, _max_timestamp))| (*session_id, items.len() as f64))
        .collect();
    session_lengths
}


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
            let session_train:SessionDataset = SessionDataset::new(io::read_data(&format!("{}/train.csv", data_path)));
            let session_valid :SessionDataset= SessionDataset::new(io::read_data(&format!("{}/valid.csv", data_path)));
            let session_test:SessionDataset = SessionDataset::new(io::read_data(&format!("{}/test.csv", data_path)));
            Datasets {
                session_datasets: Option::from((session_train, session_valid, session_test)),
                basket_datasets: None,
            }
        },
        "tifu" => {
            let baskets_filename = "baskets.csv";
            let validation_ratio = app_config.nbr
                .as_ref()
                .and_then(|sbr| sbr.validation_ratio)
                .unwrap_or(0.5); // Use default value if `validation_size` is not configured

            log::info!("Using validation ratio {:.2}", validation_ratio);

            let basket_csv_path = format!("{}/{}", data_path, baskets_filename);
            let all_baskets_by_user: NextBasketDataset =
                read_baskets_file(&basket_csv_path);
            let (basket_train, basket_valid, basket_test) =
                split_train_eval(all_baskets_by_user, validation_ratio);
            Datasets {
                session_datasets: None,
                basket_datasets: Option::from((basket_train, basket_valid, basket_test)),
            }

        },
        invalid => panic!("Unknown model type: {}", invalid),
    }
}

struct Datasets {
    session_datasets: Option<(SessionDataset, SessionDataset, SessionDataset)>,
    basket_datasets: Option<(NextBasketDataset, NextBasketDataset, NextBasketDataset)>,
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
        read_sustainable_products_info(&format!("{}/__sustainable_mapped_items.csv.csv", data_path))
    } else {
        HashSet::new()
    };

    let product_info = ProductInfo::new(sustainable_products);

    let metric_factory = MetricFactory::new(metric_config, product_info.clone());
    metric_factory
}
