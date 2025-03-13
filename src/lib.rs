use std::collections::{HashMap, HashSet};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use crate::importance::{evaluate_dataset, Dataset, Importance};
use crate::importance::k_loo::KLoo;
use crate::importance::k_mc_shapley::KMcShapley;
use crate::nbr::tifuknn::io::polars_to_purchases;
use crate::nbr::tifuknn::TIFUKNN;
use crate::nbr::tifuknn::types::HyperParams;
use crate::nbr::types::NextBasketDataset;
use crate::sessrec::io::polars_to_interactions;
use crate::sessrec::io::get_sustainable_items;
use crate::metrics::{MetricConfig, MetricFactory, MetricType};
use crate::metrics::product_info::ProductInfo;
use crate::sessrec::types::{Interaction, ItemId, SessionDataset};
use crate::sessrec::vmisknn::VMISKNN;

pub mod baselines;
pub mod conf;
pub mod importance;
pub mod nbr;
pub mod sessrec;
mod utils;
pub mod metrics;

#[pyfunction]
fn debug_sbr(pydf: PyDataFrame) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();

    // Pre-fetch columns to avoid repeated lookups
    let session_id_col = df.column("session_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let item_id_col = df.column("item_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let timestamp_col = df.column("timestamp")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Pre-allocate memory for results
    let mut results = Vec::with_capacity(df.height());

    // Iterate by row index, directly accessing each column
    for i in 0..df.height() {
        let session_id = match session_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in session_id column")),
        };
        let item_id = match item_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in item_id column")),
        };
        let timestamp = match timestamp_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in timestamp column")),
        };

        // Create an Interaction instance and store it
        let interaction = Interaction::new(session_id, item_id, timestamp);
        results.push(interaction);
    }

    let session_ids = polars::prelude::Column::Series(Series::new("session_id".into(), &[606u64, 2, 3, 4, 607]));
    let item_ids = polars::prelude::Column::Series(Series::new("item_id".into(), &[107u64, 102, 113, 104, 105]));
    let timestamps = polars::prelude::Column::Series(Series::new("timestamp".into(), &[1609459200u64, 1609459260, 1609459320, 1609459380, 1609459440]));

    // Create a DataFrame from the series
    let df = DataFrame::new(vec![session_ids, item_ids, timestamps])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

#[pyfunction]
fn data_shapley_polars(data: PyDataFrame, validation: PyDataFrame, model: &str, metric: &str,
                params: HashMap<String, f64>, sustainable: PyDataFrame) -> PyResult<PyDataFrame> {

    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = match model.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let metric_config = parse_metric_config(metric);

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let convergence_threshold: f64 = params.get("convergence_threshold").cloned().unwrap_or(0.1);
    let convergence_check_every: usize = params.get("convergence_check_every").map(|&v| v as usize).unwrap_or(10);
    let seed: usize = params.get("seed").map(|&v| v as usize).unwrap_or(42);

    let kmc_shapley_algorithm = KMcShapley::new(convergence_threshold, convergence_check_every, seed);

    let shap_values:HashMap<u32, f64> = if is_sbr {
        let session_train = match polars_to_interactions(data_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let session_valid = match polars_to_interactions(validation_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let m = params.get("m").map(|&m| m as usize).expect("param `m` is mandatory for this algorithm. e.g. 500");
        let k = params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 250");


        let model:VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        let shap_values = kmc_shapley_algorithm.compute_importance(&model, &metric_factory, &session_train, &session_valid);
        shap_values
    } else {
        let basket_train: NextBasketDataset = match polars_to_purchases(data_df) {
            Ok(purchases) => {
                NextBasketDataset::from(&purchases)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };
        let basket_valid: NextBasketDataset = match polars_to_purchases(validation_df) {
            Ok(purchases) => {
                NextBasketDataset::from(&purchases)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let tifu_hyperparameters = HyperParams {
            m: params.get("m").map(|&m| m as isize).expect("param `m` is mandatory for this algorithm. e.g. 500"),
            k: params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 500"),
            r_b: *params.get("r_b").expect("param `r_b` is mandatory for this algorithm. Commonly {0...1}"),
            r_g: *params.get("r_g").expect("param `r_g` is mandatory for this algorithm. Commonly {0...1}"),
            alpha: *params.get("alpha").expect("param `alpha` is mandatory for this algorithm. Commonly {0...1}"),
        };

        log::info!("tifu_hyperparameters: {:?}", tifu_hyperparameters);

        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);
        let shap_values = kmc_shapley_algorithm.compute_importance(&model, &metric_factory, &basket_train, &basket_valid);
        shap_values
    };

    convert_to_py_result(shap_values, if is_sbr { "session_id" } else { "user_id" })
}


#[pyfunction]
fn data_loo_polars(data: PyDataFrame, validation: PyDataFrame, model: &str, metric: &str,
                       params: HashMap<String, f64>, sustainable: PyDataFrame) -> PyResult<PyDataFrame> {

    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = match model.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let metric_config = parse_metric_config(metric);

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let k_loo_algorithm = KLoo::new();

    let loo_values:HashMap<u32, f64> = if is_sbr {
        let session_train = match polars_to_interactions(data_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let session_valid = match polars_to_interactions(validation_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let m = params.get("m").map(|&m| m as usize).expect("param `m` is mandatory for this algorithm. e.g. 500");
        let k = params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 250");

        let model:VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        let loo_importances = k_loo_algorithm.compute_importance(&model, &metric_factory, &session_train, &session_valid);
        loo_importances
    } else {
        let basket_train = match polars_to_purchases(data_df) {
            Ok(interactions) => {
                NextBasketDataset::from(&interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };
        let basket_valid = match polars_to_purchases(validation_df) {
            Ok(interactions) => {
                NextBasketDataset::from(&interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let tifu_hyperparameters = HyperParams {
            m: params.get("m").map(|&m| m as isize).expect("param `m` is mandatory for this algorithm. e.g. 500"),
            k: params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 500"),
            r_b: *params.get("r_b").expect("param `r_b` is mandatory for this algorithm. Commonly {0...1}"),
            r_g: *params.get("r_g").expect("param `r_g` is mandatory for this algorithm. Commonly {0...1}"),
            alpha: *params.get("alpha").expect("param `alpha` is mandatory for this algorithm. Commonly {0...1}"),
        };

        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);
        let loo_importances = k_loo_algorithm.compute_importance(&model, &metric_factory, &basket_train, &basket_valid);
        loo_importances
    };

    convert_to_py_result(loo_values, if is_sbr { "session_id" } else { "user_id" })
}


#[pyfunction]
fn train_and_evaluate_polars(data: PyDataFrame, validation: PyDataFrame, model: &str, metric: &str,
                   params: HashMap<String, f64>, sustainable: PyDataFrame) -> PyResult<PyDataFrame> {

    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = match model.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let metric_config = parse_metric_config(metric);

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);


    let metrics :Vec<(String, f64)> = if is_sbr {
        let session_train = match polars_to_interactions(data_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let session_valid = match polars_to_interactions(validation_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let m = params.get("m").map(|&m| m as usize).expect("param `m` is mandatory for this algorithm. e.g. 500");
        let k = params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 250");

        let model:VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        let validation_evaluation_metrics: Vec<(String, f64)> =
            evaluate_dataset(&model, &metric_factory, &session_valid);
        validation_evaluation_metrics


    } else {
        let basket_train = match polars_to_purchases(data_df) {
            Ok(interactions) => {
                NextBasketDataset::from(&interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };
        let basket_valid = match polars_to_purchases(validation_df) {
            Ok(interactions) => {
                NextBasketDataset::from(&interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
            }
        };

        let tifu_hyperparameters = HyperParams {
            m: params.get("m").map(|&m| m as isize).expect("param `m` is mandatory for this algorithm. e.g. 500"),
            k: params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 500"),
            r_b: *params.get("r_b").expect("param `r_b` is mandatory for this algorithm. Commonly {0...1}"),
            r_g: *params.get("r_g").expect("param `r_g` is mandatory for this algorithm. Commonly {0...1}"),
            alpha: *params.get("alpha").expect("param `alpha` is mandatory for this algorithm. Commonly {0...1}"),
        };

        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);
        let validation_evaluation_metrics: Vec<(String, f64)> =
            evaluate_dataset(&model, &metric_factory, &basket_valid);
        validation_evaluation_metrics
    };
    metrics_to_pydataframe(&metrics)
}

fn metrics_to_pydataframe(metrics: &Vec<(String, f64)>) -> PyResult<PyDataFrame> {
    let mut metric_name = Vec::with_capacity(metrics.len());
    let mut metric_score = Vec::with_capacity(metrics.len());

    for (name, score) in metrics.clone() {
        metric_name.push(name);
        metric_score.push(score);
    }

    let df = DataFrame::new(vec![
        polars::prelude::Column::Series(Series::new("metric".into(), &metric_name)),
        polars::prelude::Column::Series(Series::new("score".into(), &metric_score)),
    ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

// Reusable function: Convert importance values to PyDataFrame
fn convert_to_py_result(
    importance: HashMap<u32, f64>,
    id_column_name: &str,
) -> PyResult<PyDataFrame> {
    let mut ids = Vec::with_capacity(importance.len());
    let mut scores = Vec::with_capacity(importance.len());

    for (id, score) in importance {
        ids.push(id as i64);
        scores.push(score);
    }

    let df = DataFrame::new(vec![
        polars::prelude::Column::Series(Series::new(id_column_name.into(), &ids)),
        polars::prelude::Column::Series(Series::new("score".into(), &scores)),
    ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

// Reusable function: Parse metric configuration
fn parse_metric_config(metric: &str) -> MetricConfig {
    let (metric_name, at_k) = {
        let parts: Vec<&str> = metric.split('@').collect();
        (parts[0], parts[1].parse().expect("Failed to parse @k"))
    };

    let metric_type = match metric_name.to_lowercase().trim() {
        "f1score" => MetricType::F1score,
        "hitrate" => MetricType::HitRate,
        "mrr" => MetricType::MRR,
        "precision" => MetricType::Precision,
        "recall" => MetricType::Recall,
        "sustainablemrr" => MetricType::SustainableMrr,
        "sustainablendcg" => MetricType::SustainableNdcg,
        "st" => MetricType::SustainabilityCoverageTerm,
        "ndcg" => MetricType::Ndcg,
        _ => panic!("Invalid metric type: '{}'", metric_name),
    };

    let config = MetricConfig {
        importance_metric: metric_type.clone(),
        evaluation_metrics: vec![metric_type.clone()],
        length: at_k,
        alpha: 0.8,
    };

    config
}



/// A Python module implemented in Rust.
#[pymodule]
fn illoominate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    utils::init_logging();
    m.add_function(wrap_pyfunction!(debug_sbr, m)?)?;
    m.add_function(wrap_pyfunction!(data_shapley_polars, m)?)?;
    m.add_function(wrap_pyfunction!(data_loo_polars, m)?)?;
    m.add_function(wrap_pyfunction!(train_and_evaluate_polars, m)?)?;
    Ok(())
}
