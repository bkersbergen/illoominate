use std::collections::{HashMap, HashSet};
use chrono::Local;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use crate::importance::{Dataset, Importance};
use crate::importance::k_loo::KLoo;
use crate::importance::k_mc_shapley::KMcShapley;
use crate::nbr::tifuknn::io::polars_to_purchases;
use crate::nbr::tifuknn::TIFUKNN;
use crate::nbr::tifuknn::types::HyperParams;
use crate::nbr::types::NextBasketDataset;
use crate::sessrec::io::polars_to_interactions;
use crate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
use crate::sessrec::metrics::product_info::ProductInfo;
use crate::sessrec::types::{Interaction, SessionDataset};
use crate::sessrec::vmisknn::VMISKNN;

pub mod baselines;
pub mod conf;
pub mod importance;
pub mod nbr;
pub mod sessrec;
mod utils;

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
                params: HashMap<String, f64>) -> PyResult<PyDataFrame> {

    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();

    let is_sbr = match model.to_lowercase().as_str().trim() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let (metric, at_k): (&str, usize) = {
        let parts: Vec<&str> = metric.trim().split('@').take(2).collect();
        (parts[0], parts[1].parse().expect("Failed to parse at_k as a number"))
    };

    let metric_type = match metric.to_lowercase().as_str().trim() {
        "f1score" => MetricType::F1score,
        "hitrate" => MetricType::HitRate,
        "mrr" => MetricType::MRR,
        "precision" => MetricType::Precision,
        "recall" => MetricType::Recall,
        "responsiblemrr" => MetricType::ResponsibleMrr,
        "sustainabilitycoverage" => MetricType::SustainabilityCoverage,
        "ndcg" => MetricType::Ndcg,
        invalid => panic!("Invalid metric type: {}", invalid), // Include invalid value in panic message
    };

    let metric_config = MetricConfig {
        importance_metric: metric_type.clone(),
        evaluation_metrics: vec![metric_type],
        length: at_k,
        mrr_alpha: 0.8,
    };

    let product_info = ProductInfo::new(HashSet::new());
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
                SessionDataset::new(Vec::new())
            }
        };

        let session_valid = match polars_to_interactions(validation_df) {
            Ok(interactions) => {
                SessionDataset::new(interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                SessionDataset::new(Vec::new())
            }
        };

        let m = params.get("m").map(|&m| m as usize).expect("param `m` is mandatory for this algorithm. e.g. 500");
        let k = params.get("k").map(|&k| k as usize).expect("param `k` is mandatory for this algorithm. e.g. 250");


        let model:VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        // let loo_importances = k_loo_algorithm.compute_importance(&model, &metric_factory, session_train, session_valid);
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
                NextBasketDataset::from(&Vec::new())
            }
        };
        let basket_valid: NextBasketDataset = match polars_to_purchases(validation_df) {
            Ok(purchases) => {
                NextBasketDataset::from(&purchases)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                panic!("Failed to convert DataFrame: {}", e);
                NextBasketDataset::from(&Vec::new())
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

    // Initialize vectors for session_ids and shapley_values
    let mut ids = Vec::with_capacity(shap_values.len());
    let mut importance_values = Vec::with_capacity(shap_values.len());

    // Populate vectors in a consistent order
    for (key_id, shapley_value) in shap_values.iter() {
        ids.push(*key_id as i64);
        importance_values.push(*shapley_value);
    }

    // Create Polars Series from the vectors
    let id_column_name = if is_sbr { "session_id" } else { "user_id" };
    let session_ids_series = polars::prelude::Column::Series(Series::new(id_column_name.into(), &ids));
    let score_series = polars::prelude::Column::Series(Series::new("score".into(), &importance_values));

    // Create a DataFrame from the Series
    let df = DataFrame::new(vec![session_ids_series, score_series])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}


#[pyfunction]
fn data_loo_polars(data: PyDataFrame, validation: PyDataFrame, model: &str, metric: &str,
                       params: HashMap<String, f64>) -> PyResult<PyDataFrame> {

    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();

    let is_sbr = match model.to_lowercase().as_str() {
        "vmis" => true,
        "tifu" => false,
        invalid => panic!("Unknown model type: {}", invalid),
    };

    let (metric, at_k): (&str, usize) = {
        let parts: Vec<&str> = metric.split('@').take(2).collect();
        (parts[0], parts[1].parse().expect("Failed to parse at_k as a number"))
    };

    let metric_type = match metric.to_lowercase().as_str() {
        "f1score" => MetricType::F1score,
        "hitrate" => MetricType::HitRate,
        "mrr" => MetricType::MRR,
        "precision" => MetricType::Precision,
        "recall" => MetricType::Recall,
        "responsiblemrr" => MetricType::ResponsibleMrr,
        "sustainabilitycoverage" => MetricType::SustainabilityCoverage,
        "ndcg" => MetricType::Ndcg,
        invalid => panic!("Invalid metric type: {}", invalid), // Include invalid value in panic message
    };

    let metric_config = MetricConfig {
        importance_metric: metric_type.clone(),
        evaluation_metrics: vec![metric_type],
        length: at_k,
        mrr_alpha: 0.8,
    };

    let product_info = ProductInfo::new(HashSet::new());
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let k_loo_algorithm = KLoo::new();

    let loo_values:HashMap<u32, f64> = if is_sbr {
        let train = match polars_to_interactions(data_df) {
            Ok(interactions) => {
                interactions
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                Vec::new()
            }
        };

        let validation = match polars_to_interactions(validation_df) {
            Ok(interactions) => {
                interactions
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                Vec::new()
            }
        };
        let session_train = SessionDataset::new(train);
        let session_valid = SessionDataset::new(validation);

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
                NextBasketDataset::from(&Vec::new())
            }
        };
        let basket_valid = match polars_to_purchases(validation_df) {
            Ok(interactions) => {
                NextBasketDataset::from(&interactions)
            }
            Err(e) => {
                log::error!("Failed to convert DataFrame: {}", e);
                NextBasketDataset::from(&Vec::new())
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

    // Initialize vectors for session_ids and shapley_values
    let mut session_ids = Vec::with_capacity(loo_values.len());
    let mut shapley_values = Vec::with_capacity(loo_values.len());

    // Populate vectors in a consistent order
    for (session_id, loo_value) in loo_values.iter() {
        session_ids.push(*session_id as i64);
        shapley_values.push(*loo_value);
    }

    // Create Polars Series from the vectors
    let id_column_name = if is_sbr { "session_id" } else { "user_id" };
    let session_ids_series = polars::prelude::Column::Series(Series::new(id_column_name.into(), &session_ids));
    let score_series = polars::prelude::Column::Series(Series::new("score".into(), &shapley_values));

    // Create a DataFrame from the Series
    let df = DataFrame::new(vec![session_ids_series, score_series])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}


/// A Python module implemented in Rust.
#[pymodule]
fn illoominate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    utils::init_logging();
    m.add_function(wrap_pyfunction!(debug_sbr, m)?)?;
    m.add_function(wrap_pyfunction!(data_shapley_polars, m)?)?;
    m.add_function(wrap_pyfunction!(data_loo_polars, m)?)?;
    Ok(())
}
