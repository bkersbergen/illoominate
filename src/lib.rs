use crate::importance::k_loo::KLoo;
use crate::importance::k_mc_shapley::KMcShapley;
use crate::importance::{evaluate_dataset, Importance};
use crate::metrics::product_info::ProductInfo;
use crate::metrics::{MetricConfig, MetricFactory, MetricType};
use crate::nbr::tifuknn::io::next_basket_dataset_from_polars;
use crate::nbr::tifuknn::types::HyperParams;
use crate::nbr::tifuknn::TIFUKNN;
use crate::sessrec::io::{
    dense_session_dataset_from_file, dense_session_dataset_from_polars, get_sustainable_items,
    read_sustainable_products_info, session_dataset_from_polars,
};
use crate::sessrec::types::{Interaction, ItemId};
use crate::sessrec::vmisknn::VMISKNN;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::{HashMap, HashSet};

pub mod baselines;
pub mod conf;
pub mod importance;
pub mod metrics;
pub mod nbr;
pub mod sessrec;
mod utils;

#[pyfunction]
fn debug_sbr(pydf: PyDataFrame) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();

    // Pre-fetch columns to avoid repeated lookups
    let session_id_col = df
        .column("session_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let item_id_col = df
        .column("item_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let timestamp_col = df
        .column("timestamp")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Pre-allocate memory for results
    let mut results = Vec::with_capacity(df.height());

    // Iterate by row index, directly accessing each column
    for i in 0..df.height() {
        let session_id = match session_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Expected u64 in session_id column",
                ))
            }
        };
        let item_id = match item_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Expected u64 in item_id column",
                ))
            }
        };
        let timestamp = match timestamp_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Expected u64 in timestamp column",
                ))
            }
        };

        // Create an Interaction instance and store it
        let interaction = Interaction::new(session_id, item_id, timestamp);
        results.push(interaction);
    }

    let session_ids =
        polars::prelude::Column::Series(Series::new("session_id".into(), &[606u64, 2, 3, 4, 607]));
    let item_ids = polars::prelude::Column::Series(Series::new(
        "item_id".into(),
        &[107u64, 102, 113, 104, 105],
    ));
    let timestamps = polars::prelude::Column::Series(Series::new(
        "timestamp".into(),
        &[
            1609459200u64,
            1609459260,
            1609459320,
            1609459380,
            1609459440,
        ],
    ));

    // Create a DataFrame from the series
    let df = DataFrame::new(vec![session_ids, item_ids, timestamps])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

#[pyfunction]
fn data_shapley_polars(
    data: PyDataFrame,
    validation: PyDataFrame,
    model: &str,
    metric: &str,
    params: HashMap<String, f64>,
    sustainable: PyDataFrame,
) -> PyResult<PyDataFrame> {
    log::info!("Starting Data Shapley computation for model='{model}' metric='{metric}'");
    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = parse_model(model)?;

    let metric_config = parse_metric_config(metric)?;

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let convergence_threshold = params.get("convergence_threshold").copied().unwrap_or(0.1);
    let max_iterations = parse_usize_param(&params, "max_iterations", 1_000)?;
    let convergence_check_every = parse_usize_param(&params, "convergence_check_every", 10)?;
    let min_iterations = parse_usize_param(&params, "min_iterations", 100)?;
    let seed = parse_usize_param(&params, "seed", 42)?;

    let kmc_shapley_algorithm = KMcShapley::with_convergence(
        convergence_threshold,
        max_iterations,
        convergence_check_every,
        min_iterations,
        seed,
    );

    let shap_values: HashMap<u32, f64> = if is_sbr {
        log::info!("Preparing session datasets for Data Shapley");
        let session_train = session_dataset_from_polars(&data_df)?;
        let session_valid = session_dataset_from_polars(&validation_df)?;

        let m = required_usize_param(&params, "m", "500")?;
        let k = required_usize_param(&params, "k", "250")?;
        log::info!("Fitting VMIS model for Data Shapley with m={m} k={k}");
        let model: VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);
        log::info!("Running KMC-Shapley iterations");
        kmc_shapley_algorithm.compute_importance(
            &model,
            &metric_factory,
            &session_train,
            &session_valid,
        )
    } else {
        log::info!("Preparing next-basket datasets for Data Shapley");
        let basket_train = next_basket_dataset_from_polars(&data_df)?;
        let basket_valid = next_basket_dataset_from_polars(&validation_df)?;
        let tifu_hyperparameters = parse_tifu_hyperparameters(&params)?;
        log::info!("Fitting TIFU-kNN model for Data Shapley");
        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);
        log::info!("Running KMC-Shapley iterations");
        kmc_shapley_algorithm.compute_importance(
            &model,
            &metric_factory,
            &basket_train,
            &basket_valid,
        )
    };
    log::info!("Data Shapley computation finished");

    convert_to_py_result(shap_values, if is_sbr { "session_id" } else { "user_id" })
}

#[pyfunction]
fn data_loo_polars(
    data: PyDataFrame,
    validation: PyDataFrame,
    model: &str,
    metric: &str,
    params: HashMap<String, f64>,
    sustainable: PyDataFrame,
) -> PyResult<PyDataFrame> {
    log::info!("Starting leave-one-out computation for model='{model}' metric='{metric}'");
    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = parse_model(model)?;

    let metric_config = parse_metric_config(metric)?;

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let k_loo_algorithm = KLoo::new();

    let loo_values: HashMap<u32, f64> = if is_sbr {
        log::info!("Preparing session datasets for leave-one-out");
        let session_train = session_dataset_from_polars(&data_df)?;
        let session_valid = session_dataset_from_polars(&validation_df)?;
        let m = required_usize_param(&params, "m", "500")?;
        let k = required_usize_param(&params, "k", "250")?;

        log::info!("Fitting VMIS model for leave-one-out with m={m} k={k}");
        let model: VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        log::info!("Evaluating leave-one-out importance");
        k_loo_algorithm.compute_importance(&model, &metric_factory, &session_train, &session_valid)
    } else {
        log::info!("Preparing next-basket datasets for leave-one-out");
        let basket_train = next_basket_dataset_from_polars(&data_df)?;
        let basket_valid = next_basket_dataset_from_polars(&validation_df)?;
        let tifu_hyperparameters = parse_tifu_hyperparameters(&params)?;

        log::info!("Fitting TIFU-kNN model for leave-one-out");
        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);
        log::info!("Evaluating leave-one-out importance");
        k_loo_algorithm.compute_importance(&model, &metric_factory, &basket_train, &basket_valid)
    };

    log::info!("Leave-one-out computation finished");

    convert_to_py_result(loo_values, if is_sbr { "session_id" } else { "user_id" })
}

#[pyfunction]
fn train_and_evaluate_polars(
    data: PyDataFrame,
    validation: PyDataFrame,
    model: &str,
    metric: &str,
    params: HashMap<String, f64>,
    sustainable: PyDataFrame,
) -> PyResult<PyDataFrame> {
    log::info!("Starting train-and-evaluate for model='{model}' metric='{metric}'");
    let data_df: DataFrame = data.into();
    let validation_df: DataFrame = validation.into();
    let sustainable_df: DataFrame = sustainable.into();

    let is_sbr = parse_model(model)?;

    let metric_config = parse_metric_config(metric)?;

    let sustainable_products: HashSet<ItemId> = get_sustainable_items(sustainable_df);

    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let metrics: Vec<(String, f64)> = if is_sbr {
        log::info!("Preparing session datasets for evaluation");
        let session_train = dense_session_dataset_from_polars(&data_df)?;
        let session_valid = dense_session_dataset_from_polars(&validation_df)?;
        let m = required_usize_param(&params, "m", "500")?;
        let k = required_usize_param(&params, "k", "250")?;
        log::info!("Fitting VMIS model with m={m} k={k}");
        let model: VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);

        log::info!("Evaluating validation dataset");
        evaluate_dataset(&model, &metric_factory, &session_valid)
    } else {
        log::info!("Preparing next-basket datasets for evaluation");
        let basket_train = next_basket_dataset_from_polars(&data_df)?;
        let basket_valid = next_basket_dataset_from_polars(&validation_df)?;
        let tifu_hyperparameters = parse_tifu_hyperparameters(&params)?;
        log::info!("Fitting TIFU-kNN model");
        let model: TIFUKNN = TIFUKNN::new(&basket_train, &tifu_hyperparameters);

        log::info!("Evaluating validation dataset");
        evaluate_dataset(&model, &metric_factory, &basket_valid)
    };
    log::info!("Train-and-evaluate finished");
    metrics_to_pydataframe(&metrics)
}

#[pyfunction]
fn train_and_evaluate_sbr_file(
    train_path: &str,
    validation_path: &str,
    metric: &str,
    params: HashMap<String, f64>,
    separator: &str,
    sustainable_path: Option<&str>,
) -> PyResult<PyDataFrame> {
    log::info!("Starting file-based train-and-evaluate metric='{metric}'");
    let metric_config = parse_metric_config(metric)?;
    let sustainable_products = sustainable_path
        .map(read_sustainable_products_info)
        .unwrap_or_default();
    let product_info = ProductInfo::new(sustainable_products);
    let metric_factory = MetricFactory::new(&metric_config, product_info);

    let delimiter = parse_delimiter(separator)?;
    log::info!("Loading training sessions from '{train_path}'");
    let session_train = dense_session_dataset_from_file(train_path, delimiter)
        .map_err(|e| py_value_error(e.to_string()))?;
    log::info!("Loading validation sessions from '{validation_path}'");
    let session_valid = dense_session_dataset_from_file(validation_path, delimiter)
        .map_err(|e| py_value_error(e.to_string()))?;
    let m = required_usize_param(&params, "m", "500")?;
    let k = required_usize_param(&params, "k", "250")?;
    log::info!("Fitting VMIS model with m={m} k={k}");
    let model: VMISKNN = VMISKNN::fit_dataset(&session_train, m, k, 1.0);
    log::info!("Evaluating validation dataset");
    let metrics = evaluate_dataset(&model, &metric_factory, &session_valid);
    log::info!("File-based train-and-evaluate finished");

    metrics_to_pydataframe(&metrics)
}

fn metrics_to_pydataframe(metrics: &[(String, f64)]) -> PyResult<PyDataFrame> {
    let mut metric_name = Vec::with_capacity(metrics.len());
    let mut metric_score = Vec::with_capacity(metrics.len());

    for (name, score) in metrics {
        metric_name.push(name.clone());
        metric_score.push(*score);
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
fn parse_metric_config(metric: &str) -> PyResult<MetricConfig> {
    let (metric_name, at_k) = {
        let parts: Vec<&str> = metric.split('@').collect();
        if parts.len() != 2 {
            return Err(py_value_error(format!(
                "Metric must follow the pattern '<name>@<k>', got '{metric}'"
            )));
        }
        let at_k = parts[1]
            .parse()
            .map_err(|_| py_value_error(format!("Failed to parse @k from '{metric}'")))?;
        (parts[0], at_k)
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
        _ => {
            return Err(py_value_error(format!(
                "Invalid metric type: '{metric_name}'"
            )))
        }
    };

    Ok(MetricConfig {
        importance_metric: metric_type.clone(),
        evaluation_metrics: vec![metric_type.clone()],
        length: at_k,
        alpha: 0.8,
    })
}

fn parse_model(model: &str) -> PyResult<bool> {
    match model.trim().to_lowercase().as_str() {
        "vmis" => Ok(true),
        "tifu" => Ok(false),
        invalid => Err(py_value_error(format!("Unknown model type: {invalid}"))),
    }
}

fn parse_usize_param(params: &HashMap<String, f64>, key: &str, default: usize) -> PyResult<usize> {
    let value = params.get(key).copied().unwrap_or(default as f64);
    if value < 0.0 {
        return Err(py_value_error(format!(
            "Parameter '{key}' must be non-negative"
        )));
    }
    Ok(value as usize)
}

fn required_usize_param(
    params: &HashMap<String, f64>,
    key: &str,
    example: &str,
) -> PyResult<usize> {
    params
        .get(key)
        .copied()
        .ok_or_else(|| {
            py_value_error(format!(
                "Parameter '{key}' is mandatory. Example: {example}"
            ))
        })
        .and_then(|value| {
            if value < 0.0 {
                Err(py_value_error(format!(
                    "Parameter '{key}' must be non-negative"
                )))
            } else {
                Ok(value as usize)
            }
        })
}

fn required_f64_param(params: &HashMap<String, f64>, key: &str) -> PyResult<f64> {
    params
        .get(key)
        .copied()
        .ok_or_else(|| py_value_error(format!("Parameter '{key}' is mandatory")))
}

fn parse_tifu_hyperparameters(params: &HashMap<String, f64>) -> PyResult<HyperParams> {
    Ok(HyperParams {
        m: required_usize_param(params, "m", "500")? as isize,
        k: required_usize_param(params, "k", "500")?,
        r_b: required_f64_param(params, "r_b")?,
        r_g: required_f64_param(params, "r_g")?,
        alpha: required_f64_param(params, "alpha")?,
    })
}

fn py_value_error(message: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.to_string())
}

fn parse_delimiter(separator: &str) -> PyResult<u8> {
    let bytes = separator.as_bytes();
    if bytes.len() != 1 {
        return Err(py_value_error(format!(
            "Expected a single-character separator, got '{separator}'"
        )));
    }
    Ok(bytes[0])
}

/// A Python module implemented in Rust.
#[pymodule]
fn illoominate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    utils::init_logging();
    m.add_function(wrap_pyfunction!(debug_sbr, m)?)?;
    m.add_function(wrap_pyfunction!(data_shapley_polars, m)?)?;
    m.add_function(wrap_pyfunction!(data_loo_polars, m)?)?;
    m.add_function(wrap_pyfunction!(train_and_evaluate_polars, m)?)?;
    m.add_function(wrap_pyfunction!(train_and_evaluate_sbr_file, m)?)?;
    Ok(())
}
