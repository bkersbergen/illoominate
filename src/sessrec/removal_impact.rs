use crate::importance::{evaluate_dataset, Dataset};
use crate::metrics::MetricFactory;
use crate::sessrec::types::{SessionDataset, SessionId};
use crate::sessrec::vmisknn::VMISKNN;


use std::cmp::max;

use std::fs::{File, OpenOptions};
use std::io::Write;

pub fn create_file(name: &str, overwrite: bool) -> File {
    let mut options = OpenOptions::new();
    options.write(true).append(false);

    if overwrite {
        options.create(true); // Create the file if it doesn't exist, and overwrite if it does
    } else {
        options.create_new(true); // Create a new file, and fail if the file already exists
    }

    options.open(name).expect("Failed to create file")
}

pub fn vmis_evaluate_removal_impact<D: Dataset + Sync>(
    exp: &str,
    metric_factory: &MetricFactory,
    train: &SessionDataset,
    valid: &D,
    test: &D,
    keys_to_remove: &[SessionId],
    m: usize,
    k: usize,
    seed: usize,
    qty_impact_resolution: usize,
    output_files_evaluation_metric_results: &mut Vec<File>,
) {
    let mut train = train.clone();
    assert!(train.len() >= keys_to_remove.len());

    let retrained_model = VMISKNN::fit_dataset(&train, m, k, 1.0);

    let validation_evaluation_metrics: Vec<(String, f64)> =
        evaluate_dataset(&retrained_model, metric_factory, valid);
    let test_evaluation_metrics: Vec<(String, f64)> =
        evaluate_dataset(&retrained_model, metric_factory, test);

    for (
        (output_file, (_valid_metric_name, valid_metric_score)),
        (_test_metric_name, test_metric_score),
    ) in output_files_evaluation_metric_results
        .iter_mut()
        .zip(validation_evaluation_metrics.iter())
        .zip(test_evaluation_metrics.iter())
    {
        let data_to_append = format!(
            "{},{},{:.4},{:.4},{}",
            exp, seed, valid_metric_score, test_metric_score, 0
        );
        log::info!("{}", data_to_append);
        writeln!(output_file, "{}", data_to_append).expect("Results Failed to write to file");
    }

    let mut num_sessions_removed = 0;

    let impact_interval = max(
        1,
        (keys_to_remove.len() as f64 / qty_impact_resolution as f64) as usize,
    );
    for key in keys_to_remove {
        // Remove SessionId from the Training data
        if train.sessions.remove(key).is_some() {
            num_sessions_removed += 1;
            let compute_impact =
                (train.sessions.len() == 1) || (num_sessions_removed % impact_interval == 0);
            if compute_impact {
                // evaluate on validation data and write output
                let retrained_model = VMISKNN::fit_dataset(&train, m, k, 1.0);

                let validation_evaluation_metrics: Vec<(String, f64)> =
                    evaluate_dataset(&retrained_model, metric_factory, valid);
                let test_evaluation_metrics: Vec<(String, f64)> =
                    evaluate_dataset(&retrained_model, metric_factory, test);

                for (
                    (output_file, (_valid_metric_name, valid_metric_score)),
                    (_test_metric_name, test_metric_score),
                ) in output_files_evaluation_metric_results
                    .iter_mut()
                    .zip(validation_evaluation_metrics.iter())
                    .zip(test_evaluation_metrics.iter())
                {
                    let data_to_append = format!(
                        "{},{},{:.4},{:.4},{}",
                        exp, seed, valid_metric_score, test_metric_score, num_sessions_removed
                    );
                    log::info!("{}", data_to_append);
                    writeln!(output_file, "{}", data_to_append)
                        .expect("Results Failed to write to file");
                }
            }
        };
    }
}
