use crate::importance::{Dataset, evaluate_dataset};
use crate::nbr::tifuknn::types::{HyperParams, UserId};
use crate::nbr::tifuknn::TIFUKNN;
use crate::nbr::types::NextBasketDataset;
use crate::sessrec::metrics::MetricFactory;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::cmp::max;
use std::collections::{HashMap};
use std::fs::File;
use std::io::Write;





pub fn tifu_evaluate_removal_impact(
    experiment_type: &str,
    metric_factory: &MetricFactory,
    training_baskets: &NextBasketDataset,
    valid: &NextBasketDataset,
    test: &NextBasketDataset,
    keys_to_remove: &[UserId],
    hp: &HyperParams,
    seed: usize,
    qty_impact_resolution: usize,
    output_files_evaluation_metric_results: &mut Vec<File>,
) {
    let mut training_baskets = training_baskets.clone();

    assert!(training_baskets.len() >= keys_to_remove.len());

    let model = TIFUKNN::new(&training_baskets, hp);


    let validation_evaluation_metrics: Vec<(String, f64)> =
        evaluate_dataset(&model, metric_factory, valid);
    let test_evaluation_metrics: Vec<(String, f64)> =
        evaluate_dataset(&model, metric_factory, test);

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
            experiment_type, seed, valid_metric_score, test_metric_score, 0
        );
        log::info!("{}", data_to_append);
        writeln!(output_file, "{}", data_to_append).expect("Results Failed to write to file");
    }

    let mut num_users_removed = 0;

    let impact_interval = max(
        1,
        (keys_to_remove.len() as f64 / qty_impact_resolution as f64) as usize,
    );
    log::debug!("start removing users in evaluate_removal_impact");
    for key in keys_to_remove {
        if training_baskets.user_baskets.remove(key).is_some() {
            num_users_removed += 1;
            let compute_impact =
                (training_baskets.len() == 1) || (num_users_removed % impact_interval == 0);
            if compute_impact {
                // evaluate on validation data and write output
                let model = TIFUKNN::new(&training_baskets, hp);
                let validation_evaluation_metrics: Vec<(String, f64)> =
                    evaluate_dataset(&model, metric_factory, valid);
                let test_evaluation_metrics: Vec<(String, f64)> =
                    evaluate_dataset(&model, metric_factory, test);

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
                        experiment_type,
                        seed,
                        valid_metric_score,
                        test_metric_score,
                        num_users_removed
                    );
                    log::info!("{}", data_to_append);
                    writeln!(output_file, "{}", data_to_append)
                        .expect("Results Failed to write to file");
                }
            }
        };
    }
    log::debug!("end removing users in evaluate_removal_impact");
}


pub fn split_train_eval(
    all_baskets_by_user: NextBasketDataset,
    validation_ratio: f64,
) -> (
    NextBasketDataset,
    NextBasketDataset,
    NextBasketDataset,
) {
    let mut train_baskets_by_user = HashMap::new();
    let mut heldout_baskets_by_user = Vec::new();

    for (user_id, mut baskets) in all_baskets_by_user.user_baskets {
        if baskets.len() > 1 {
            let last_basket = baskets.pop().unwrap();
            train_baskets_by_user.insert(user_id, baskets);
            heldout_baskets_by_user.push((user_id, last_basket));
        }
    }
    let mut rng = thread_rng();
    heldout_baskets_by_user.shuffle(&mut rng);

    // Calculate the split point based on the validation_ratio
    let split_index = (heldout_baskets_by_user.len() as f64 * validation_ratio).round() as usize;

    let (left, right) = heldout_baskets_by_user.split_at(split_index);

    let mut evaluate = HashMap::new();
    for (k, v) in left {
        evaluate.insert(*k, vec![v.clone()]);
    }

    let mut test = HashMap::new();
    for (k, v) in right {
        test.insert(*k, vec![v.clone()]);
    }

    (NextBasketDataset::from(&train_baskets_by_user), NextBasketDataset::from(&evaluate), NextBasketDataset::from(&test))
}
