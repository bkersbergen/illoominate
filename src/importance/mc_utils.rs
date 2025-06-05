use crate::importance::{Dataset, RetrievalBasedModel};
use crate::metrics::{Metric, MetricFactory};
use crate::sessrec::types::{SessionId};
use crate::sessrec::vmisknn::Scored;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub fn permutation<D: Dataset>(dataset: &D, seed: usize, iteration: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = dataset.collect_keys();
    let mut rng = StdRng::seed_from_u64((seed + iteration) as u64);

    ids.shuffle(&mut rng);
    ids
}

pub fn mean_tolerance_score_dataset<R: RetrievalBasedModel, D: Dataset>(
    model: &R,
    metric_factory: &MetricFactory,
    test: &D,
) -> (f64, f64) {
    // "Computes the average performance and its error using bagging."
    let mut scores = Vec::new();
    let keys = &test.collect_keys();
    let seed = 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let how_many = metric_factory.config.length;

    let mut metric_binding = metric_factory.create_importance_metric();
    let metric = metric_binding.as_mut();
    for _ in 0..100 {
        let bag_idx: Vec<_> = (0..keys.len())
            .map(|_| keys[rng.gen_range(0..keys.len())])
            .collect();
        metric.reset();
        bag_idx.iter().for_each(|&key| {
            let dataset_entry = test.__get_entry__(key);
            for sequence in dataset_entry.sequences {
                let query = sequence.input;
                let actual_next_items = sequence.target;
                let neighbors = model.retrieve_k(&query);
                let recommended_items = model.predict(&query, &neighbors, how_many);
                metric.add(&recommended_items, &actual_next_items);
            }
        });
        scores.push(metric.result())
    }
    let (mean_score, stddev_score) = mean_stddev(&scores);
    (mean_score, stddev_score)
}

pub fn random_score_dataset<R: RetrievalBasedModel + Sync, D: Dataset + Sync>(
    model: &R,
    metric_factory: &MetricFactory,
    heldout: &D,
) -> (f64, f64) {
    let keys: Vec<u32> = {
        let mut k = heldout.collect_keys();
        k.sort();
        k
    };
    // Give the value of on initial untrained model
    // We permute the clicks in the `heldout` dataset.
    // Then evaluate 100 times and take the average of those runs as a result.
    let all_heldout_values: Vec<_> = keys
        .iter()
        .flat_map(|&key| heldout.__get_items__(key))
        .sorted()
        .collect();

    let scores: Vec<f64> = (0..100)
        .into_par_iter()
        .enumerate()
        .map(|(iteration, _)| {
            let mut local_heldout_values = all_heldout_values.clone();
            let mut local_keys = keys.clone();

            let seed = (iteration as u64) + 1; // Unique seed for each iteration
            let mut local_rng = StdRng::seed_from_u64(seed);
            local_keys.shuffle(&mut local_rng);
            local_heldout_values.shuffle(&mut local_rng);

            let mut metric_binding: Box<dyn Metric + Send + Sync> =
                metric_factory.create_importance_metric();
            let metric = metric_binding.as_mut();

            let mut idx = 0;

            local_keys.iter().for_each(|&key| {
                let items = heldout.__get_items__(key);
                let max = idx + items.len();
                let permuted_session_items = local_heldout_values[idx..max].to_vec();
                idx = max;
                for pos in 1..permuted_session_items.len() {
                    let query = &permuted_session_items[0..pos]
                        .iter()
                        .map(|&id| Scored::new(id, 1.0))
                        .collect_vec();

                    let neighbors = model.retrieve_k(query);
                    let recommended_items = model.predict(query, &neighbors, 21);
                    let actual_next_items = &permuted_session_items[pos..]
                        .iter()
                        .map(|&id| Scored::new(id, 1.0))
                        .collect_vec();

                    metric.add(&recommended_items, actual_next_items);
                }
            });
            metric.result()
        })
        .collect();

    let (mean_score, stddev_score) = mean_stddev(&scores);
    (mean_score, stddev_score)
}

pub fn mean_stddev(data: &[f64]) -> (f64, f64) {
    fn mean(data: &[f64]) -> f64 {
        let sum: f64 = data.iter().sum();
        sum / data.len() as f64
    }

    let meanval = mean(data);
    let variance: f64 =
        data.iter().map(|&x| (x - meanval).powi(2)).sum::<f64>() / data.len() as f64;
    (meanval, variance.sqrt())
}

pub fn error_dataset(mem: &HashMap<SessionId, Vec<f64>>, qty_minimal_mc_iterations: usize) -> f64 {
    // The Rust equivalent for this code. Note in python mem are numpy arrays like Datapoint:
    // def error(mem):
    //     if len(mem) < 100:
    //         return 1.0
    //     all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    //     errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    //     return np.max(errors)
    //

    let (mem_as_datapoints, _idx_to_key_mapping) = convert_session_hashmap_to_datapoints(mem);
    error(&mem_as_datapoints, qty_minimal_mc_iterations)
}


pub(crate) fn error(mem: &Vec<Vec<f64>>, qty_minimal_mc_iterations: usize) -> f64 {
    if qty_minimal_mc_iterations < 100 {
        log::warn!("Warning: `qty_minimal_mc_iterations` should be 100 for real experiments.");
    }
    if mem.len() < qty_minimal_mc_iterations {
        log::debug!("mem length {} is below qty_minimal_mc_iterations: {}. Returning 1.0 error", mem.len(), qty_minimal_mc_iterations);
        return 1.0;
    }

    // Calculate cumulative sums (cumsum)
    let cumsum: Vec<Vec<f64>> = cumsum(mem);

    // Compute all_vals by dividing cumulative sums by the number of iterations so far
    let all_vals: Vec<Vec<f64>> = cumsum
        .into_par_iter()
        .enumerate()
        .map(|(i, session_importances)| {
            session_importances
                .into_iter()
                .map(|val| val / (i + 1) as f64)
                .collect()
        })
        .collect();

    // Get the start index of the last 100 rows
    let start_index = if all_vals.len() > 100 {
        log::debug!("mem length {} > 100 using last 100 iterations", mem.len());
        all_vals.len() - 100
    } else {
        0
    };

    // Retrieve the last vector in `all_vals` for standardization
    let max_value = if let Some(last_importance_for_all_sessions) = all_vals.last() {
        let last_importance_for_all_sessions = last_importance_for_all_sessions.clone();

        // Calculate the max error across the last 100 rows in parallel
        all_vals[start_index..]
            .par_iter()
            .filter_map(|importance_for_sessions_run| {
                // Normalize each session importance with respect to the last vector
                let valid_values: Vec<f64> = importance_for_sessions_run
                    .iter()
                    .zip(&last_importance_for_all_sessions)
                    .map(|(importance, &last_importance)| {
                        let divisor = if last_importance.abs() < 1e-12 {
                            1e-12
                        } else {
                            last_importance.abs()
                        };

                        ((*importance - last_importance) / divisor + 1e-12).abs()
                    })
                    .filter(|&x| !x.is_nan())
                    .collect();

                // Calculate mean of valid values if non-empty
                if !valid_values.is_empty() {
                    Some(valid_values.iter().sum::<f64>() / valid_values.len() as f64)
                } else {
                    None
                }
            })
            // Filter out any remaining None values
            .filter_map(Some)
            .max_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    } else {
        log::error!("no last MC iteration data");
        0.0
    };

    max_value
}
fn cumsum(v: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_columns = v[0].len();
    let num_rows = v.len();

    // Calculate cumulative sums in parallel for each column
    let column_sums: Vec<Vec<f64>> = (0..num_columns)
        .into_par_iter()
        .map(|col_idx| {
            let mut column_cumsum = Vec::with_capacity(num_rows);
            let mut sum = 0.0;
            for row_idx in 0..num_rows {
                sum += v[row_idx][col_idx];
                column_cumsum.push(sum);
            }
            column_cumsum
        })
        .collect();

    // Transpose column_sums to get the final cumulative_sums matrix
    let mut cumulative_sums = vec![vec![0.0; num_columns]; num_rows];
    for col_idx in 0..num_columns {
        for row_idx in 0..num_rows {
            cumulative_sums[row_idx][col_idx] = column_sums[col_idx][row_idx];
        }
    }

    cumulative_sums
}

pub(crate) fn convert_session_hashmap_to_datapoints(
    a: &HashMap<SessionId, Vec<f64>>,
) -> (Vec<Vec<f64>>, HashMap<usize, SessionId>) {
    // Determine the length of each vector (assuming all vectors in `a` have the same length)
    let inner_len = a.values().next().map_or(0, |v| v.len());
    let outer_len = a.len();

    // Create the `idx_to_key_mapping` in parallel
    let idx_to_key_mapping: HashMap<usize, SessionId> = a.keys()
        .enumerate()
        .map(|(idx, &key)| (idx, key))
        .collect();

    // Initialize the `result` vector with the required size and capacity
    let mut result: Vec<Vec<f64>> = vec![Vec::with_capacity(outer_len); inner_len];

    // Collect intermediate results in parallel
    let intermediate: Vec<Vec<f64>> = (0..outer_len)
        .into_par_iter()
        .map(|idx| {
            let key = idx_to_key_mapping.get(&idx).unwrap();
            a.get(key).unwrap().clone()
        })
        .collect();

    // Merge intermediate results into `result`
    for values in intermediate {
        for (i, value) in values.into_iter().enumerate() {
            result[i].push(value);
        }
    }

    (result, idx_to_key_mapping)
}



#[cfg(test)]
mod error_test {
    use super::*;

    #[test]
    fn should_datapoint_cumsum() {
        let v: Vec<Vec<f64>> = vec![
            vec![0.3, 0.2, 0.5],
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.2, 0.1],
            vec![0.5, 0.5, 0.1],
        ];
        let result = cumsum(&v);
        let expected = vec![
            [0.3, 0.2, 0.5],
            [0.5, 0.7, 0.8],
            [0.6, 0.8999999999999999, 0.9],
            [1.1, 1.4, 1.0],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn should_convergence_error_same_as_python() {
        // each vector represents one shapley iteration.
        let v: Vec<Vec<f64>> = vec![
            vec![0.3, 0.2, 0.5],
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.2, 0.1],
            vec![0.5, 0.5, 0.1],
        ];
        let error = error(&v, 1);
        let python_error = 0.5064935064916548;
        assert!((python_error - error).abs() < 0.00000000001);
    }

    #[test]
    fn should_proof_same_error_datapoint_vs_dataset() {
        let my_datapoint: Vec<Vec<f64>> = vec![
            vec![0.3, 0.2, 0.5],
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.2, 0.1],
            vec![0.5, 0.5, 0.1],
        ];
        let datapoint_error = error(&my_datapoint, 1);

        let mut my_dataset = HashMap::new();
        my_dataset.insert(1 as SessionId, vec![0.3, 0.2, 0.1, 0.5]);
        my_dataset.insert(2 as SessionId, vec![0.2, 0.5, 0.2, 0.5]);
        my_dataset.insert(3 as SessionId, vec![0.5, 0.3, 0.1, 0.1]);

        let dataset_error = error_dataset(&my_dataset, 1);

        log::info!("datapoint_error: {:?}", datapoint_error);
        log::info!("dataset_error: {:?}", dataset_error);
        assert_eq!(datapoint_error, dataset_error);
    }
}
