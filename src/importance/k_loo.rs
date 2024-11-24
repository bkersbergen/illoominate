use std::cmp::min;
use std::collections::HashMap;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::importance::{Dataset, Importance, RetrievalBasedModel};
use crate::sessrec::metrics::MetricFactory;

pub struct KLoo {}

impl Default for KLoo {
    fn default() -> Self {
        Self::new()
    }
}

impl KLoo {
    pub fn new() -> Self {
        Self {}
    }
}

#[allow(non_snake_case)]
impl Importance for KLoo {
    fn compute_importance<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
        &self,
        model: &R,
        metric_factory: &MetricFactory,
        train: &D,
        valid: &D,
    ) -> HashMap<u32, f64> {
        let qty_evaluations = valid.num_interactions();
        let qty_keys = train.collect_keys().len();
        let num_recommendations = metric_factory.config.length;

        let contributions = valid
            .collect_keys()
            .par_iter()
            .map(|&key| {
                let mut local_contributions: Vec<f64> = vec![0.0; qty_keys];

                let entry = valid.__get_entry__(key);
                for interaction in entry.sequences {
                    let query_session = interaction.input;
                    let actual_next_items = interaction.target;

                    let N_q = model.retrieve_all(&query_session);
                    let kp = min(model.k(), N_q.len());

                    let loo_neighbors = &N_q[0..kp].to_vec();
                    let recommended_items =
                        model.predict(&query_session, loo_neighbors, num_recommendations);
                    let metric_binding = metric_factory.create_importance_metric();
                    let metric = metric_binding.as_ref();

                    let original_utility = metric.compute(&recommended_items, &actual_next_items);
                    let enough_neighbors = N_q.len() > model.k();
                    for leave_out_index in 0..kp {
                        let loo_neighbors = if enough_neighbors {
                            [
                                &N_q[0..leave_out_index],
                                &N_q[(leave_out_index + 1)..(kp + 1)],
                            ]
                            .concat()
                        } else {
                            [&N_q[0..leave_out_index], &N_q[(leave_out_index + 1)..kp]].concat()
                        };

                        let recommended_items =
                            model.predict(&query_session, &loo_neighbors, num_recommendations);

                        let score = metric.compute(&recommended_items, &actual_next_items);

                        let training_key_id = N_q[leave_out_index].id as usize;
                        let importance = (original_utility - score) / qty_evaluations as f64;
                        local_contributions[training_key_id] += importance
                    }
                }
                local_contributions
            })
            .reduce_with(|left, right| {
                left.into_iter()
                    .zip(right)
                    .map(|(x, y)| x + y)
                    .collect()
            })
            .unwrap_or_else(|| vec![0.0; qty_keys]);

        let mut result = HashMap::with_capacity(contributions.len());
        for session_id in train.collect_keys() {
            let marginal_contribution = contributions[session_id as usize];
            result
                .entry(session_id)
                .and_modify(|value| *value += marginal_contribution)
                .or_insert(marginal_contribution);
        }
        result
    }
}
