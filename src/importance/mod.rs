use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::{Send, Sync};
use std::sync::Mutex;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::{IntoParallelIterator, ParallelSliceMut};

use crate::metrics::{Metric, MetricFactory};
use crate::sessrec::types::{SessionId, Time};
use crate::sessrec::vmisknn::Scored;

pub mod candidate_neighbors;
pub mod k_loo;
pub mod k_mc_shapley;
pub mod mc_utils;

pub trait RetrievalBasedModel {
    fn k(&self) -> usize;

    // retrieve k neighbors
    fn retrieve_k(&self, query_session: &Vec<Scored>) -> Vec<Scored>;

    // retrieve all neighbors session or user_ids
    fn retrieve_all(&self, query_session: &Vec<Scored>) -> Vec<Scored>;

    fn create_preaggregate(&self) -> HashMap<u64, f64>;

    fn add_to_preaggregate<T: AsRef<Scored>>(
        &self,
        agg: &mut HashMap<u64, f64>,
        query_session: &Vec<Scored>,
        neighbor: &T,
    );

    fn remove_from_preaggregate<T: AsRef<Scored>>(
        &self,
        agg: &mut HashMap<u64, f64>,
        query_session: &Vec<Scored>,
        neighbor: &T,
    );

    fn generate_from_preaggregate(
        &self,
        query_session: &Vec<Scored>,
        agg: &HashMap<u64, f64>,
    ) -> Vec<Scored>;

    fn predict(&self, query: &Vec<Scored>, neighbors: &Vec<Scored>, how_many: usize)
        -> Vec<Scored>;
}

pub trait Dataset {
    fn collect_keys(&self) -> Vec<u32>;

    fn num_interactions(&self) -> usize;

    fn __get_entry__(&self, key: u32) -> DatasetEntry;
    fn __get_items__(&self, key: u32) -> Vec<u32>;
    //Return the number of keys in this dataset. This could be different to the amount of interactions that are in this dataset.
    fn len(&self) -> usize;
}
#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct DatasetEntry {
    pub key: u32,
    pub sequences: Vec<Sequence>,
    pub max_timestamp: Time,
}
#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct Sequence {
    pub input: Vec<Scored>,
    pub target: Vec<Scored>,
}

pub trait Importance {
    fn compute_importance<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
        &self,
        model: &R,
        metric_factory: &MetricFactory,
        train: &D,
        valid: &D,
    ) -> HashMap<SessionId, f64>;
}

impl Hash for Scored {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        // Hashing f64 requires converting it to bits, since floating-point values
        // are not directly hashable.
        self.score.to_bits().hash(state);
    }
}

pub fn score_sessions<F, D: Dataset + Sync>(heldout_data: &D, consumer: F)
where
    F: Fn(&Vec<Scored>, &Vec<Scored>) + Send + Sync + Clone,
{
    let mut ids: Vec<SessionId> = heldout_data.collect_keys();
    ids.par_sort();

    // Process each session in parallel
    ids.into_par_iter()
        .map(|key| heldout_data.__get_entry__(key))
        .for_each(|dataset_entry: DatasetEntry| {
            for interaction in dataset_entry.sequences {
                let query = interaction.input;
                let actual_next_items = interaction.target;
                consumer(&query, &actual_next_items);
            }
        });
}

pub fn score_contributions_parallel<F, D: Dataset + Sync>(
    heldout_data: &D,
    num_training_keys: usize,
    consumer: F,
) -> Vec<f64>
where
    F: FnMut(&Vec<Scored>, &Vec<Scored>, &mut Vec<f64>) + Send + Sync + Clone,
{
    let contributions = heldout_data
        .collect_keys()
        .par_iter()
        .map(|&key| {
            let mut chunk_consumer = consumer.clone();
            let mut local_contributions: Vec<f64> = vec![0.0; num_training_keys];

            let entry = heldout_data.__get_entry__(key);
            for interaction in entry.sequences {
                let query = interaction.input;
                let actual_next_items = interaction.target;

                chunk_consumer(&query, &actual_next_items, &mut local_contributions);
            }
            local_contributions
        })
        .reduce_with(|left, right| {
            left.into_iter()
                .zip(right)
                .map(|(x, y)| x + y)
                .collect()
        })
        .unwrap_or_else(|| vec![0.0; num_training_keys]);

    contributions
}

pub fn evaluate_dataset<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
    model: &R,
    metric_factory: &MetricFactory,
    test_data: &D,
) -> Vec<(String, f64)> {
    // Metrics need to be shared among threads, using Mutex to ensure thread safety
    let metrics: Vec<Mutex<Box<dyn Metric + Send + Sync>>> = metric_factory
        .create_evaluation_metrics()
        .into_iter()
        .map(Mutex::new)
        .collect();

    score_sessions(test_data, |query_session, actual_next_items| {
        let neighbors = model.retrieve_k(query_session);
        let recommended_items = model.predict(query_session, &neighbors, 21);

        metrics.iter().for_each(|metric| {
            let mut metric = metric.lock().unwrap();
            metric.add(&recommended_items, actual_next_items);
        });
    });

    // Collect results from metrics
    let results: Vec<(String, f64)> = metrics
        .into_iter()
        .map(|metric| {
            let metric = metric.into_inner().unwrap();
            (metric.get_name(), metric.result())
        })
        .collect();

    results
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        t.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_hash_equality_for_identical_structs() {
        let score1 = Scored {
            id: 42,
            score: 3.14,
        };
        let score2 = Scored {
            id: 42,
            score: 3.14,
        };

        // Calculate the hashes of both structs
        let hash1 = calculate_hash(&score1);
        let hash2 = calculate_hash(&score2);

        // They should have the same hash
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_difference_for_different_structs() {
        let score1 = Scored {
            id: 42,
            score: 3.14,
        };
        let score2 = Scored {
            id: 43,
            score: 2.71,
        };

        // Calculate the hashes of both structs
        let hash1 = calculate_hash(&score1);
        let hash2 = calculate_hash(&score2);

        // They should have different hashes
        assert_ne!(hash1, hash2);
    }
}
