use crate::sessrec::metrics::MetricFactory;
use crate::sessrec::types::SessionDataset;

pub mod k_mc_shapley;
pub mod k_mc_shapley_aggregate;
pub mod loo;
pub mod mcshapley_sparse;
pub mod tmcshapley;

#[derive(Clone)]
pub struct ExperimentPayload {
    pub train_dataset: SessionDataset,
    pub valid_dataset: SessionDataset,
    pub vmis_k: usize,
    pub vmis_m: usize,
    pub random_score: f64,
    pub metric_factory: MetricFactory<'static>,
    pub monte_carlo_iterations: usize,
    pub seed: usize,
}
