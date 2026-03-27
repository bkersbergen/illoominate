use indicatif::{ProgressBar, ProgressStyle};
use rayon::current_num_threads;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::marker::{Send, Sync};
use std::time::Instant;

use crate::importance::candidate_neighbors::CandidateNeighbors;
use crate::importance::mc_utils::{error_dataset, random_score_dataset};
use crate::importance::{mc_utils, Dataset, Importance, RetrievalBasedModel};

use crate::metrics::MetricFactory;
use crate::sessrec::vmisknn::Scored;

enum ProgressReporter {
    Bar(ProgressBar),
    Log {
        started_at: Instant,
        last_reported_at: Instant,
    },
}

impl ProgressReporter {
    fn new(max_iterations: usize) -> Self {
        if std::io::stderr().is_terminal() {
            let bar = ProgressBar::new(max_iterations as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} | Monte Carlo Iteration: {pos} | Elapsed: {elapsed_precise} | ETA (worst case): {eta_precise}")
                    .unwrap(),
            );
            Self::Bar(bar)
        } else {
            let now = Instant::now();
            Self::Log {
                started_at: now,
                last_reported_at: now,
            }
        }
    }

    fn report(&mut self, iteration: usize, max_iterations: usize, message: &str, force: bool) {
        match self {
            Self::Bar(bar) => {
                bar.set_message(message.to_owned());
                bar.inc(1);
            }
            Self::Log {
                started_at,
                last_reported_at,
            } => {
                let now = Instant::now();
                if force || now.duration_since(*last_reported_at).as_secs_f64() >= 5.0 {
                    log::info!(
                        "KMC-Shapley progress: {iteration}/{max_iterations} iterations | elapsed: {:.1}s | {message}",
                        now.duration_since(*started_at).as_secs_f64()
                    );
                    *last_reported_at = now;
                }
            }
        }
    }

    fn finish(&mut self, actual_iterations: usize, message: &str) {
        match self {
            Self::Bar(bar) => {
                bar.set_length(actual_iterations as u64);
                bar.finish_with_message(message.to_owned());
            }
            Self::Log { started_at, .. } => {
                log::info!(
                    "KMC-Shapley finished after {actual_iterations} iterations in {:.1}s | {message}",
                    started_at.elapsed().as_secs_f64()
                );
            }
        }
    }
}

pub struct KMcShapley {
    error: f64,
    max_shapley_num_iterations: usize,
    convergence_check_every: usize,
    min_shapley_num_iterations: usize,
    seed: usize,
}

impl KMcShapley {
    pub fn new(error: f64, max_shapley_num_iterations: usize, seed: usize) -> Self {
        Self::with_convergence(error, max_shapley_num_iterations, 10, 100, seed)
    }

    pub fn with_convergence(
        error: f64,
        max_shapley_num_iterations: usize,
        convergence_check_every: usize,
        min_shapley_num_iterations: usize,
        seed: usize,
    ) -> Self {
        Self {
            error,
            max_shapley_num_iterations,
            convergence_check_every: convergence_check_every.max(1),
            min_shapley_num_iterations,
            seed,
        }
    }
}

impl Importance for KMcShapley {
    fn compute_importance<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
        &self,
        model: &R,
        metric_factory: &MetricFactory,
        train: &D,
        valid: &D,
    ) -> HashMap<u32, f64> {
        importance_kmc_dataset::<R, D>(
            model,
            metric_factory,
            train,
            valid,
            self.error,
            self.max_shapley_num_iterations,
            self.convergence_check_every,
            self.min_shapley_num_iterations,
            self.seed,
        )
    }
}

fn importance_kmc_dataset<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
    model: &R,
    metric_factory: &MetricFactory,
    train: &D,
    valid: &D,
    convergence_threshold: f64,
    max_shapley_num_iterations: usize,
    convergence_check_every: usize,
    min_shapley_num_iterations: usize,
    seed: usize,
) -> HashMap<u32, f64> {
    let mut mem_tmc: HashMap<u32, Vec<f64>> = HashMap::with_capacity(train.len());
    log::info!("random_score_dataset()");
    log::info!("metric_factory: {:?}", metric_factory);
    let (random_score, _random_stddev_score) = random_score_dataset(model, metric_factory, valid);
    log::info!(
        "random_score: {:.4} _random_stddev_score: {:.4}",
        random_score,
        _random_stddev_score
    );
    let mut qty_actual_iterations = 0;
    log::info!("Running with {} Rayon threads", current_num_threads());

    let mut progress = ProgressReporter::new(max_shapley_num_iterations);
    let mut last_message: String = String::new();
    let mut mc_error = f64::INFINITY;
    while qty_actual_iterations < max_shapley_num_iterations {
        let iter_start = Instant::now();
        qty_actual_iterations += 1;
        let marginal_contribs = one_iteration_dataset(
            model,
            metric_factory,
            train,
            valid,
            random_score,
            seed,
            qty_actual_iterations,
        );

        let performance: f64 = marginal_contribs.par_iter().sum();

        for &key in train.collect_keys().iter() {
            let entry = mem_tmc.entry(key).or_default();
            let marginal_contribution = marginal_contribs[key as usize];
            entry.push(marginal_contribution);
        }
        let duration = iter_start.elapsed().as_secs_f64();
        let current_message: String = format!(
            "Time/iter: {:.1}s | Metric Score: {:.3} | MC error: {:.1} (goal: {:.1})",
            duration, performance, mc_error, convergence_threshold
        );

        last_message.clear();
        last_message.push_str(&current_message.clone());

        let should_force_report = qty_actual_iterations == 1
            || qty_actual_iterations == max_shapley_num_iterations
            || qty_actual_iterations % convergence_check_every == 0;
        progress.report(
            qty_actual_iterations,
            max_shapley_num_iterations,
            &last_message,
            should_force_report,
        );

        if qty_actual_iterations % convergence_check_every == 0
            && qty_actual_iterations >= min_shapley_num_iterations
        {
            mc_error = error_dataset(&mem_tmc, min_shapley_num_iterations);
            log::debug!("mc_error: {:?}", mc_error);
            if mc_error < convergence_threshold {
                break;
            }
        }
    }
    progress.finish(qty_actual_iterations, &last_message);

    // Calculate average for each session id
    let mut kmc: HashMap<u32, f64> = HashMap::with_capacity(mem_tmc.len());

    for (key, importances) in mem_tmc.into_iter() {
        let average_importance = importances.iter().sum::<f64>() / importances.len() as f64;
        kmc.insert(key, average_importance);
    }

    kmc
}

#[allow(non_snake_case)]
pub fn one_iteration_dataset<R: RetrievalBasedModel + Send + Sync, D: Dataset + Sync>(
    model: &R,
    metric_factory: &MetricFactory,
    train: &D,
    valid: &D,
    random_score: f64,
    seed: usize,
    iteration: usize,
) -> Vec<f64> {
    let training_keys_permuted = mc_utils::permutation(train, seed, iteration);
    let qty_training_keys = train.collect_keys().len();
    let mut permutation_index = vec![0; qty_training_keys];
    for (idx, &key) in training_keys_permuted.iter().enumerate() {
        permutation_index[key as usize] = idx;
    }
    let mut contributions = valid
        .collect_keys()
        .into_par_iter()
        .map(|key| {
            let mut local_contributions: Vec<f64> = vec![0.0; qty_training_keys];
            let entry = valid.__get_entry__(key);
            for sequence in entry.sequences {
                let input = &sequence.input;
                let target = sequence.target;
                let mut N_q: Vec<Scored> = model.retrieve_all(input);
                N_q.sort_by_key(|session_score| permutation_index[session_score.id as usize]);

                let mut candidate_neighbors = CandidateNeighbors::new(model.k());
                let mut prev_score = 0.0;

                let metric_binding = metric_factory.create_importance_metric();
                let metric = metric_binding.as_ref();

                let mut agg = model.create_preaggregate();

                for neighbor in N_q {
                    let (topk_updated, dropped_out) = candidate_neighbors.offer(neighbor);
                    if topk_updated {
                        &model.add_to_preaggregate(&mut agg, &input, &neighbor);

                        if dropped_out.is_some() {
                            &model.remove_from_preaggregate(
                                &mut agg,
                                &input,
                                &dropped_out.unwrap(),
                            );
                        }

                        //let neighbors: Vec<_> = candidate_neighbors.iter().collect();

                        let recommended_items = &model.generate_from_preaggregate(input, &agg);

                        let metric_result = metric.compute(recommended_items, &target);

                        let new_score = metric_result - prev_score;
                        prev_score = metric_result;
                        local_contributions[neighbor.id as usize] += new_score
                    }
                }
            }
            local_contributions
        })
        .reduce_with(|left, right| left.into_iter().zip(right).map(|(x, y)| x + y).collect())
        .unwrap_or_else(|| vec![0.0; qty_training_keys]);

    let qty_evaluations = valid.num_interactions();

    // Parallelize the normalization of contributions
    contributions.par_iter_mut().for_each(|contribution| {
        *contribution /= qty_evaluations as f64;
    });

    // subtract the random score from the first session in the permuted list
    contributions[training_keys_permuted[0] as usize] -= random_score;
    log::debug!("----------------------------------------------------------------Illoominate K-MC-Shapley------------------------------------------------------------------------------------");
    contributions
}
