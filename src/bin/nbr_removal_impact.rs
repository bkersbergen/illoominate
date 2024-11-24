// use chrono::Local;
// use env_logger::Builder;
// use illoominate::importance::candidate_neighbors::CandidateNeighbors;
// use illoominate::importance::k_mc_shapley::KMcShapley;
// use illoominate::importance::tmc_utils::{error_dataset, mean_stddev};
// use illoominate::importance::{score_contributions_parallel, tmc_utils, RetrievalBasedModel, Dataset};
// use illoominate::nbr::caboose::sparse_topk_index::SparseTopKIndex;
// use illoominate::nbr::caboose::types::Score;
// use illoominate::nbr::caboose::types::SimilarRow;
// use illoominate::nbr::tifuknn::hyperparams::{
//     PARAMS_BOL, PARAMS_DUNNHUMBY, PARAMS_INSTACART, PARAMS_TAFANG, PARAMS_VALUEDSHOPPER,
// };
// use illoominate::nbr::tifuknn::io::read_baskets_file;
// use illoominate::nbr::tifuknn::types::{Basket, HyperParams, SparseItemVector, UserId};
// use illoominate::nbr::tifuknn::{TIFUIndex, TIFUKNN};
// use illoominate::sessrec::io::read_sustainable_products_info;
// use illoominate::sessrec::metrics::hitrate::HitRate;
// use illoominate::sessrec::metrics::mrr::Mrr;
// use illoominate::sessrec::metrics::product_info::ProductInfo;
// use illoominate::sessrec::metrics::recall::Recall;
// use illoominate::sessrec::metrics::{Metric, MetricConfig, MetricFactory, MetricType};
// use illoominate::sessrec::types::{ItemId, SessionDataset, SessionId};
// use illoominate::sessrec::vmisknn::{Scored, VMISKNN};
// use indicatif::{ProgressBar, ProgressStyle};
// use itertools::{enumerate, Itertools};
// use log::LevelFilter;
// use rand::prelude::{IteratorRandom, SliceRandom, StdRng};
// use rand::{thread_rng, SeedableRng};
// use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
// use rayon::prelude::{
//     IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelSliceMut,
// };
// use sprs::SpIndex;
// use std::cmp::{max, min};
// use std::collections::{BinaryHeap, HashMap, HashSet};
// use std::fmt::format;
// use std::fs::{File, OpenOptions};
// use std::io::Write;
// use std::sync::Mutex;
// use std::{env, fs};
// use illoominate::nbr::removal_impact::split_train_eval;
// use illoominate::nbr::types::NextBasketDataset;
//
// fn main() {
//     // let qty_logical_cores = num_cpus::get();
//     let qty_physical_cores = num_cpus::get_physical();
//     rayon::ThreadPoolBuilder::new()
//         .num_threads(max(1, qty_physical_cores - 1))
//         .build_global()
//         .unwrap();
//     init_logger();
//     // let data_path = "data/nbr_eu2m_with_index/";
//     // let data_path = "experiments/datasets/raw/processed/tafang";
//     let data_path = "experiments/datasets/raw/processed/dunnhumby";
//     // let data_path = "experiments/datasets/raw/processed/valuedshopper";
//     // let data_path = "experiments/datasets/raw/processed/tafang";
//
//     let hp = PARAMS_DUNNHUMBY;
//
//     if hp.alpha > 0.4 {
//         log::warn!("With high alpha values the user representation dominates the outcome of the prediction. This makes it unlikely that you can improve prediction by removing other users from the training data.");
//     }
//
//     let config = MetricConfig {
//         importance_metric: MetricType::Ndcg,
//         evaluation_metrics: vec![MetricType::Ndcg],
//         length: 21,
//         mrr_alpha: 0.8,
//     };
//     run_importance_experiment_for(&data_path, &config, &hp);
//
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::MRR,
//     //     evaluation_metrics: vec![MetricType::MRR],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     // run_importance_experiment_for(&data_path, &config, &hp);
//     //
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::Recall,
//     //     evaluation_metrics: vec![MetricType::Recall],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     // run_importance_experiment_for(&data_path, &config, &hp);
//     //
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::HitRate,
//     //     evaluation_metrics: vec![MetricType::HitRate],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     // run_importance_experiment_for(&data_path, &config, &hp);
//
//     // let config = MetricConfig {
//     //     importance_metric: MetricType::Ndcg,
//     //     evaluation_metrics: vec![MetricType::Ndcg, MetricType::MRR, MetricType::Recall],
//     //     length: 21,
//     //     mrr_alpha: 0.8,
//     // };
//     //
//     // run_importance_experiment_for(&data_path, &config);
// }
//
// fn run_importance_experiment_for(
//     data_path: &&str,
//     config: &MetricConfig,
//     hyper_params: &HyperParams,
// ) {
//     // Check if train.csv and test.csv exist
//     let baskets_filename = "baskets.csv";
//     let validation_ratio = 0.5;
//
//     let basket_csv_path = format!("{}/{}", data_path, baskets_filename);
//     let all_baskets_by_user: NextBasketDataset = read_baskets_file(&basket_csv_path);
//     let (training_baskets, evaluation_baskets, test_baskets) =
//         split_train_eval(all_baskets_by_user, validation_ratio);
//
//     let output_path = format!(
//         "{}/test_{}_valid_{}",
//         data_path,
//         test_baskets.user_baskets.len(),
//         evaluation_baskets.user_baskets.len(),
//     );
//     fs::create_dir_all(&output_path).unwrap();
//     log::info!(
//         "Evaluating with testset number of users: {} and validation number of users: {}",
//         test_baskets.user_baskets.len(),
//         evaluation_baskets.user_baskets.len()
//     );
//     single_run(
//         &output_path,
//         &config,
//         &hyper_params,
//         &training_baskets,
//         &test_baskets,
//         &evaluation_baskets,
//     );
//
//     let do_test_size_reduction_experiment = false;
//     if do_test_size_reduction_experiment {
//         // Experiment setup:
//         // * fix training data
//         // * fix test data to 20% of holdout dataset
//         // * And only increase the validation set size and see it the results of testset improve
//
//         let mut evaluation_baskets = evaluation_baskets.clone();
//
//         let qty_eval = evaluation_baskets.len();
//         let qty_steps = 5;
//         let mut rng = StdRng::seed_from_u64(42);
//
//         let mut keys_to_remove: Vec<_> = evaluation_baskets.collect_keys();
//         keys_to_remove.sort();
//         keys_to_remove.shuffle(&mut rng);
//
//         for (iteration, i) in (0..=qty_eval)
//             .rev()
//             .step_by(qty_eval / qty_steps)
//             .enumerate()
//         {
//             for key in &keys_to_remove {
//                 if evaluation_baskets.len() > i {
//                     evaluation_baskets.user_baskets.remove(key);
//                 }
//             }
//             if iteration == qty_steps {
//                 break;
//             }
//
//             let output_path = format!(
//                 "{}/test_{}_valid_{}",
//                 data_path,
//                 test_baskets.len(),
//                 evaluation_baskets.len()
//             );
//             fs::create_dir_all(&output_path).unwrap();
//             log::info!(
//                 "Evaluating with testset number of users: {} and validation number of users: {}",
//                 test_baskets.len(),
//                 evaluation_baskets.len()
//             );
//             single_run(
//                 &output_path,
//                 &config,
//                 &hyper_params,
//                 &training_baskets,
//                 &test_baskets,
//                 &evaluation_baskets,
//             );
//         }
//     }
// }
//
// fn single_run(
//     output_path: &str,
//     config: &MetricConfig,
//     hyper_params: &HyperParams,
//     training_baskets: &NextBasketDataset,
//     test_baskets: &NextBasketDataset,
//     evaluation_baskets: &NextBasketDataset,
// ) {
//     let metric_factory = MetricFactory::new(&config, ProductInfo::new(HashSet::new()));
//
//     let importance_metric_binding = metric_factory.create_importance_metric();
//
//     let importance_metric_friendly = importance_metric_binding.as_ref().get_name().to_lowercase();
//
//     let mut output_file_shapley_importances = create_file(&format!(
//         "{}/__removal_impact_shapley_importance_{}_eval_{}.csv",
//         output_path, importance_metric_friendly, importance_metric_friendly
//     ));
//     let mut output_file_loo_importances = create_file(&format!(
//         "{}/__removal_impact_loo_importance_{}_eval_{}.csv",
//         output_path, importance_metric_friendly, importance_metric_friendly
//     ));
//
//     let mut output_files_evaluation_metric_results: Vec<File> = metric_factory
//         .create_evaluation_metrics()
//         .iter_mut()
//         .map(|metric| {
//             let metric_friendly_name = metric.get_name().to_lowercase();
//             create_file(&format!(
//                 "{}/__removal_impact_results_importance_{}_eval_{}.csv",
//                 output_path, importance_metric_friendly, metric_friendly_name
//             ))
//         })
//         .collect();
//
//     let qty_impact_resolution = 50;
//
//     let model = TIFUKNN::new(&training_baskets, &hyper_params);
//     let training_data_user_embeddings = model.get_all_user_embeddings();
//
//     // let seeds = [1313, 18102022, 12345];
//     let seeds = [1313];
//     for seed in seeds {
//         log::info!("determine removal impact with seed: {}", seed);
//
//         let loo_values = compute_importance_loo_efficient(
//             &training_data_user_embeddings,
//             &metric_factory,
//             &training_baskets,
//             &evaluation_baskets,
//             &hyper_params,
//         );
//
//         let mut loo_indices_sorted_by_importance: Vec<(u32, f64)> =
//             loo_values.iter().map(|(&k, &v)| (k as u32, v)).collect();
//         loo_indices_sorted_by_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//         for (index, importance) in &loo_indices_sorted_by_importance {
//             let data_to_append = format!("{},{},{:.10}", seed, index, importance);
//             writeln!(output_file_loo_importances, "{}", data_to_append)
//                 .expect("Failed to write to file");
//         }
//
//         let important_first_loo = positive_by_importance(&loo_values);
//         evaluate_removal_impact(
//             "important_first_loo",
//             &metric_factory,
//             &training_data_user_embeddings,
//             &training_baskets,
//             &evaluation_baskets,
//             &test_baskets,
//             &hyper_params,
//             &important_first_loo,
//             seed,
//             qty_impact_resolution,
//             &mut output_files_evaluation_metric_results,
//         );
//
//         let least_first_loo = negative_by_importance_reverse(&loo_values);
//         evaluate_removal_impact(
//             "least_first_loo",
//             &metric_factory,
//             &training_data_user_embeddings,
//             &training_baskets,
//             &evaluation_baskets,
//             &test_baskets,
//             &hyper_params,
//             &least_first_loo,
//             seed,
//             qty_impact_resolution,
//             &mut output_files_evaluation_metric_results,
//         );
//
//         let shapley_values = compute_importance_mcshapley_effective(
//             &training_data_user_embeddings,
//             &metric_factory,
//             &training_baskets,
//             &evaluation_baskets,
//             &hyper_params,
//         );
//         let mut indices_sorted_by_importance: Vec<_> =
//             shapley_values.clone().into_iter().collect_vec();
//         indices_sorted_by_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//
//         for (index, importance) in &indices_sorted_by_importance {
//             let data_to_append = format!("{},{},{:.10}", seed, index, importance);
//             writeln!(output_file_shapley_importances, "{}", data_to_append)
//                 .expect("Failed to write to file");
//         }
//
//         let important_first = positive_by_importance(&shapley_values);
//         evaluate_removal_impact(
//             "important_first_shapley",
//             &metric_factory,
//             &training_data_user_embeddings,
//             &training_baskets,
//             &evaluation_baskets,
//             &test_baskets,
//             &hyper_params,
//             &important_first,
//             seed,
//             qty_impact_resolution,
//             &mut output_files_evaluation_metric_results,
//         );
//
//         let least_first = negative_by_importance_reverse(&shapley_values);
//         evaluate_removal_impact(
//             "least_first_shapley",
//             &metric_factory,
//             &training_data_user_embeddings,
//             &training_baskets,
//             &evaluation_baskets,
//             &test_baskets,
//             &hyper_params,
//             &least_first,
//             seed,
//             qty_impact_resolution,
//             &mut output_files_evaluation_metric_results,
//         );
//
//         let num_random_sessions_to_remove = *[
//             important_first.len(),
//             least_first.len(),
//             important_first_loo.len(),
//             least_first_loo.len(),
//         ]
//         .iter()
//         .max()
//         .unwrap();
//
//         let mut rng = StdRng::seed_from_u64((seed) as u64);
//         let mut random_sessions_to_remove: Vec<_> = loo_values
//             .clone()
//             .into_iter()
//             .choose_multiple(&mut rng, num_random_sessions_to_remove)
//             .into_iter()
//             .map(|(index, _)| index)
//             .collect();
//
//         random_sessions_to_remove.shuffle(&mut rng);
//
//         evaluate_removal_impact(
//             "random",
//             &metric_factory,
//             &training_data_user_embeddings,
//             &training_baskets,
//             &evaluation_baskets,
//             &test_baskets,
//             &hyper_params,
//             &random_sessions_to_remove,
//             seed,
//             qty_impact_resolution,
//             &mut output_files_evaluation_metric_results,
//         );
//     }
// }
//
// fn load_data(
//     data_path: &str,
// ) -> (
//     HashMap<UserId, Vec<Basket>>,
//     HashMap<UserId, Basket>,
//     HashMap<UserId, Basket>,
// ) {
//     // Check if train.csv and test.csv exist
//     let baskets_filename = "baskets.csv";
//     let validation_ratio = 0.8;
//
//     let basket_csv_path = format!("{}/{}", data_path, baskets_filename);
//     let all_baskets_by_user: HashMap<UserId, Vec<Basket>> = read_baskets_file(&basket_csv_path);
//     let (training_baskets, evaluation_baskets, test_baskets) =
//         split_train_eval(all_baskets_by_user, validation_ratio);
//     return (training_baskets, evaluation_baskets, test_baskets);
// }
//
// fn compute_importance_mcshapley_effective(
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     metrics_factory: &MetricFactory,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
//     hp: &HyperParams,
// ) -> HashMap<UserId, f64> {
//     let train_baskets_by_user = training.clone();
//
//     let mut mem_tmc: HashMap<UserId, Vec<f64>> =
//         HashMap::with_capacity(train_baskets_by_user.len());
//     let model = TIFUKNN::new(&train_baskets_by_user, hp);
//     log::info!("random_score_dataset()");
//     let (random_score, _random_stddev_score) = random_score_dataset(
//         &model,
//         &training_data_user_embeddings,
//         metrics_factory,
//         &validation,
//     );
//     log::info!(
//         "random_score: {:.4} _random_stddev_score: {:.4}",
//         random_score,
//         _random_stddev_score
//     );
//
//     let mut tmc_run = true;
//     let iterations = 100;
//     let max_iterations = 200; // Add max iteration limit
//     println!("Using KMC Iterations: {}", iterations);
//
//     let pb = ProgressBar::new(iterations as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("KMCShapley {percent} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap(),
//     );
//     let err = 0.1;
//     let mut qty_actual_iterations = 0;
//
//     let model = TIFUKNN::new(&training, &hp);
//     while tmc_run {
//         let mem_tmc_session: HashMap<SessionId, Vec<f64>> = mem_tmc
//             .par_iter()
//             .map(|(user_id, vec)| (*user_id as SessionId, vec.clone()))
//             .collect();
//         let tmc_error = error_dataset(&mem_tmc_session, iterations);
//         log::debug!("tmc_error: {:?}", tmc_error);
//         if tmc_error < err || qty_actual_iterations >= max_iterations {
//             tmc_run = false
//         } else {
//             for iteration in 0..iterations {
//                 qty_actual_iterations += 1;
//                 pb.inc(1);
//                 log::info!("{} KMC_Shapley iterations.", qty_actual_iterations,);
//                 let marginal_contribs = one_iteration_shapley_effective(
//                     &model,
//                     training_data_user_embeddings,
//                     training,
//                     validation,
//                     metrics_factory,
//                     iteration,
//                     random_score,
//                     hp,
//                 );
//                 for &user_id in training.keys() {
//                     let entry = mem_tmc.entry(user_id).or_insert_with(|| Vec::new());
//                     let marginal_contribution = marginal_contribs[user_id as usize];
//                     entry.push(marginal_contribution);
//                 }
//             }
//         }
//     }
//     pb.finish();
//     log::info!("iterations used: {}", qty_actual_iterations);
//
//     // Calculate average importance for each session id
//     let vals_tmc: HashMap<UserId, f64> = mem_tmc
//         .drain()
//         .map(|(session_idx, tmc_importances)| {
//             let average_session_importance =
//                 tmc_importances.iter().sum::<f64>() / tmc_importances.len() as f64;
//             (session_idx, average_session_importance)
//         })
//         .collect();
//
//     log::info!("Efficient Monte Carlo Shapley values calculated!");
//     vals_tmc
// }
//
// fn compute_importance_mcshapley_naive(
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     metrics_factory: &MetricFactory,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
//     hp: &HyperParams,
// ) -> HashMap<UserId, f64> {
//     let train_baskets_by_user = training.clone();
//     let keys: Vec<_> = train_baskets_by_user.keys().cloned().collect();
//
//     let mut mem_tmc: HashMap<UserId, Vec<f64>> = HashMap::with_capacity(keys.len());
//     let model = TIFUKNN::new(&train_baskets_by_user, hp);
//     log::info!("random_score_dataset()");
//     let (random_score, _random_stddev_score) = random_score_dataset(
//         &model,
//         &training_data_user_embeddings,
//         metrics_factory,
//         &validation,
//     );
//     log::info!(
//         "random_score: {:.4} _random_stddev_score: {:.4}",
//         random_score,
//         _random_stddev_score
//     );
//
//     let mut tmc_run = true;
//     let iterations = 100;
//     let pb = ProgressBar::new(iterations as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("MCShapley {wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap(),
//     );
//     let err = 0.1;
//     let mut qty_actual_iterations = 0;
//     while tmc_run {
//         let mem_tmc_session: HashMap<SessionId, Vec<f64>> = mem_tmc
//             .par_iter()
//             .map(|(user_id, vec)| (*user_id as SessionId, vec.clone()))
//             .collect();
//         let tmc_error = error_dataset(&mem_tmc_session, iterations);
//         log::debug!("tmc_error: {:?}", tmc_error);
//         if tmc_error < err {
//             tmc_run = false
//         } else {
//             for iteration in 0..iterations {
//                 qty_actual_iterations += 1;
//                 pb.inc(1);
//                 log::debug!(
//                     "{} out of {} TMC_Shapley iterations.",
//                     iteration + 1,
//                     iterations
//                 );
//                 let marginal_contribs = one_iteration_shapley_naive(
//                     &model,
//                     training_data_user_embeddings,
//                     training,
//                     validation,
//                     metrics_factory,
//                     iteration,
//                     random_score,
//                     hp,
//                 );
//                 for &user_id in training.keys() {
//                     let entry = mem_tmc.entry(user_id).or_insert_with(|| Vec::new());
//                     let marginal_contribution = marginal_contribs[user_id as usize];
//                     entry.push(marginal_contribution);
//                 }
//             }
//         }
//     }
//     pb.finish();
//     log::info!("iterations used: {}", qty_actual_iterations);
//
//     // Calculate average importance for each session id
//     let vals_tmc: HashMap<UserId, f64> = mem_tmc
//         .drain()
//         .map(|(session_idx, tmc_importances)| {
//             let average_session_importance =
//                 tmc_importances.iter().sum::<f64>() / tmc_importances.len() as f64;
//             (session_idx, average_session_importance)
//         })
//         .collect();
//
//     log::info!("Truncated Monte Carlo Shapley values calculated!");
//     vals_tmc
// }
//
// pub fn permutation(
//     training: &HashMap<UserId, Vec<Basket>>,
//     seed: usize,
//     iteration: usize,
// ) -> Vec<UserId> {
//     let mut ids: Vec<_> = training.keys().cloned().collect();
//     let mut rng = StdRng::seed_from_u64((seed + iteration) as u64);
//
//     ids.shuffle(&mut rng);
//     ids
// }
//
// fn one_iteration_shapley_effective(
//     model: &TIFUKNN,
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
//     metrics_factory: &MetricFactory,
//     iteration: usize,
//     random_score: f64,
//     hp: &HyperParams,
// ) -> Vec<f64> {
//     log::debug!("----------------------------------------------------------------START K-MC-Shapley------------------------------------------------------------------------------------");
//
//     let permutation: Vec<UserId> = permutation(training, 42, iteration);
//
//     let max_id = (*permutation.iter().max().unwrap_or(&0) + 1) as usize;
//     let mut permutation_index = vec![0; max_id];
//     for (idx, &value) in permutation.iter().enumerate() {
//         permutation_index[value as usize] = idx;
//     }
//     let k = hp.k;
//     let mut hp = hp.clone();
//     hp.k = training.len(); // we want to retrieve all neighbors
//     log::debug!("start Training TIFU model");
//     log::debug!("done Training TIFU model");
//     let mut ids: Vec<_> = validation.keys().cloned().collect();
//     ids.par_sort();
//
//     // let degree_of_parallelism = num_cpus::get();
//
//     log::debug!("start estimate contributions");
//     let mut contributions = ids
//         .into_par_iter()
//         // .with_min_len(degree_of_parallelism)
//         .map(|user_id| {
//             let mut local_contributions: Vec<f64> = vec![0.0; max_id];
//             let basket = validation.get(&user_id).unwrap();
//             let actual_items: Vec<_> = basket
//                 .items
//                 .iter()
//                 .map(|&item| Scored::new(item as u32, 1.0))
//                 .collect();
//
//             let mut N_q: Vec<_> = model.find_neighbors(&user_id); // 3k per user
//
//             // let mut N_q = top_n_similar_rows(N_q, 500);
//             N_q.sort_by_key(|id_score| permutation_index[id_score.row as usize]);
//
//             let mut candidate_neighbors = CandidateNeighbors::new(k);
//             let mut prev_score = 0.0;
//
//             let metric_binding = metrics_factory.create_importance_metric();
//             let metric = metric_binding.as_ref();
//             for similar_row in N_q {
//                 let similar_session = Scored::new(similar_row.row, similar_row.similarity as f64);
//                 let (topk_updated, _) = candidate_neighbors.offer(similar_session);
//                 if topk_updated {
//                     let neighbors: Vec<SimilarRow> = candidate_neighbors
//                         .iter()
//                         .map(|s| SimilarRow::new(s.id, s.score as Score))
//                         .collect();
//                     let recommended_items =
//                         model.predict_for(&user_id, &neighbors, metrics_factory.config.length);
//
//                     let metric_result = metric.compute(&recommended_items, &actual_items);
//
//                     let new_score = metric_result - prev_score;
//                     prev_score = metric_result;
//                     local_contributions[similar_session.id as usize] += new_score
//                 }
//             }
//
//             local_contributions
//         })
//         .reduce_with(|left, right| left.iter().zip(right.iter()).map(|(x, y)| x + y).collect())
//         .unwrap_or_else(|| vec![]);
//     log::debug!("end estimate contributions");
//
//     let qty_evaluations = validation.len();
//
//     // Parallelize the normalization of contributions
//     contributions.par_iter_mut().for_each(|contribution| {
//         *contribution /= qty_evaluations as f64;
//     });
//
//     // subtract the random score from the first session in the permuted list
//     contributions[permutation[0] as usize] -= random_score;
//     log::debug!("----------------------------------------------------------------DONE K-MC-Shapley------------------------------------------------------------------------------------");
//     contributions
// }
//
// fn top_n_similar_rows(mut a: Vec<SimilarRow>, n: usize) -> Vec<SimilarRow> {
//     if a.len() <= n {
//         a
//     } else {
//         let mut heap = BinaryHeap::from(a.drain(..n).collect::<Vec<_>>());
//         for item in a {
//             if item < *heap.peek().unwrap() {
//                 heap.pop();
//                 heap.push(item);
//             }
//         }
//         heap.into_vec()
//     }
// }
//
// fn one_iteration_shapley_naive(
//     _model: &TIFUKNN,
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
//     metrics_factory: &MetricFactory,
//     iteration: usize,
//     random_score: f64,
//     hp: &HyperParams,
// ) -> Vec<f64> {
//     let mut X_batch = HashMap::new();
//     let permutation: Vec<UserId> = permutation(training, 42, iteration);
//
//     let max_id = *permutation.iter().max().unwrap_or(&0);
//     // because of truncation stoppage we assign a default importance of 0.0
//     let mut contributions: Vec<f64> = vec![0.0; (max_id + 1) as usize];
//
//     let new_score = random_score;
//     let mut old_score = new_score;
//
//     for user_id in &permutation {
//         let payload = training.get(user_id).unwrap().clone();
//
//         X_batch.insert(*user_id, payload);
//
//         let model = TIFUKNN::new(&X_batch, hp);
//         let (_metric_name, metric_value) =
//             evaluate_dataset_importance_metric(&model, metrics_factory, validation);
//         let new_score = &metric_value;
//         // log::debug!("_metric_name: {} new_score:{:.4}", _metric_name, new_score);
//         let u_q = new_score - old_score;
//         old_score = *new_score;
//         contributions[*user_id as usize] = u_q;
//     }
//     log::debug!("----------------------------------------------------------------nested_loops------------------------------------------------------------------------------------");
//     contributions
// }
//
// fn compute_importance_loo_efficient(
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     metrics_factory: &MetricFactory,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
//     hp: &HyperParams,
// ) -> HashMap<UserId, f64> {
//     log::info!("k: {}", hp.k);
//
//     let model = TIFUKNN::new(&training, &hp);
//     let (metric_name, baseline_value) =
//         evaluate_dataset_importance_metric(&model, &metrics_factory, &validation);
//     log::info!("_metric_name: {:?}", metric_name);
//     log::info!("baseline_value: {:?}", baseline_value);
//
//     let mut ids: Vec<UserId> = validation.keys().cloned().collect();
//     ids.par_sort();
//
//     let metric_binding = metrics_factory.create_importance_metric();
//     let metric = metric_binding.as_ref();
//
//     let num_recommendations = metrics_factory.config.length;
//     let num_training_users = (*training.keys().max().unwrap() + 1) as usize;
//
//     let contributions: Vec<f64> = ids
//         .into_par_iter()
//         .map(|user_id| {
//             let neighbors = model.find_neighbors(&user_id);
//
//             let kp = min(hp.k, neighbors.len());
//
//             let loo_neighbors = &neighbors[0..kp].to_vec();
//             let recommended_items =
//                 model.predict_for(&user_id, &loo_neighbors, num_recommendations);
//
//             let basket = validation.get(&user_id).unwrap();
//             let actual_items: Vec<_> = basket
//                 .items
//                 .iter()
//                 .map(|&item| Scored::new(item as u32, 1.0))
//                 .collect();
//
//             let original_utility = metric.compute(&recommended_items, &actual_items);
//             let enough_neighbors = neighbors.len() >= hp.k + 1;
//
//             let mut local_contributions = vec![0.0; num_training_users];
//
//             for leave_out_index in 0..kp {
//                 let loo_neighbors = if enough_neighbors {
//                     [
//                         &neighbors[0..leave_out_index],
//                         &neighbors[(leave_out_index + 1)..(kp + 1)],
//                     ]
//                     .concat()
//                 } else {
//                     [
//                         &neighbors[0..leave_out_index],
//                         &neighbors[(leave_out_index + 1)..kp],
//                     ]
//                     .concat()
//                 };
//
//                 let removed_user_id = neighbors[leave_out_index].row as usize;
//                 let recommended_items =
//                     model.predict_for(&user_id, &loo_neighbors, num_recommendations);
//
//                 let score = metric.compute(&recommended_items, &actual_items);
//                 let importance = original_utility - score;
//                 local_contributions[removed_user_id] += importance;
//             }
//
//             local_contributions
//         })
//         .reduce_with(|left, right| left.iter().zip(right.iter()).map(|(x, y)| x + y).collect())
//         .unwrap_or_else(|| vec![]);
//     let mut result = HashMap::with_capacity(contributions.len());
//     let qty_evaluations = validation.len();
//     for &user_id in training.keys() {
//         let marginal_contribution = contributions[user_id as usize] / qty_evaluations as f64;
//         result
//             .entry(user_id)
//             .and_modify(|value| *value += marginal_contribution)
//             .or_insert(marginal_contribution);
//     }
//     result
// }
//
// fn compute_importance_loo_naive(
//     training_data_user_embeddings: &HashMap<UserId, SparseItemVector>,
//     metrics_factory: &MetricFactory,
//     hp: HyperParams,
//     training: &HashMap<UserId, Vec<Basket>>,
//     validation: &HashMap<UserId, Basket>,
// ) -> HashMap<UserId, f64> {
//     let mut train_baskets_by_user = training.clone();
//     let keys: Vec<_> = train_baskets_by_user.keys().cloned().collect();
//
//     let model = TIFUKNN::new(&train_baskets_by_user, &hp);
//     let (metric_name, baseline_value) =
//         evaluate_dataset_importance_metric(&model, &metrics_factory, &validation);
//     log::debug!("_metric_name: {:?}", metric_name);
//     log::debug!("baseline_value: {:?}", baseline_value);
//
//     let mut values_leave_one_out = HashMap::with_capacity(train_baskets_by_user.len());
//
//     let pb = ProgressBar::new(keys.len() as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap(),
//     );
//     for user_id in keys.into_iter() {
//         let removed = train_baskets_by_user.remove(&user_id).unwrap();
//
//         let model = TIFUKNN::new(&train_baskets_by_user, &hp);
//
//         let (_metric_name, metric_value) =
//             evaluate_dataset_importance_metric(&model, &metrics_factory, &validation);
//
//         let contribution = baseline_value - metric_value;
//         log::debug!("without {} : {:.4}", user_id, contribution);
//         values_leave_one_out.insert(user_id, contribution);
//
//         train_baskets_by_user.insert(user_id, removed);
//         pb.inc(1);
//     }
//     pb.finish();
//     values_leave_one_out
// }
//
// fn init_logger() {
//     Builder::new()
//         .filter_level(LevelFilter::Debug)
//         .format(|buf, record| {
//             writeln!(
//                 buf,
//                 "[{} {}] {} {}",
//                 Local::now().format("%Y-%m-%dT%H:%M:%S"),
//                 record.module_path().unwrap_or_else(|| "-".into()),
//                 record.level(),
//                 record.args()
//             )
//         })
//         .init();
// }
//
// fn evaluate_removal_impact(
//     experiment_type: &str,
//     metric_factory: &MetricFactory,
//     user_embeddings_training_data: &HashMap<UserId, SparseItemVector>,
//     training_baskets: &HashMap<UserId, Vec<Basket>>,
//     valid: &HashMap<UserId, Basket>,
//     test: &HashMap<UserId, Basket>,
//     hp: &HyperParams,
//     users_to_remove: &[UserId],
//     seed: usize,
//     qty_impact_resolution: usize,
//     output_files_evaluation_metric_results: &mut Vec<File>,
// ) {
//     let mut training_baskets = training_baskets.clone();
//
//     assert!(training_baskets.len() >= users_to_remove.len());
//
//     let model = TIFUKNN::new(&training_baskets, hp);
//
//     let validation_evaluation_metrics: Vec<(String, f64)> = evaluate_dataset(
//         &model,
//         user_embeddings_training_data,
//         metric_factory,
//         &valid,
//     );
//     let test_evaluation_metrics: Vec<(String, f64)> =
//         evaluate_dataset(&model, user_embeddings_training_data, metric_factory, &test);
//
//     for (
//         (output_file, (_valid_metric_name, valid_metric_score)),
//         (_test_metric_name, test_metric_score),
//     ) in output_files_evaluation_metric_results
//         .iter_mut()
//         .zip(validation_evaluation_metrics.iter())
//         .zip(test_evaluation_metrics.iter())
//     {
//         let data_to_append = format!(
//             "{},{},{:.4},{:.4},{}",
//             experiment_type, seed, valid_metric_score, test_metric_score, 0
//         );
//         log::info!("{}", data_to_append);
//         writeln!(output_file, "{}", data_to_append).expect("Results Failed to write to file");
//     }
//
//     let mut num_users_removed = 0;
//
//     let impact_interval = max(
//         1,
//         (users_to_remove.len() as f64 / qty_impact_resolution as f64) as usize,
//     );
//     log::debug!("start removing users in evaluate_removal_impact");
//     for user_id in users_to_remove {
//         // Remove SessionId from the Training data
//         if let Some(_) = training_baskets.remove(&(*user_id)) {
//             num_users_removed += 1;
//             let compute_impact =
//                 (training_baskets.len() == 1) || (num_users_removed % impact_interval == 0);
//             if compute_impact {
//                 // evaluate on validation data and write output
//                 let model = TIFUKNN::new(&training_baskets, hp);
//                 let validation_evaluation_metrics: Vec<(String, f64)> =
//                     evaluate_dataset(&model, user_embeddings_training_data, metric_factory, valid);
//                 let test_evaluation_metrics: Vec<(String, f64)> =
//                     evaluate_dataset(&model, user_embeddings_training_data, metric_factory, test);
//
//                 for (
//                     (output_file, (_valid_metric_name, valid_metric_score)),
//                     (_test_metric_name, test_metric_score),
//                 ) in output_files_evaluation_metric_results
//                     .iter_mut()
//                     .zip(validation_evaluation_metrics.iter())
//                     .zip(test_evaluation_metrics.iter())
//                 {
//                     let data_to_append = format!(
//                         "{},{},{:.4},{:.4},{}",
//                         experiment_type,
//                         seed,
//                         valid_metric_score,
//                         test_metric_score,
//                         num_users_removed
//                     );
//                     log::info!("{}", data_to_append);
//                     writeln!(output_file, "{}", data_to_append)
//                         .expect("Results Failed to write to file");
//                 }
//             }
//         };
//     }
//     log::debug!("end removing users in evaluate_removal_impact");
// }
//
// fn positive_by_importance(original_importances: &HashMap<UserId, f64>) -> Vec<UserId> {
//     let mut importances: Vec<(UserId, f64)> = original_importances.clone().into_iter().collect();
//     importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//     importances
//         .into_iter()
//         .filter(|(_, importance)| *importance > 0.0)
//         .map(|(session_id, _)| session_id)
//         .collect()
// }
//
// fn negative_by_importance_reverse(original_importances: &HashMap<UserId, f64>) -> Vec<UserId> {
//     let mut importances: Vec<(UserId, f64)> = original_importances.clone().into_iter().collect();
//     importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//     importances
//         .into_iter()
//         .filter(|(_, importance)| *importance < 0.0)
//         .map(|(session_id, _)| session_id)
//         .collect()
// }
//
// fn evaluate_dataset_importance_metric(
//     model: &TIFUKNN,
//     metric_factory: &MetricFactory,
//     valid: &HashMap<UserId, Basket>,
// ) -> (String, f64) {
//     log::info!("start evaluate_dataset_importance_metric");
//     // Metrics need to be shared among threads, using Mutex to ensure thread safety
//     let metric_mutex: Mutex<Box<dyn Metric + Send + Sync>> =
//         Mutex::new(metric_factory.create_importance_metric());
//
//     let num_recommendations = metric_factory.config.length;
//     for (user_id, basket) in valid {
//         let neighbors = model.find_neighbors(user_id);
//         let recommended_items = model.predict_for(user_id, &neighbors, num_recommendations);
//
//         let actual_items: Vec<_> = basket
//             .items
//             .iter()
//             .map(|&item| Scored::new(item as u32, 1.0))
//             .collect();
//
//         let mut metric = metric_mutex.lock().unwrap();
//         metric.add(&recommended_items, &actual_items);
//     }
//     let metric = metric_mutex.into_inner().unwrap();
//     // Collect results from metrics
//     let results: (String, f64) = (metric.get_name(), metric.result());
//     log::info!("end evaluate_dataset_importance_metric");
//     results
// }
//
// fn evaluate_dataset(
//     model: &TIFUKNN,
//     user_embeddings_training_data: &HashMap<UserId, SparseItemVector>,
//     metric_factory: &MetricFactory,
//     valid: &HashMap<UserId, Basket>,
// ) -> Vec<(String, f64)> {
//     // Metrics need to be shared among threads, using Mutex to ensure thread safety
//     let metrics: Vec<Mutex<Box<dyn Metric + Send + Sync>>> = metric_factory
//         .create_evaluation_metrics()
//         .into_iter()
//         .map(Mutex::new)
//         .collect();
//     let num_recommendations = metric_factory.config.length;
//     let mut ids: Vec<UserId> = valid.keys().cloned().collect();
//     ids.par_sort();
//
//     ids.into_par_iter()
//         .map(|user_id| (user_id, valid.get(&user_id).unwrap()))
//         .for_each(|(user_id, basket)| {
//             let neighbors = model.find_neighbors(&user_id);
//             let recommended_items = model.predict_for(&user_id, &neighbors, num_recommendations);
//
//             let actual_items: Vec<_> = basket
//                 .items
//                 .iter()
//                 .map(|&item| Scored::new(item as u32, 1.0))
//                 .collect();
//
//             metrics.iter().for_each(|metric| {
//                 let mut metric = metric.lock().unwrap();
//                 metric.add(&recommended_items, &actual_items);
//             });
//         });
//
//     // Collect results from metrics
//     let results: Vec<(String, f64)> = metrics
//         .into_iter()
//         .map(|metric| {
//             let metric = metric.into_inner().unwrap();
//             (metric.get_name(), metric.result())
//         })
//         .collect();
//
//     results
// }
//
// fn random_score_dataset(
//     model: &TIFUKNN,
//     user_embeddings_training_data: &HashMap<UserId, SparseItemVector>,
//     metrics_factory: &MetricFactory,
//     valid: &HashMap<UserId, Basket>,
// ) -> (f64, f64) {
//     let (_random_score, _random_stddev_score) = (0.01, 0.01);
//
//     let all_heldout_values: Vec<_> = valid
//         .values()
//         .flat_map(|basket| basket.items.iter())
//         .sorted()
//         .cloned()
//         .collect();
//
//     let keys: Vec<UserId> = valid.keys().cloned().collect();
//     // let num_recommendations = metrics_factory.config.length;
//
//     let scores: Vec<f64> = (0..100)
//         .into_par_iter()
//         .enumerate()
//         .map(|(iteration, _)| {
//             let mut local_heldout_values = all_heldout_values.clone();
//             let mut local_keys = keys.clone();
//
//             let seed = (iteration as u64) + 1; // Unique seed for each iteration
//             let mut local_rng = StdRng::seed_from_u64(seed);
//             local_keys.shuffle(&mut local_rng);
//             local_heldout_values.shuffle(&mut local_rng);
//
//             let mut metric_binding: Box<dyn Metric + Send + Sync> =
//                 metrics_factory.create_importance_metric();
//             let metric = metric_binding.as_mut();
//
//             let num_recommendations = metrics_factory.config.length;
//
//             let mut idx = 0;
//
//             local_keys.iter().for_each(|&user_id| {
//                 if let Some(basket) = valid.get(&user_id) {
//                     let max = idx + basket.items.len();
//                     let actual_items = local_heldout_values[idx..max]
//                         .iter()
//                         .map(|&id| Scored::new(id as u32, 1.0))
//                         .collect_vec();
//                     idx = max;
//
//                     let neighbors = model.find_neighbors(&user_id);
//                     let recommended_items =
//                         model.predict_for(&user_id, &neighbors, num_recommendations);
//
//                     metric.add(&recommended_items, &actual_items);
//                 }
//             });
//
//             metric.result()
//         })
//         .collect();
//
//     let (mean_score, stddev_score) = mean_stddev(&scores);
//     (mean_score, stddev_score)
// }
//
//
// fn create_file(name: &str) -> File {
//     OpenOptions::new()
//         .create_new(true)
//         .write(true)
//         .append(false)
//         .open(name)
//         .expect(&format!("Failed to create file: {}", name))
// }

fn main() {

}