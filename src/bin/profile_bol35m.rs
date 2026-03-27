use illoominate::importance::score_sessions;
use illoominate::metrics::product_info::ProductInfo;
use illoominate::metrics::{Metric, MetricConfig, MetricFactory, MetricType};
use illoominate::sessrec::io::read_data;
use illoominate::sessrec::types::SessionDataset;
use illoominate::sessrec::vmisknn::{
    profile_find_neighbors, Scored, SimilarityComputationNew, VMISKNN,
};
use std::collections::HashSet;
use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() {
    let train_path = env::var("BOL35M_TRAIN")
        .unwrap_or_else(|_| "data/bol35m/bolcom-clicks-50m_train.txt".to_string());
    let valid_path = env::var("BOL35M_VALID")
        .unwrap_or_else(|_| "data/bol35m/bolcom-clicks-50m_test.txt".to_string());
    let m = env::var("BOL35M_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize);
    let k = env::var("BOL35M_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize);

    println!("train_path={train_path}");
    println!("valid_path={valid_path}");
    println!("m={m} k={k}");

    let started = Instant::now();
    let train_interactions = read_data(&train_path);
    let train_read_s = started.elapsed().as_secs_f64();
    println!(
        "read_train_s={:.3} interactions={}",
        train_read_s,
        train_interactions.len()
    );

    let started = Instant::now();
    let valid_interactions = read_data(&valid_path);
    let valid_read_s = started.elapsed().as_secs_f64();
    println!(
        "read_valid_s={:.3} interactions={}",
        valid_read_s,
        valid_interactions.len()
    );

    let started = Instant::now();
    let train_dataset = SessionDataset::new(train_interactions);
    let train_dataset_s = started.elapsed().as_secs_f64();
    println!(
        "build_train_dataset_s={:.3} sessions={}",
        train_dataset_s,
        train_dataset.sessions.len()
    );

    let started = Instant::now();
    let valid_dataset = SessionDataset::new(valid_interactions);
    let valid_dataset_s = started.elapsed().as_secs_f64();
    println!(
        "build_valid_dataset_s={:.3} sessions={}",
        valid_dataset_s,
        valid_dataset.sessions.len()
    );

    let started = Instant::now();
    let model = Arc::new(VMISKNN::fit_dataset(&train_dataset, m, k, 1.0));
    let fit_s = started.elapsed().as_secs_f64();
    println!("fit_dataset_s={:.3}", fit_s);

    let metric_config = MetricConfig {
        importance_metric: MetricType::MRR,
        evaluation_metrics: vec![MetricType::MRR],
        length: 20,
        alpha: 0.8,
    };
    let metric_factory = MetricFactory::new(&metric_config, ProductInfo::new(HashSet::new()));
    let metric: Arc<Mutex<Box<dyn Metric + Send + Sync>>> =
        Arc::new(Mutex::new(metric_factory.create_importance_metric()));

    let retrieve_ns = Arc::new(AtomicU64::new(0));
    let retrieve_prepare_ns = Arc::new(AtomicU64::new(0));
    let retrieve_accumulate_ns = Arc::new(AtomicU64::new(0));
    let retrieve_rank_ns = Arc::new(AtomicU64::new(0));
    let predict_ns = Arc::new(AtomicU64::new(0));
    let predict_build_ns = Arc::new(AtomicU64::new(0));
    let predict_match_ns = Arc::new(AtomicU64::new(0));
    let predict_accumulate_ns = Arc::new(AtomicU64::new(0));
    let predict_rank_ns = Arc::new(AtomicU64::new(0));
    let metric_ns = Arc::new(AtomicU64::new(0));
    let query_count = Arc::new(AtomicUsize::new(0));
    let retrieve_similar_session_visits = Arc::new(AtomicU64::new(0));
    let retrieve_unique_query_items = Arc::new(AtomicU64::new(0));
    let neighbor_count = Arc::new(AtomicU64::new(0));
    let candidate_count = Arc::new(AtomicU64::new(0));

    let started = Instant::now();
    score_sessions(&valid_dataset, {
        let retrieve_ns = Arc::clone(&retrieve_ns);
        let retrieve_prepare_ns = Arc::clone(&retrieve_prepare_ns);
        let retrieve_accumulate_ns = Arc::clone(&retrieve_accumulate_ns);
        let retrieve_rank_ns = Arc::clone(&retrieve_rank_ns);
        let predict_ns = Arc::clone(&predict_ns);
        let predict_build_ns = Arc::clone(&predict_build_ns);
        let predict_match_ns = Arc::clone(&predict_match_ns);
        let predict_accumulate_ns = Arc::clone(&predict_accumulate_ns);
        let predict_rank_ns = Arc::clone(&predict_rank_ns);
        let metric_ns = Arc::clone(&metric_ns);
        let query_count = Arc::clone(&query_count);
        let retrieve_similar_session_visits = Arc::clone(&retrieve_similar_session_visits);
        let retrieve_unique_query_items = Arc::clone(&retrieve_unique_query_items);
        let neighbor_count = Arc::clone(&neighbor_count);
        let candidate_count = Arc::clone(&candidate_count);
        let metric = Arc::clone(&metric);
        let model = Arc::clone(&model);

        move |query_session, actual_next_items| {
            let started = Instant::now();
            let profiled_neighbors = profile_find_neighbors(&model.index, query_session, k, m);
            retrieve_prepare_ns.fetch_add(profiled_neighbors.stats.prepare_ns, Ordering::Relaxed);
            retrieve_accumulate_ns
                .fetch_add(profiled_neighbors.stats.accumulate_ns, Ordering::Relaxed);
            retrieve_rank_ns.fetch_add(profiled_neighbors.stats.rank_ns, Ordering::Relaxed);
            retrieve_similar_session_visits.fetch_add(
                profiled_neighbors.stats.similar_session_visits,
                Ordering::Relaxed,
            );
            retrieve_unique_query_items.fetch_add(
                profiled_neighbors.stats.unique_query_items,
                Ordering::Relaxed,
            );
            let neighbors = profiled_neighbors.neighbors;
            retrieve_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let started = Instant::now();
            let recommended_items = profile_predict_for(
                &model,
                query_session,
                &neighbors,
                21,
                &predict_build_ns,
                &predict_match_ns,
                &predict_accumulate_ns,
                &predict_rank_ns,
                &neighbor_count,
                &candidate_count,
            );
            predict_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let started = Instant::now();
            metric
                .lock()
                .unwrap()
                .add(&recommended_items, actual_next_items);
            metric_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            query_count.fetch_add(1, Ordering::Relaxed);
        }
    });
    let evaluate_wall_s = started.elapsed().as_secs_f64();

    let retrieve_s = retrieve_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let retrieve_prepare_s = retrieve_prepare_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let retrieve_accumulate_s = retrieve_accumulate_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let retrieve_rank_s = retrieve_rank_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let predict_s = predict_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let predict_build_s = predict_build_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let predict_match_s = predict_match_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let predict_accumulate_s = predict_accumulate_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let predict_rank_s = predict_rank_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let metric_s = metric_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let queries = query_count.load(Ordering::Relaxed);
    let total_similar_session_visits = retrieve_similar_session_visits.load(Ordering::Relaxed);
    let total_unique_query_items = retrieve_unique_query_items.load(Ordering::Relaxed);
    let total_neighbors = neighbor_count.load(Ordering::Relaxed);
    let total_candidates = candidate_count.load(Ordering::Relaxed);
    let score = metric.lock().unwrap().result();

    println!("evaluate_wall_s={:.3}", evaluate_wall_s);
    println!("retrieve_cpu_s={:.3}", retrieve_s);
    println!("retrieve_prepare_cpu_s={:.3}", retrieve_prepare_s);
    println!("retrieve_accumulate_cpu_s={:.3}", retrieve_accumulate_s);
    println!("retrieve_rank_cpu_s={:.3}", retrieve_rank_s);
    println!("predict_cpu_s={:.3}", predict_s);
    println!("predict_build_cpu_s={:.3}", predict_build_s);
    println!("predict_match_cpu_s={:.3}", predict_match_s);
    println!("predict_accumulate_cpu_s={:.3}", predict_accumulate_s);
    println!("predict_rank_cpu_s={:.3}", predict_rank_s);
    println!("metric_cpu_s={:.3}", metric_s);
    println!("queries={queries}");
    println!(
        "avg_unique_query_items={:.3}",
        total_unique_query_items as f64 / queries as f64
    );
    println!(
        "avg_similar_session_visits_per_query={:.3}",
        total_similar_session_visits as f64 / queries as f64
    );
    println!(
        "avg_neighbors_per_query={:.3}",
        total_neighbors as f64 / queries as f64
    );
    println!(
        "avg_raw_contributions_per_query={:.3}",
        total_candidates as f64 / queries as f64
    );
    println!("mrr20={:.12}", score);
}

fn profile_predict_for(
    model: &VMISKNN,
    session: &Vec<Scored>,
    neighbors: &Vec<Scored>,
    how_many: usize,
    predict_build_ns: &AtomicU64,
    predict_match_ns: &AtomicU64,
    predict_accumulate_ns: &AtomicU64,
    predict_rank_ns: &AtomicU64,
    neighbor_count: &AtomicU64,
    candidate_count: &AtomicU64,
) -> Vec<Scored> {
    let mut contributions: Vec<(u64, f64)> = Vec::with_capacity(neighbors.len() * 8);
    let mut local_match_ns = 0u64;
    let mut local_accumulate_ns = 0u64;
    let local_neighbor_count = neighbors.len() as u64;
    let build_started = Instant::now();

    for scored_session in neighbors.iter() {
        let training_item_ids = model.index.items_for_session(&scored_session.id);

        let started = Instant::now();
        let first_match_pos = profiled_first_match_position(session, training_item_ids);
        local_match_ns += started.elapsed().as_nanos() as u64;

        let Some(first_match_pos) = first_match_pos else {
            continue;
        };
        let session_weight = linear_score(first_match_pos);
        if session_weight <= 0.0 {
            continue;
        }
        let weighted_score = session_weight * scored_session.score;

        let started = Instant::now();
        for item_id in training_item_ids.iter() {
            contributions.push((*item_id, weighted_score));
        }
        local_accumulate_ns += started.elapsed().as_nanos() as u64;
    }

    let local_candidate_count = contributions.len() as u64;
    let local_build_ns = build_started.elapsed().as_nanos() as u64;

    let started = Instant::now();
    let top_items = top_scored_items_from_contributions(
        contributions,
        how_many,
        session.last().unwrap().id as u64,
    );
    let local_rank_ns = started.elapsed().as_nanos() as u64;

    predict_build_ns.fetch_add(local_build_ns, Ordering::Relaxed);
    predict_match_ns.fetch_add(local_match_ns, Ordering::Relaxed);
    predict_accumulate_ns.fetch_add(local_accumulate_ns, Ordering::Relaxed);
    predict_rank_ns.fetch_add(local_rank_ns, Ordering::Relaxed);
    neighbor_count.fetch_add(local_neighbor_count, Ordering::Relaxed);
    candidate_count.fetch_add(local_candidate_count, Ordering::Relaxed);
    top_items
}

fn profiled_first_match_position(
    query_session: &[Scored],
    training_item_ids: &[u64],
) -> Option<usize> {
    query_session
        .iter()
        .rev()
        .take(9)
        .position(|item_id| training_item_ids.contains(&(item_id.id as u64)))
        .map(|index| index + 1)
}

fn linear_score(pos: usize) -> f64 {
    if pos < 10 {
        1.0 - (0.1 * pos as f64)
    } else {
        0.0
    }
}

fn top_scored_items_from_contributions(
    mut contributions: Vec<(u64, f64)>,
    how_many: usize,
    excluded_item_id: u64,
) -> Vec<Scored> {
    if how_many == 0 || contributions.is_empty() {
        return Vec::new();
    }

    contributions.sort_unstable_by_key(|(item_id, _)| *item_id);

    let mut aggregated_scores: Vec<Scored> = Vec::with_capacity(contributions.len());
    let mut contributions_iter = contributions.into_iter();

    if let Some((mut current_item, mut current_score)) = contributions_iter.next() {
        for (item_id, score) in contributions_iter {
            if item_id == current_item {
                current_score += score;
            } else {
                if current_item != excluded_item_id {
                    aggregated_scores.push(Scored::new(current_item as u32, current_score));
                }
                current_item = item_id;
                current_score = score;
            }
        }

        if current_item != excluded_item_id {
            aggregated_scores.push(Scored::new(current_item as u32, current_score));
        }
    }

    if aggregated_scores.len() > how_many {
        aggregated_scores.select_nth_unstable(how_many);
        aggregated_scores.truncate(how_many);
    }

    aggregated_scores.sort_unstable();
    aggregated_scores.truncate(how_many);
    aggregated_scores
}
