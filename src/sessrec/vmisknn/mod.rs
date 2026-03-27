use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::importance::RetrievalBasedModel;
use dary_heap::OctonaryHeap;
use itertools::Itertools;

use crate::nbr::caboose::types::SimilarRow;
use crate::sessrec::types::{ItemId, SessionDataset, SessionId, Time};
use topk::TopK;

pub mod topk;

#[derive(Debug)]
pub struct VMISIndex {
    pub(crate) item_to_top_sessions_ordered: HashMap<u64, Vec<u32>>,
    pub session_to_max_time_stamp: Vec<u32>,
    pub session_to_items_sorted: Vec<Vec<u64>>,
    pub session_idx_to_id: HashMap<usize, usize>,
}

pub struct VMISKNN {
    pub index: VMISIndex,
    m: usize,
    k: usize, // top-k scored historical sessions out of the 'm' historical sessions
}

#[derive(Clone, Debug, Default)]
pub struct FindNeighborsStats {
    pub prepare_ns: u64,
    pub accumulate_ns: u64,
    pub rank_ns: u64,
    pub unique_query_items: u64,
    pub similar_session_visits: u64,
}

#[derive(Clone, Debug)]
pub struct ProfiledNeighbors {
    pub neighbors: Vec<Scored>,
    pub stats: FindNeighborsStats,
}

#[derive(Eq, Debug)]
pub struct SessionTime {
    pub session_id: u32,
    pub time: u32,
}

impl SessionTime {
    pub fn new(session_id: u32, time: u32) -> Self {
        SessionTime { session_id, time }
    }
}

impl Ord for SessionTime {
    fn cmp(&self, other: &Self) -> Ordering {
        other.time.cmp(&self.time)
    }
}

impl PartialOrd for SessionTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SessionTime {
    fn eq(&self, other: &Self) -> bool {
        self.session_id == other.session_id
    }
}

impl VMISKNN {
    pub fn fit_dataset(
        training_dataset: &SessionDataset,
        m_most_recent_sessions: usize,
        k: usize,
        _idf_weighting: f64,
    ) -> Self {
        let item_to_top_sessions_ordered =
            create_most_recent_sessions_per_item(training_dataset, m_most_recent_sessions);
        let session_to_max_time_stamp = create_session_to_max_time_stamp(training_dataset);
        let session_to_items_sorted = create_session_to_items(training_dataset);
        VMISKNN {
            index: VMISIndex {
                item_to_top_sessions_ordered,
                session_to_max_time_stamp,
                session_to_items_sorted,
                session_idx_to_id: HashMap::new(),
            },
            m: m_most_recent_sessions,
            k,
        }
    }

    pub fn predict_for<T>(
        &self,
        session: &Vec<Scored>,
        neighbors: &[T],
        how_many: usize,
    ) -> Vec<Scored>
    where
        T: AsRef<Scored>,
    {
        let mut contributions: Vec<(u64, f64)> = Vec::with_capacity(neighbors.len() * 8);

        for ss in neighbors.iter() {
            let scored_session = ss.as_ref();
            let training_item_ids: &[u64] = self.index.items_for_session(&scored_session.id);

            let Some(first_match_pos) = first_match_position(session, training_item_ids) else {
                continue;
            };
            let session_weight = linear_score(first_match_pos);
            if session_weight <= 0.0 {
                continue;
            }
            let weighted_score = session_weight * scored_session.score;

            for item_id in training_item_ids.iter() {
                contributions.push((*item_id, weighted_score));
            }
        }

        top_scored_items_from_contributions(
            contributions,
            how_many,
            session.last().unwrap().id as ItemId,
        )
    }

    // Function to predict based on given session data
    pub fn predict(&self, session: &Vec<Scored>) -> Vec<Scored> {
        let neighbors = self.index.find_neighbors(session, self.k, self.m);
        self.predict_for(session, &neighbors, 21)
    }
}

impl RetrievalBasedModel for VMISKNN {
    fn k(&self) -> usize {
        self.k
    }

    fn retrieve_k(&self, query_session: &Vec<Scored>) -> Vec<Scored> {
        self.index.find_neighbors(query_session, self.k, self.m)
    }

    fn retrieve_all(&self, query_session: &Vec<Scored>) -> Vec<Scored> {
        self.index.find_neighbors(query_session, self.m, self.m)
    }

    // fn generate<T: AsRef<Scored>>(&self, session: &Vec<Scored>, neighbors: &[T]) -> Vec<Scored> {
    //     self.predict_for(session, neighbors, 21)
    // }

    fn create_preaggregate(&self) -> HashMap<u64, f64> {
        HashMap::with_capacity(1000)
    }

    fn add_to_preaggregate<T: AsRef<Scored>>(
        &self,
        agg: &mut HashMap<u64, f64>,
        query_session: &Vec<Scored>,
        neighbor: &T,
    ) {
        let scored_session = neighbor.as_ref();
        let training_item_ids: &[u64] = self.index.items_for_session(&scored_session.id);

        let Some(first_match_pos) = first_match_position(query_session, training_item_ids) else {
            return;
        };
        let session_weight = linear_score(first_match_pos);
        if session_weight <= 0.0 {
            return;
        }
        let weighted_score = session_weight * scored_session.score;

        for item_id in training_item_ids.iter() {
            *agg.entry(*item_id).or_insert(0.0) += weighted_score;
        }
    }

    fn remove_from_preaggregate<T: AsRef<Scored>>(
        &self,
        agg: &mut HashMap<u64, f64>,
        query_session: &Vec<Scored>,
        neighbor: &T,
    ) {
        let scored_session = neighbor.as_ref();
        let training_item_ids: &[u64] = self.index.items_for_session(&scored_session.id);

        let Some(first_match_pos) = first_match_position(query_session, training_item_ids) else {
            return;
        };
        let session_weight = linear_score(first_match_pos);
        if session_weight <= 0.0 {
            return;
        }
        let weighted_score = session_weight * scored_session.score;

        for item_id in training_item_ids.iter() {
            *agg.entry(*item_id).or_insert(0.0) -= weighted_score;
        }
    }

    fn generate_from_preaggregate(
        &self,
        query_session: &Vec<Scored>,
        agg: &HashMap<u64, f64>,
    ) -> Vec<Scored> {
        let most_recent_item = *query_session.last().unwrap();
        top_scored_items_excluding(agg, 21, most_recent_item.id as u64)
            .into_iter()
            .map(|scored| Scored::new(scored.id, 1.0))
            .collect()
    }

    fn predict(
        &self,
        query: &Vec<Scored>,
        neighbors: &Vec<Scored>,
        how_many: usize,
    ) -> Vec<Scored> {
        self.predict_for(query, neighbors, how_many)
    }
}

// TODO this should go to types
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Scored {
    pub id: u32,
    pub score: f64,
}

impl Scored {
    pub fn new(id: u32, score: f64) -> Self {
        Scored { id, score }
    }
}

impl From<&SimilarRow> for Scored {
    fn from(row: &SimilarRow) -> Self {
        Scored {
            id: row.row,
            score: row.similarity as f64,
        }
    }
}

impl AsRef<Scored> for Scored {
    fn as_ref(&self) -> &Scored {
        self
    }
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => Ordering::Greater,
            Some(Ordering::Greater) => Ordering::Less,
            //_ => Ordering::Equal,
            _ => self.id.cmp(&other.id),
        }
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn linear_score(pos: usize) -> f64 {
    if pos < 10 {
        1.0 - (0.1 * pos as f64)
    } else {
        0.0
    }
}

pub trait SimilarityComputationNew {
    fn items_for_session(&self, session_idx: &u32) -> &[u64];

    /// find neighboring sessions for the given evolving_session.
    /// param m select the 'm' most recent historical sessions
    /// param k defines the top 'k' scored historical sessions out of the 'm' historical sessions.
    fn find_neighbors(&self, evolving_session: &Vec<Scored>, k: usize, m: usize) -> Vec<Scored>;
}

impl SimilarityComputationNew for VMISIndex {
    fn items_for_session(&self, session: &u32) -> &[u64] {
        &self.session_to_items_sorted[*session as usize]
    }

    fn find_neighbors(&self, evolving_session: &Vec<Scored>, k: usize, m: usize) -> Vec<Scored> {
        find_neighbors_internal(self, evolving_session, k, m, None)
    }
}

pub fn profile_find_neighbors(
    index: &VMISIndex,
    evolving_session: &Vec<Scored>,
    k: usize,
    m: usize,
) -> ProfiledNeighbors {
    let mut stats = FindNeighborsStats::default();
    let neighbors = find_neighbors_internal(index, evolving_session, k, m, Some(&mut stats));
    ProfiledNeighbors { neighbors, stats }
}

fn find_neighbors_internal(
    index: &VMISIndex,
    evolving_session: &Vec<Scored>,
    k: usize,
    m: usize,
    stats: Option<&mut FindNeighborsStats>,
) -> Vec<Scored> {
    let prepare_started = std::time::Instant::now();
    let len_evolving_session = evolving_session.len();
    let qty_unique_session_items = evolving_session
        .iter()
        .map(|id_score| id_score.id)
        .collect::<HashSet<_>>()
        .len();
    let mut seen_items = HashSet::with_capacity(qty_unique_session_items);
    let prepare_ns = prepare_started.elapsed().as_nanos() as u64;

    let accumulate_started = std::time::Instant::now();
    let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(m);
    let mut session_similarities = HashMap::with_capacity(m);
    let mut similar_session_visits = 0u64;

    for (pos, id_score) in evolving_session.iter().rev().enumerate() {
        let item_id = id_score.id as u64;
        if !seen_items.insert(item_id) {
            continue;
        }

        if let Some(similar_sessions) = index.item_to_top_sessions_ordered.get(&item_id) {
            let decay_factor =
                (len_evolving_session - pos) as f64 / qty_unique_session_items as f64;
            'session_loop: for session_id in similar_sessions {
                similar_session_visits += 1;
                match session_similarities.get_mut(session_id) {
                    Some(similarity) => *similarity += decay_factor,
                    None => {
                        let session_time_stamp =
                            index.session_to_max_time_stamp[*session_id as usize];
                        if session_similarities.len() < m {
                            session_similarities.insert(*session_id, decay_factor);
                            heap_timestamps.push(SessionTime::new(*session_id, session_time_stamp));
                        } else {
                            let mut bottom = heap_timestamps.peek_mut().unwrap();
                            if session_time_stamp > bottom.time {
                                session_similarities.remove_entry(&bottom.session_id);
                                session_similarities.insert(*session_id, decay_factor);
                                *bottom = SessionTime::new(*session_id, session_time_stamp);
                            } else {
                                break 'session_loop;
                            }
                        }
                    }
                }
            }
        }
    }
    let accumulate_ns = accumulate_started.elapsed().as_nanos() as u64;

    let rank_started = std::time::Instant::now();
    let mut topk = TopK::new(k);
    let mut similarity_entries: Vec<_> = session_similarities.into_iter().collect();
    similarity_entries.sort_unstable_by_key(|(session_id, _)| *session_id);
    for (session_id, score) in similarity_entries {
        topk.add(
            Scored::new(session_id, score),
            &index.session_to_max_time_stamp,
        );
    }
    let neighbors = topk.iter().cloned().collect_vec();
    let rank_ns = rank_started.elapsed().as_nanos() as u64;

    if let Some(stats) = stats {
        stats.prepare_ns = prepare_ns;
        stats.accumulate_ns = accumulate_ns;
        stats.rank_ns = rank_ns;
        stats.unique_query_items = qty_unique_session_items as u64;
        stats.similar_session_visits = similar_session_visits;
    }

    neighbors
}

fn create_session_to_items(sessions: &SessionDataset) -> Vec<Vec<u64>> {
    let max_session_id: usize = *sessions.sessions.keys().max().unwrap_or(&0) as usize;
    let mut result: Vec<Vec<u64>> = vec![vec![]; max_session_id + 1];

    for (&session_id, (items, _timestamp)) in sessions.sessions.iter() {
        let mut items: Vec<u64> = items
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        items.sort_unstable();
        result[session_id as usize] = items;
    }
    result
}

fn create_session_to_max_time_stamp(sessions: &SessionDataset) -> Vec<u32> {
    let max_session_id: usize = *sessions.sessions.keys().max().unwrap() as usize;
    let mut result = vec![0; max_session_id + 1];

    for (&session_id, (_items, timestamp)) in sessions.sessions.iter() {
        result[session_id as usize] = *timestamp as u32
    }
    result
}

fn first_match_position(query_session: &[Scored], training_item_ids: &[u64]) -> Option<usize> {
    query_session
        .iter()
        .rev()
        .take(9)
        .position(|item_id| training_item_ids.contains(&(item_id.id as u64)))
        .map(|index| index + 1)
}

fn top_scored_items_excluding(
    item_scores: &HashMap<u64, f64>,
    how_many: usize,
    excluded_item_id: u64,
) -> Vec<Scored> {
    collect_top_scored_items(item_scores.iter(), how_many, excluded_item_id)
}

fn collect_top_scored_items<'a>(
    item_scores: impl Iterator<Item = (&'a u64, &'a f64)>,
    how_many: usize,
    excluded_item_id: u64,
) -> Vec<Scored> {
    if how_many == 0 {
        return Vec::new();
    }

    let mut top_items: Vec<Scored> = item_scores
        .filter(|(item_id, _)| **item_id != excluded_item_id)
        .map(|(&item_id, &score)| Scored::new(item_id as u32, score))
        .collect();

    if top_items.len() > how_many {
        top_items.select_nth_unstable(how_many);
        top_items.truncate(how_many);
    }

    top_items.sort_unstable();
    top_items.truncate(how_many);
    top_items
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

pub fn create_most_recent_sessions_per_item(
    sessions: &SessionDataset,
    m: usize,
) -> HashMap<ItemId, Vec<SessionId>> {
    // Intermediate mapping: for each item, collect (time, session_id) pairs.
    let mut item_sessions: HashMap<ItemId, Vec<(Time, SessionId)>> = HashMap::new();

    // Iterate over each session in the dataset.
    for (session_id, (item_ids, time)) in sessions.sessions.iter() {
        // Use unique() to avoid counting duplicates within the same session.
        for &item_id in item_ids.iter().unique() {
            // Insert the (time, session_id) tuple for the given item.
            item_sessions
                .entry(item_id)
                .or_default()
                .push((*time, *session_id));
        }
    }

    // Build the final result.
    // For each item, sort by time descending (most recent first) and keep only m sessions.
    let mut result: HashMap<ItemId, Vec<SessionId>> = HashMap::new();
    for (item_id, mut sessions_vec) in item_sessions {
        sessions_vec.sort_by(|a, b| b.0.cmp(&a.0)); // descending order by time
        let recent_sessions = sessions_vec
            .into_iter()
            .take(m)
            .map(|(_, session_id)| session_id)
            .collect();
        result.insert(item_id, recent_sessions);
    }

    result
}
