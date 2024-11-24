use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::importance::RetrievalBasedModel;
use dary_heap::OctonaryHeap;
use itertools::Itertools;

use crate::nbr::caboose::types::{SimilarRow};
use crate::sessrec::types::{ItemId, SessionDataset, SessionId, Time};
use topk::TopK;

pub mod topk;

#[derive(Debug)]
pub struct VMISIndex {
    pub(crate) item_to_top_sessions_ordered: HashMap<u64, Vec<u32>>,
    pub session_to_max_time_stamp: Vec<u32>,
    pub(crate) item_to_idf_score: HashMap<u64, f64>,
    pub session_to_items_sorted: Vec<Vec<u64>>,
    pub session_idx_to_id: HashMap<usize, usize>,
}

pub struct VMISKNN {
    pub index: VMISIndex,
    m: usize,
    k: usize, // top-k scored historical sessions out of the 'm' historical sessions
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
        let item_to_idf_score = create_item_idf(training_dataset);
        let session_to_items_sorted = create_session_to_items(training_dataset);
        VMISKNN {
            index: VMISIndex {
                item_to_top_sessions_ordered,
                session_to_max_time_stamp,
                item_to_idf_score,
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
        let mut item_scores: HashMap<u64, f64> = HashMap::with_capacity(1000);

        for ss in neighbors.iter() {
            let scored_session = ss.as_ref();
            let training_item_ids: &[u64] = self.index.items_for_session(&scored_session.id);

            let (first_match_index, _) = session
                .iter()
                .rev()
                .enumerate()
                .find(|(_, &id_score)| training_item_ids.contains(&(id_score.id as ItemId)))
                .unwrap();

            let first_match_pos = first_match_index + 1;

            let session_weight = linear_score(first_match_pos);

            for item_id in training_item_ids.iter() {
                // let _item_idf = self.index.item_to_idf_score[item_id];
                let item_idf = 0.0;
                if item_idf > 0.0 {
                    *item_scores.entry(*item_id).or_insert(0.0) +=
                        session_weight * item_idf * scored_session.score;
                } else {
                    *item_scores.entry(*item_id).or_insert(0.0) +=
                        session_weight * scored_session.score;
                }
            }
        }

        // Remove most recent item if it has been scored as well
        let most_recent_item = *session.last().unwrap();
        if let Entry::Occupied(entry) = item_scores.entry(most_recent_item.id as ItemId) {
            entry.remove_entry();
        }
        item_scores
            .iter_mut()
            .sorted_by(|a, b| match a.1.partial_cmp(&b.1) {
                Some(Ordering::Less) => Ordering::Greater,
                Some(Ordering::Greater) => Ordering::Less,
                _ => a.0.cmp(b.0),
            })
            .take(how_many)
            .map(|(&item, weight)| Scored::new(item as u32, *weight))
            .collect_vec()
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

        let (first_match_index, _) = query_session
            .iter()
            .rev()
            .enumerate()
            .find(|(_, item_id)| training_item_ids.contains(&(item_id.id as u64)))
            .unwrap();

        let first_match_pos = first_match_index + 1;

        let session_weight = linear_score(first_match_pos);

        for item_id in training_item_ids.iter() {
            // let _item_idf = self.index.item_to_idf_score[item_id];
            let item_idf = 0.0;
            if item_idf > 0.0 {
                *agg.entry(*item_id).or_insert(0.0) +=
                    session_weight * item_idf * scored_session.score;
            } else {
                *agg.entry(*item_id).or_insert(0.0) += session_weight * scored_session.score;
            }
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

        let (first_match_index, _) = query_session
            .iter()
            .rev()
            .enumerate()
            .find(|(_, item_id)| training_item_ids.contains(&(item_id.id as ItemId)))
            .unwrap();

        let first_match_pos = first_match_index + 1;

        let session_weight = linear_score(first_match_pos);

        for item_id in training_item_ids.iter() {
            // TODO the or_insert should never happen!
            let item_idf = 0.0;
            if item_idf > 0.0 {
                *agg.entry(*item_id).or_insert(0.0) -=
                    session_weight * item_idf * scored_session.score;
            } else {
                *agg.entry(*item_id).or_insert(0.0) -= session_weight * scored_session.score;
            }
        }
    }

    fn generate_from_preaggregate(
        &self,
        query_session: &Vec<Scored>,
        agg: &HashMap<u64, f64>,
    ) -> Vec<Scored> {
        let most_recent_item = *query_session.last().unwrap();

        agg.iter()
            .filter(|(&item, _)| item != most_recent_item.id as u64) // Skip most recent item
            .sorted_by(|a, b| match a.1.partial_cmp(b.1) {
                Some(Ordering::Less) => Ordering::Greater,
                Some(Ordering::Greater) => Ordering::Less,
                _ => a.0.cmp(b.0),
            })
            .take(21)
            .map(|(&item, _)| Scored::new(item as u32, 1.0))
            .collect_vec()
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
        // reverse order by time
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
        // == is defined as being based on the contents of an object.
        self.session_id == other.session_id
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

    fn idf(&self, item_id: &u64) -> f64;

    /// find neighboring sessions for the given evolving_session.
    /// param m select the 'm' most recent historical sessions
    /// param k defines the top 'k' scored historical sessions out of the 'm' historical sessions.
    fn find_neighbors(&self, evolving_session: &Vec<Scored>, k: usize, m: usize) -> Vec<Scored>;
}

impl SimilarityComputationNew for VMISIndex {
    fn items_for_session(&self, session: &u32) -> &[u64] {
        &self.session_to_items_sorted[*session as usize]
    }

    fn idf(&self, item: &u64) -> f64 {
        self.item_to_idf_score[item]
    }

    fn find_neighbors(&self, evolving_session: &Vec<Scored>, k: usize, m: usize) -> Vec<Scored> {
        // We use a d-ary heap for the (timestamp, session_id) tuple, a hashmap for the (session_id, score) tuples, and a hashmap for the unique items in the evolving session
        let mut heap_timestamps = OctonaryHeap::<SessionTime>::with_capacity(m);
        let mut session_similarities = HashMap::with_capacity(m);
        let len_evolving_session = evolving_session.len();
        let mut unique = evolving_session
            .iter()
            .map(|id_score| id_score.id)
            .collect_vec();
        unique.sort_unstable();
        unique.dedup();

        let qty_unique_session_items = unique.len() as f64;

        let mut hash_items = HashMap::with_capacity(len_evolving_session);

        //  Loop over items in evolving session in reverse order
        for (pos, id_score) in evolving_session.iter().rev().enumerate() {
            let item_id = id_score.id as u64;
            // Duplicate items: only calculate similarity score for the item in the farthest position in the evolving session
            match hash_items.insert(item_id, pos) {
                Some(_) => {}
                None => {
                    // Find similar sessions in training data
                    if let Some(similar_sessions) = self.item_to_top_sessions_ordered.get(&item_id)
                    {
                        let decay_factor =
                            (len_evolving_session - pos) as f64 / qty_unique_session_items;
                        // Loop over all similar sessions.
                        'session_loop: for session_id in similar_sessions {
                            match session_similarities.get_mut(session_id) {
                                Some(similarity) => *similarity += decay_factor,
                                None => {
                                    let session_time_stamp =
                                        self.session_to_max_time_stamp[*session_id as usize];
                                    if session_similarities.len() < m {
                                        session_similarities.insert(*session_id, decay_factor);
                                        heap_timestamps.push(SessionTime::new(
                                            *session_id,
                                            session_time_stamp,
                                        ));
                                    } else {
                                        let mut bottom = heap_timestamps.peek_mut().unwrap();
                                        if session_time_stamp > bottom.time {
                                            // log::debug!("{:?} {:?}", session_time_stamp, bottom.time);
                                            // Remove the the existing minimum time stamp.
                                            session_similarities.remove_entry(&bottom.session_id);
                                            // Set new minimum timestamp
                                            session_similarities.insert(*session_id, decay_factor);
                                            *bottom =
                                                SessionTime::new(*session_id, session_time_stamp);
                                        } else {
                                            break 'session_loop;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut session_similarity_keys: Vec<_> = session_similarities.keys().collect();
        session_similarity_keys.sort();

        //dbg!(&session_similarities);
        let mut topk = TopK::new(k);
        //for (session_id, score) in session_similarities.iter() {
        for session_id in session_similarity_keys {
            let score = session_similarities.get(session_id).unwrap();
            let scored_session = Scored::new(*session_id, *score);
            topk.add(scored_session, &self.session_to_max_time_stamp);
        }
        // Closest neigbours contain unique session_ids and corresponding top-k similarity scores.as
        // Return top-k neighbors in arbitrary order.
        topk.iter().cloned().collect_vec()
    }
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

fn create_item_idf(sessions: &SessionDataset) -> HashMap<u64, f64> {
    // TODO why is this not used?
    // let qty_total_sessions: usize= sessions.sessions.keys().len();
    let mut unique_sessions_count: HashMap<u64, HashSet<SessionId>> = HashMap::new();
    // Count unique sessions for each item
    for (&session_id, (item_ids, _)) in sessions.sessions.iter() {
        for item_id in item_ids {
            let item_sessions = unique_sessions_count
                .entry(*item_id)
                .or_default();
            item_sessions.insert(session_id);
        }
    }
    // let result: HashMap<u64, f64> = unique_sessions_count.iter().map(|(item_id, session_ids)| (*item_id, (qty_total_sessions as f64/ session_ids.len() as f64).ln() )).collect();
    let result: HashMap<u64, f64> = unique_sessions_count.keys().map(|item_id| (*item_id, 1.0))
        .collect();
    result
}

fn create_most_recent_sessions_per_item(
    sessions: &SessionDataset,
    m: usize,
) -> HashMap<ItemId, Vec<SessionId>> {
    // HashMap to store the XXXX most recent TrainingSessionIds for each ItemId
    let mut result: HashMap<ItemId, Vec<(Reverse<Time>, SessionId)>> = HashMap::new();

    for (session_id, (item_ids, time)) in sessions.sessions.iter() {
        for item_id in item_ids.clone().iter().unique() {
            // Retrieve or create an entry for the item_id
            let entry = result.entry(*item_id).or_default();

            // Insert the session_id into the binary heap (min heap)
            entry.push((Reverse(*time), *session_id));
            entry.sort_by_key(|&(Reverse(key), _)| key);

            // Truncate to keep only the `m` most recent session_ids
            if entry.len() > m {
                entry.truncate(m);
            }
        }
    }
    // Extract just the session ids from each entry
    
    result
        .into_iter()
        .map(|(item_id, sessions)| {
            (
                item_id,
                sessions
                    .into_iter()
                    .map(|(_, session_id)| session_id)
                    .collect(),
            )
        })
        .collect()
}
