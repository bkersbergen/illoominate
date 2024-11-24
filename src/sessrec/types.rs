use std::collections::HashMap;

use itertools::Itertools;

use crate::importance::{Dataset, DatasetEntry, Sequence};
use crate::sessrec::vmisknn::Scored;

pub type SessionId = u32;
pub type ItemId = u64;
pub type Time = usize;

#[derive(Clone, Debug)]
pub struct SessionDataset {
    pub sessions: HashMap<SessionId, (Vec<ItemId>, Time)>,
}

impl Dataset for SessionDataset {
    fn collect_keys(&self) -> Vec<u32> {
        self.sessions.keys().cloned().collect_vec()
    }

    fn num_interactions(&self) -> usize {
        self.sessions
            .iter()
            .map(|(_session_id, (items, _max_timestamp))| items.len() - 1)
            .sum()
    }

    fn __get_entry__(&self, key: u32) -> DatasetEntry {
        let mut sequences = Vec::new();
        let mut max_timestamp = 0;
        if let Some((session_items, session_max_timestamp)) = self.sessions.get(&key) {
            max_timestamp = *session_max_timestamp;
            for pos in 1..session_items.len() {
                let input = &session_items[..pos]
                    .iter()
                    .map(|&id| Scored::new(id as u32, 1.0))
                    .collect_vec();
                let target = &session_items[pos..]
                    .iter()
                    .map(|&id| Scored::new(id as u32, 1.0))
                    .collect_vec();
                sequences.push(Sequence {
                    input: input.clone(),
                    target: target.clone(),
                })
            }
        }
        DatasetEntry {
            key,
            sequences,
            max_timestamp,
        }
    }

    fn __get_items__(&self, key: u32) -> Vec<u32> {
        if let Some((session_items, _session_max_timestamp)) = self.sessions.get(&key) {
            let input = &session_items.iter().map(|&id| id as u32).collect_vec();
            input.clone()
        } else {
            Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.sessions.keys().len()
    }
}

#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct Interaction {
    pub session_id: SessionId,
    pub item_id: ItemId,
    pub timestamp: Time,
}

impl Interaction {
    pub fn new(session_id: SessionId, item_id: ItemId, timestamp: Time) -> Self {
        Self {
            session_id,
            item_id,
            timestamp,
        }
    }
}
