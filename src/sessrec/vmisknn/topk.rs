use crate::sessrec::vmisknn::Scored;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;

pub struct TopK {
    closest_neighbors: BinaryHeap<Scored>,
    k: usize,
}

impl TopK {
    pub(crate) fn new(k: usize) -> TopK {
        Self {
            closest_neighbors: BinaryHeap::with_capacity(k),
            k,
        }
    }

    pub(crate) fn add(&mut self, scored_session: Scored, session_id_to_max_time_stamp: &Vec<u32>) {
        if self.closest_neighbors.len() < self.k {
            self.closest_neighbors.push(scored_session);
        } else {
            let mut bottom = self.closest_neighbors.peek_mut().unwrap();
            if scored_session.score > bottom.score {
                *bottom = scored_session;
            } else if (scored_session.score - bottom.score).abs() < f64::EPSILON
                && (session_id_to_max_time_stamp[scored_session.id as usize]
                    > session_id_to_max_time_stamp[bottom.id as usize])
            {
                *bottom = scored_session;
            }
        }
    }

    /// Returns an iterator visiting all values in the topk, in
    /// arbitrary order.
    pub(crate) fn iter(&self) -> Iter<'_, Scored> {
        self.closest_neighbors.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_max_size() {
        let session_id_to_max_time_stamp: Vec<u32> = vec![10, 10, 10, 10, 10, 10];

        let k = 3;
        let mut top_k = TopK::new(k);
        top_k.add(Scored::new(1, 50.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(2, 70.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(3, 30.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(4, 80.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(5, 60.0), &session_id_to_max_time_stamp);

        assert_eq!(top_k.closest_neighbors.len(), k);
    }

    #[test]
    fn test_top_k_content() {
        let session_id_to_max_time_stamp: Vec<u32> = vec![10, 10, 10, 10, 10, 10];

        let k = 3;
        let mut top_k = TopK::new(k);
        top_k.add(Scored::new(1, 50.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(2, 70.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(3, 30.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(4, 80.0), &session_id_to_max_time_stamp);
        top_k.add(Scored::new(5, 60.0), &session_id_to_max_time_stamp);

        let expected_result = vec![
            Scored::new(4, 80.0),
            Scored::new(2, 70.0),
            Scored::new(5, 60.0),
        ];

        for (index, session_score) in top_k.closest_neighbors.into_sorted_vec().iter().enumerate() {
            assert_eq!(*session_score, expected_result[index]);
        }
    }
}
