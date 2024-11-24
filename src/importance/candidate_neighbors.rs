use crate::sessrec::vmisknn::Scored;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;

pub struct CandidateNeighbors {
    closest_neighbors: BinaryHeap<Scored>,
    k: usize,
}

impl CandidateNeighbors {
    pub fn new(k: usize) -> Self {
        Self {
            closest_neighbors: BinaryHeap::with_capacity(k),
            k,
        }
    }

    /// Offers a `SessionScore` to the TopK
    ///
    /// If the number of neighbors is below `k`, the new session is added. If there are already `k` neighbors,
    /// it replaces the neighbor with the lowest score if the new session's score is higher.
    ///
    /// # Arguments
    ///
    /// * `scored_session` - The `SessionScore` to potentially add or replace in the collection.
    ///
    /// # Returns
    ///
    /// `true` if the session was added or replaced; `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use illoominate::importance::candidate_neighbors::CandidateNeighbors;
    /// use illoominate::sessrec::vmisknn::Scored;
    /// let mut manager = CandidateNeighbors::new(25);
    /// let session = Scored { id: 0, score: 75.0};
    /// let added = manager.offer(session);
    /// assert!(added);
    /// ```
    pub fn offer(&mut self, scored_session: Scored) -> (bool, Option<Scored>) {
        if self.closest_neighbors.len() < self.k {
            self.closest_neighbors.push(scored_session);
            return (true, None);
        } else {
            let mut bottom = self.closest_neighbors.peek_mut().unwrap();
            if scored_session.score > bottom.score {
                let dropped_out = *bottom;
                *bottom = scored_session;
                return (true, Some(dropped_out));
            }
        }
        (false, None)
    }

    /// Returns an iterator visiting all values in the topk, in
    /// arbitrary order.
    pub fn iter(&self) -> Iter<'_, Scored> {
        self.closest_neighbors.iter()
    }
}
