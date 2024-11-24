use crate::importance::{Dataset, DatasetEntry, Sequence};
use crate::nbr::tifuknn::types::{Basket, UserId};
use crate::sessrec::vmisknn::Scored;
use itertools::Itertools;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct NextBasketDataset {
    pub user_baskets: HashMap<u32, Vec<Basket>>,
}

impl From<&HashMap<u32, Vec<Basket>>> for NextBasketDataset {
    fn from(order_history: &HashMap<UserId, Vec<Basket>>) -> Self {
        NextBasketDataset {
            user_baskets: order_history
                .iter()
                .map(|(&user_id, baskets)| (user_id, baskets.clone())) // Convert usize to u32 and clone baskets
                .collect::<HashMap<u32, Vec<Basket>>>(),
        }
    }
}

impl Dataset for NextBasketDataset {
    fn collect_keys(&self) -> Vec<u32> {
        self.user_baskets.keys().cloned().collect_vec()
    }

    fn num_interactions(&self) -> usize {
        self.user_baskets.values().map(|baskets| baskets.len())
            .sum()
    }

    fn __get_entry__(&self, key: u32) -> DatasetEntry {
        let baskets = self.user_baskets.get(&key).unwrap();
        assert_eq!(
            baskets.len(),
            1_usize,
            "next basket recommendations evaluate on only one basket"
        );

        let input_sequence = vec![Scored::new(key, 1.0)];
        let target_sequence = baskets.first()
            .unwrap()
            .items
            .iter()
            .map(|&id| Scored::new(id as u32, 1.0))
            .collect_vec();
        let sequence = Sequence {
            input: input_sequence,
            target: target_sequence,
        };
        DatasetEntry {
            key,
            sequences: vec![sequence],
            max_timestamp: 0,
        }
    }

    fn __get_items__(&self, key: u32) -> Vec<u32> {
        if let Some(baskets) = self.user_baskets.get(&key) {
            let all_heldout_values: Vec<_> = baskets
                .iter()
                .flat_map(|basket| basket.items.iter())
                .sorted()
                .map(|&id| id as u32)
                .collect();
            all_heldout_values
        } else {
            Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.user_baskets.len()
    }
}
