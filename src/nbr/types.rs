use crate::importance::{Dataset, DatasetEntry, Sequence};
use crate::nbr::tifuknn::types::{Basket, UserId};
use crate::sessrec::vmisknn::Scored;
use itertools::Itertools;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use csv::Writer;
use grouping_by::GroupingBy;
use crate::nbr::tifuknn::io::Purchase;

#[derive(Clone, Debug)]
pub struct NextBasketDataset {
    pub user_baskets: HashMap<u32, Vec<Basket>>,
}

impl NextBasketDataset {
    pub fn to_csv(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        log::info!("NextBasketDataset writing to {} ", file_path);
        // Open the file for writing
        let file = File::create(file_path)?;
        let mut wtr = Writer::from_writer(file);

        // Write the header
        wtr.write_record(&["user_id", "basket_id", "items"])?;

        // Write the data from user_baskets
        for (user_id, baskets) in &self.user_baskets {
            for basket in baskets {
                let items_str = basket.items.iter().map(|item| item.to_string()).collect::<Vec<String>>().join(", ");
                wtr.write_record(&[user_id.to_string(), basket.id.to_string(), items_str])?;
            }
        }

        // Flush the contents of the internal buffer to the underlying writer.
        wtr.flush()?;

        Ok(())
    }
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

impl From<&Vec<Purchase>> for NextBasketDataset {
    fn from(purchases: &Vec<Purchase>) -> Self {
        let baskets_by_user: HashMap<UserId, Vec<Basket>> = purchases
            .into_iter()
            .grouping_by(|p| p.user)
            .into_iter()
            .map(|(user, user_purchases)| {
                let mut baskets: Vec<Basket> = user_purchases
                    .into_iter()
                    .grouping_by(|p| p.basket)
                    .into_iter()
                    .map(|(basket_id, basket_purchases)| {
                        let items = basket_purchases.into_iter().map(|p| p.item).collect();
                        Basket::new(basket_id, items)
                    })
                    .collect();

                baskets.sort_by_key(|b| b.id);

                (user, baskets)
            })
            .collect();

        NextBasketDataset {
            user_baskets: baskets_by_user,
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
        // assert_eq!(
        //     baskets.len(),
        //     1_usize,
        //     "next basket recommendations evaluate on only one basket, but found {} baskets for key {:?}: {:?}",
        //     baskets.len(),
        //     key,
        //     baskets
        // );

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
