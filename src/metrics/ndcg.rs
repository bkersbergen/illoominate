use crate::metrics::Metric;
use crate::sessrec::types::ItemId;
use crate::sessrec::vmisknn::Scored;
use std::cmp;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Ndcg {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Ndcg {
    fn dcg(&self, top_recos: &[u64], next_items: &[u64]) -> f64 {
        let mut result = 0_f64;
        let next_items_set: HashSet<&u64> = next_items.iter().collect::<HashSet<_>>();
        for (index, _item_id) in top_recos.iter().enumerate() {
            if next_items_set.contains(&top_recos[index]) {
                if index == 0 {
                    result += 1_f64;
                } else {
                    result += 1_f64 / ((index as f64) + 1_f64).log2();
                }
            }
        }
        result
    }
}

impl Ndcg {
    //
    /// Calculate Ndcg for predicted recommendations and the given next items that will be interacted with.
    pub fn new(length: usize) -> Ndcg {
        Ndcg {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl Metric for Ndcg {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) {
        self.sum_of_scores += self.compute(recommendations, next_items);
        self.qty += 1;
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Ndcg@{}", self.length)
    }

    fn reset(&mut self) {
        self.sum_of_scores = 0.0;
        self.qty = 0;
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        let top_recos: Vec<ItemId> = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .map(|id_score| id_score.id as ItemId)
            .collect(); // Collect into a Vec<ItemId>

        let top_next_items: Vec<ItemId> = next_items
            .iter()
            .take(cmp::min(next_items.len(), self.length))
            .map(|id_score| id_score.id as ItemId)
            .collect(); // Collect into a Vec<ItemId>

        let next_items: Vec<ItemId> = next_items
            .iter()
            .map(|id_score| id_score.id as ItemId)
            .collect(); // Collect into a Vec<ItemId>

        let dcg: f64 = self.dcg(&top_recos, &next_items);
        let dcg_max: f64 = self.dcg(&top_next_items, &next_items);
        
        dcg / dcg_max
    }
}

#[cfg(test)]
mod ndcg_test {
    use itertools::Itertools;
    use super::*;

    #[test]
    fn should_calculate_ndcg() {
        let mut mymetric = Ndcg::new(20);
        let recommendations: Vec<Scored> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]
        .iter()
        .map(|&id| Scored::new(id, 1.0))
        .collect_vec();
        let actual_next_items: Vec<Scored> = vec![3, 55, 88, 4]
            .iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        mymetric.add(&recommendations, &actual_next_items);
        assert_eq!(0.36121211352040195, mymetric.result());
        assert_eq!("Ndcg@20", mymetric.get_name());
    }

    #[test]
    fn should_compute() {
        let mymetric = Ndcg::new(20);
        let recommendations: Vec<Scored> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let actual_next_items: Vec<Scored> = vec![3, 55, 88, 4].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.36121211352040195, result);
    }
}
