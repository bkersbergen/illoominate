use std::cmp;
use std::collections::HashSet;

use crate::sessrec::metrics::Metric;
use crate::sessrec::vmisknn::Scored;

#[derive(Debug, Clone)]
pub struct Recall {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Recall {
    /// Returns a Recall evaluation metric.
    /// Recall quantifies the number of positive recommendations made out
    /// of all interacted items.
    ///
    /// # Arguments
    ///
    /// * `length` - the length aka 'k' that will be used for evaluation.
    ///

    pub fn new(length: usize) -> Self {
        Recall {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl Metric for Recall {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) {
        self.qty += 1;

        self.sum_of_scores += self.compute(recommendations, next_items);
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("Recall@{}", self.length)
    }

    fn reset(&mut self) {
        self.sum_of_scores = 0.0;
        self.qty = 0;
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        if recommendations.is_empty() || next_items.is_empty() {
            return 0.0;
        }
        let top_recos: HashSet<_> = recommendations
            .iter()
            .take(cmp::min(recommendations.len(), self.length))
            .collect();

        let unique_next_items: HashSet<_> = next_items.iter().collect();

        let intersection_count = top_recos.intersection(&unique_next_items).count();

        //intersection_count as f64 / next_items.len() as f64

        intersection_count as f64 / unique_next_items.len() as f64
    }
}

#[cfg(test)]
mod recall_test {
    use itertools::Itertools;
    use super::*;

    #[test]
    fn should_calculate_recall() {
        let length = 20;
        let mut under_test = Recall::new(length);
        let recommendations: Vec<Scored> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let actual_next_items: Vec<Scored> = vec![3, 55, 3, 4].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        under_test.add(&recommendations, &actual_next_items);
        assert!((2.0/3.0 - under_test.result()).abs() < f64::EPSILON);
        assert_eq!("Recall@20", under_test.get_name());
    }

    #[test]
    fn handle_empty_recommendations() {
        let mymetric = Recall::new(20);
        let recommendations: Vec<Scored> = vec![];
        let actual_next_items: Vec<Scored> = vec![1, 2].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.0, result);
        assert_eq!(mymetric.result(), result);
    }
}
