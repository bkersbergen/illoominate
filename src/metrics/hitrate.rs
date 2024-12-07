use std::cmp;

use crate::metrics::Metric;
use crate::sessrec::vmisknn::Scored;

#[derive(Debug, Clone)]
pub struct HitRate {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl HitRate {
    pub fn new(length: usize) -> Self {
        HitRate {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}

impl Metric for HitRate {
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
        format!("HitRate@{}", self.length)
    }

    fn reset(&mut self) {
        self.sum_of_scores = 0.0;
        self.qty = 0;
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        let num_recos = cmp::min(recommendations.len(), self.length);
        let top_recos = &recommendations[0..num_recos];

        let next_item = next_items[0];
        let index = top_recos.iter().position(|&item_id| item_id == next_item);
        if let Some(_rank) = index {
            1_f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod hitrate_test {
    use super::*;

    #[test]
    fn should_happyflow_hitrate() {
        let mut undertest = HitRate::new(20);
        let recommendations: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let actual_next_items: Vec<Scored> = vec![Scored::new(2_u32, 1.0), Scored::new(3_u32, 1.0)];
        undertest.add(&recommendations, &actual_next_items);
        assert_eq!("HitRate@20", undertest.get_name());
        assert!((1.0 - undertest.result()).abs() < f64::EPSILON);
    }

    #[test]
    fn should_handle_divide_by_zero() {
        let undertest = HitRate::new(20);
        assert!((0.0 - undertest.result()).abs() < f64::EPSILON);
    }
    #[test]
    fn handle_empty_recommendations() {
        let mymetric = HitRate::new(20);
        let recommendations: Vec<Scored> = vec![];
        let actual_next_items: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.0, result);
        assert_eq!(mymetric.result(), result);
    }
}
