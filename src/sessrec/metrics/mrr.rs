use crate::sessrec::metrics::Metric;
use crate::sessrec::vmisknn::Scored;
use std::cmp;

#[derive(Debug, Clone)]
pub struct Mrr {
    sum_of_scores: f64,
    qty: usize,
    length: usize,
}

impl Mrr {
    pub fn new(length: usize) -> Self {
        Mrr {
            sum_of_scores: 0_f64,
            qty: 0,
            length,
        }
    }
}
impl Metric for Mrr {
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
        format!("Mrr@{}", self.length)
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
        if let Some(rank) = index {
            1_f64 / (rank as f64 + 1_f64)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod mrr_test {
    use super::*;

    #[test]
    fn should_calculate_mrr() {
        let mut mymetric = Mrr::new(20);
        let recommendations: Vec<Scored> = vec![
            Scored::new(1_u32,1.0),Scored::new(2_u32,1.0),Scored::new(3_u32,1.0),Scored::new(4_u32,1.0),
            Scored::new(5_u32,1.0),Scored::new(6_u32,1.0),Scored::new(7_u32,1.0),Scored::new(8_u32,1.0),
            Scored::new(9_u32,1.0),Scored::new(10_u32,1.0),Scored::new(11_u32,1.0),Scored::new(12_u32,1.0),
            Scored::new(13_u32,1.0),Scored::new(14_u32,1.0),Scored::new(15_u32,1.0),Scored::new(16_u32,1.0),
            Scored::new(17_u32,1.0),Scored::new(18_u32,1.0),Scored::new(19_u32,1.0),Scored::new(20_u32,1.0),
            Scored::new(21_u32,1.0),Scored::new(22_u32,1.0),Scored::new(23_u32,1.0),Scored::new(24_u32,1.0),
        ];
        let actual_next_items: Vec<Scored> = vec![Scored::new(3_u32,1.0), Scored::new(55_u32, 1.0), Scored::new(3_u32,1.0), Scored::new(4_u32, 1.0)];
        mymetric.add(&recommendations, &actual_next_items);
        assert_eq!(0.3333333333333333, mymetric.result());
        assert_eq!("Mrr@20", mymetric.get_name());
    }

    #[test]
    fn handle_empty_recommendations() {
        let mymetric = Mrr::new(20);
        let recommendations: Vec<Scored> = vec![];
        let actual_next_items: Vec<Scored> = vec![Scored::new(1_u32,1.0),Scored::new(2_u32,1.0)];
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.0, result);
    }
}
