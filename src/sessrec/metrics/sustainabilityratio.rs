use crate::sessrec::metrics::product_info::ProductInfo;
use crate::sessrec::metrics::Metric;
use crate::sessrec::types::ItemId;
use crate::sessrec::vmisknn::Scored;
use std::cmp;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SustainabilityCoverage<'a> {
    sum_of_scores: f64,
    qty: usize,
    product_info: &'a ProductInfo,
    length: usize,
}

impl<'a> SustainabilityCoverage<'a> {
    pub fn new(product_info: &'a ProductInfo, length: usize) -> Self {
        SustainabilityCoverage {
            sum_of_scores: 0_f64,
            qty: 0,
            product_info,
            length,
        }
    }
}

impl<'a> Metric for SustainabilityCoverage<'a> {
    fn add(&mut self, recommendations: &Vec<Scored>, _next_items: &Vec<Scored>) {
        self.qty += 1;

        self.sum_of_scores += self.compute(recommendations, _next_items);
    }

    fn result(&self) -> f64 {
        if self.qty > 0 {
            self.sum_of_scores / self.qty as f64
        } else {
            0.0
        }
    }

    fn get_name(&self) -> String {
        format!("SustainabilityCoverage@{}", self.length)
    }

    fn reset(&mut self) {
        self.sum_of_scores = 0.0;
        self.qty = 0;
    }

    fn compute(&self, recommendations: &Vec<Scored>, _next_items: &Vec<Scored>) -> f64 {
        if recommendations.is_empty() {
            return 0.0;
        }
        let n = cmp::min(recommendations.len(), self.length);

        let sustainable_recos: HashSet<_> = recommendations
            .iter()
            .take(n)
            .map(|&id_score| id_score.id as ItemId)
            .collect();

        let num_sustainable_recos = sustainable_recos
            .intersection(&self.product_info.sustainable_products)
            .count();

        num_sustainable_recos as f64 / n as f64
    }
}

#[cfg(test)]
mod sustainabilitycoverage_test {
    use itertools::Itertools;
    use super::*;
    use crate::sessrec::types::ItemId;

    #[test]
    fn should_calculate_sustainabilitycoverage() {
        let sustainable_products: HashSet<ItemId> = vec![5, 10, 15].into_iter().collect();
        let qty_sustainable_items_also_in_recommendations = sustainable_products.len();
        let product_info = ProductInfo::new(sustainable_products);

        let length = 20;
        let mut under_test = SustainabilityCoverage::new(&product_info, length);
        let recommendations: Vec<Scored> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let _next_items: Vec<Scored> = vec![3, 55, 3, 4].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        under_test.add(&recommendations, &_next_items);
        let expected = qty_sustainable_items_also_in_recommendations as f64 / length as f64;
        assert!((expected - under_test.result()).abs() < f64::EPSILON);
        assert_eq!("SustainabilityCoverage@20", under_test.get_name());
    }
    #[test]
    fn handle_empty_recommendations() {
        let sustainable_products: HashSet<ItemId> = vec![5, 10, 15].into_iter().collect();
        let product_info = ProductInfo::new(sustainable_products);
        let length = 21;
        let mut under_test = SustainabilityCoverage::new(&product_info, length);
        let recommendations: Vec<Scored> = vec![];
        let _next_items: Vec<Scored> = vec![1, 2, 3].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        under_test.add(&recommendations, &_next_items);

        let expected_qty_evaluations = 1;
        assert_eq!(expected_qty_evaluations, under_test.qty);

        let expected_score = 0.0;
        assert!((expected_score - under_test.result()).abs() < f64::EPSILON);
    }
}
