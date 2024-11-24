use crate::sessrec::metrics::mrr::Mrr;
use crate::sessrec::metrics::product_info::ProductInfo;
use crate::sessrec::metrics::sustainabilityratio::SustainabilityCoverage;
use crate::sessrec::metrics::Metric;
use crate::sessrec::vmisknn::Scored;

#[derive(Debug, Clone)]
pub struct ResponsibleMrr<'a> {
    mrr: Mrr,
    sustainability_coverage: SustainabilityCoverage<'a>,
    mrr_alpha: f64,
    length: usize,
}

impl<'a> ResponsibleMrr<'a> {
    pub fn new(product_info: &'a ProductInfo, mrr_alpha: f64, length: usize) -> Self {
        ResponsibleMrr {
            mrr: Mrr::new(length),
            sustainability_coverage: SustainabilityCoverage::new(product_info, length),
            mrr_alpha,
            length,
        }
    }
}

impl Metric for ResponsibleMrr<'_> {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) {
        self.mrr.add(recommendations, next_items);
        self.sustainability_coverage
            .add(recommendations, next_items);
    }

    fn result(&self) -> f64 {
        let mrr_weighted = self.mrr.result() * self.mrr_alpha;
        let sustainable_coverage = self.sustainability_coverage.result() * (1.0 - self.mrr_alpha);
        mrr_weighted + sustainable_coverage
    }

    fn get_name(&self) -> String {
        format!("ResponsibleMrr@{}", self.length)
    }

    fn reset(&mut self) {
        self.mrr.reset();
        self.sustainability_coverage.reset()
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        let mrr_weighted_score = self.mrr.compute(recommendations, next_items) * self.mrr_alpha;
        let sustainability_coverage_weighted_score = self
            .sustainability_coverage
            .compute(recommendations, next_items)
            * (1.0 - self.mrr_alpha);
        mrr_weighted_score + sustainability_coverage_weighted_score
    }
}

#[cfg(test)]
mod responsible_mrr_test {
    use super::*;
    use crate::sessrec::types::ItemId;
    use std::collections::HashSet;
    use itertools::Itertools;

    #[test]
    fn should_happyflow_responsible_mrr() {
        let sustainable_products: HashSet<ItemId> = vec![5, 10, 15].into_iter().collect();
        let product_info = ProductInfo::new(sustainable_products);

        let length = 20;

        let recommendations: Vec<Scored> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let actual_next_items: Vec<Scored> = vec![3, 55, 3, 4].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();

        let mrr_alpha = 0.8;
        let mut sustain_mrr = ResponsibleMrr::new(&product_info, mrr_alpha, length);
        sustain_mrr.add(&recommendations, &actual_next_items);

        assert_eq!(format!("ResponsibleMrr@{length}"), sustain_mrr.get_name());
        assert!((0.29666666666666663 - sustain_mrr.result()).abs() < f64::EPSILON);
    }

    #[test]
    fn handle_empty_recommendations() {
        let sustainable_products: HashSet<ItemId> = vec![5, 10, 15].into_iter().collect();
        let product_info = ProductInfo::new(sustainable_products);

        let length = 20;

        let mrr_alpha = 0.8;
        let mymetric = ResponsibleMrr::new(&product_info, mrr_alpha, length);
        let recommendations: Vec<Scored> = vec![];
        let actual_next_items: Vec<Scored> = vec![1, 2].iter()
            .map(|&id| Scored::new(id, 1.0))
            .collect_vec();
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.0, result);
        assert_eq!(mymetric.result(), result);
    }
}
