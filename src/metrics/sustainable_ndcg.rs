use crate::metrics::ndcg::Ndcg;
use crate::metrics::product_info::ProductInfo;
use crate::metrics::st::St;
use crate::metrics::Metric;
use crate::sessrec::vmisknn::Scored;


#[derive(Debug, Clone)]
pub struct SustainableNdcg<'a> {
    ndcg: Ndcg,
    sustainability_coverage_term: St<'a>,
    alpha: f64,
    length: usize,
}

impl<'a> SustainableNdcg<'a> {
    pub fn new(product_info: &'a ProductInfo, alpha: f64, length: usize) -> Self {
        SustainableNdcg {
            ndcg: Ndcg::new(length),
            sustainability_coverage_term: St::new(product_info, length),
            alpha,
            length,
        }
    }
}

impl Metric for SustainableNdcg<'_> {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) {
        self.ndcg.add(recommendations, next_items);
        self.sustainability_coverage_term
            .add(recommendations, next_items);
    }

    fn result(&self) -> f64 {
        let mrr_weighted = self.ndcg.result() * self.alpha;
        let sustainable_coverage = self.sustainability_coverage_term.result() * (1.0 - self.alpha);
        mrr_weighted + sustainable_coverage
    }

    fn get_name(&self) -> String {
        format!("SustainableNdcg@{}", self.length)
    }

    fn reset(&mut self) {
        self.ndcg.reset();
        self.sustainability_coverage_term.reset()
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        let mrr_weighted_score = self.ndcg.compute(recommendations, next_items) * self.alpha;
        let sustainability_coverage_weighted_score = self
            .sustainability_coverage_term
            .compute(recommendations, next_items)
            * (1.0 - self.alpha);
        mrr_weighted_score + sustainability_coverage_weighted_score
    }
}
