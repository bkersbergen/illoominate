use crate::metrics::f1metric::F1score;
use crate::metrics::hitrate::HitRate;
use crate::metrics::mrr::Mrr;
use crate::metrics::ndcg::Ndcg;
use crate::metrics::precision::Precision;
use crate::metrics::product_info::ProductInfo;
use crate::metrics::recall::Recall;
use crate::metrics::sustainable_mrr::SustainableMrr;
use serde::{Deserialize, Serialize};
// use crate::sessrec::metrics::responsible_mrr::ResponsibleMrr;
use crate::metrics::st::St;
use crate::metrics::sustainable_ndcg::SustainableNdcg;
use crate::sessrec::vmisknn::Scored;

pub mod f1metric;
pub mod hitrate;
pub mod mrr;
pub mod ndcg;
pub mod precision;
pub mod product_info;
pub mod recall;
pub mod st;
pub mod sustainable_mrr;
pub mod sustainable_ndcg;

pub trait Metric {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>);
    fn result(&self) -> f64;
    fn get_name(&self) -> String;
    fn reset(&mut self);
    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64;
}

// Configuration and factory
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub enum MetricType {
    F1score,
    HitRate,
    MRR,
    Precision,
    Recall,
    SustainableMrr,
    SustainableNdcg,
    SustainabilityCoverageTerm,
    Ndcg,
}

#[derive(Clone, Debug)]
pub struct MetricConfig {
    pub importance_metric: MetricType,
    pub evaluation_metrics: Vec<MetricType>,
    pub length: usize,
    pub alpha: f64,
}

#[derive(Clone, Debug)]
pub struct MetricFactory<'a> {
    pub config: &'a MetricConfig,
    product_info: ProductInfo,
}

impl<'a> MetricFactory<'a> {
    pub fn new(config: &'a MetricConfig, product_info: ProductInfo) -> Self {
        MetricFactory {
            config,
            product_info,
        }
    }

    pub fn create_importance_metric(&self) -> Box<dyn Metric + Send + Sync + '_> {
        self.create_metric(&self.config.importance_metric)
    }

    pub fn create_evaluation_metrics(&self) -> Vec<Box<dyn Metric + Send + Sync + '_>> {
        self.config
            .evaluation_metrics
            .iter()
            .map(|metric| self.create_metric(metric))
            .collect()
    }

    fn create_metric(&self, metric_type: &MetricType) -> Box<dyn Metric + Send + Sync + '_> {
        match metric_type {
            MetricType::F1score => Box::new(F1score::new(self.config.length)),
            MetricType::HitRate => Box::new(HitRate::new(self.config.length)),
            MetricType::MRR => Box::new(Mrr::new(self.config.length)),
            MetricType::Precision => Box::new(Precision::new(self.config.length)),
            MetricType::Recall => Box::new(Recall::new(self.config.length)),
            MetricType::SustainableMrr => Box::new(SustainableMrr::new(
                &self.product_info,
                self.config.alpha,
                self.config.length,
            )),
            MetricType::SustainableNdcg => Box::new(SustainableNdcg::new(
                &self.product_info,
                self.config.alpha,
                self.config.length,
            )),
            MetricType::SustainabilityCoverageTerm => Box::new(St::new(
                &self.product_info,
                self.config.length,
            )),
            MetricType::Ndcg => Box::new(Ndcg::new(self.config.length)),
        }
    }
}
