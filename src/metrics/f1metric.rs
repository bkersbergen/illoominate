use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;
use crate::metrics::Metric;
use crate::sessrec::vmisknn::Scored;

#[derive(Debug, Clone)]
pub struct F1score {
    precision: Precision,
    recall: Recall,
    beta_squared: f64,
    length: usize,
}

impl F1score {
    // measures the effectiveness of retrieval with respect to a user who attaches
    // beta times as much importance to recall as precision. https://en.wikipedia.org/wiki/F-score
    pub fn new(length: usize) -> Self {
        let beta = 1.0; // make recall `beta` times more important than precision
        F1score {
            precision: Precision::new(length),
            recall: Recall::new(length),
            beta_squared: beta * beta,
            length,
        }
    }
    pub fn with_beta(length: usize, beta: f64) -> Self {
        F1score {
            precision: Precision::new(length),
            recall: Recall::new(length),
            beta_squared: beta * beta,
            length,
        }
    }
}

impl Metric for F1score {
    fn add(&mut self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) {
        self.precision.add(recommendations, next_items);
        self.recall.add(recommendations, next_items);
    }

    fn result(&self) -> f64 {
        let precision_score = self.precision.result();
        let recall_score = self.recall.result();
        let f1score = if precision_score.abs() < f64::EPSILON || recall_score.abs() < f64::EPSILON {
            0.0
        } else {
            (1.0 + self.beta_squared) * (precision_score * recall_score)
                / (self.beta_squared * precision_score + recall_score)
        };
        if f1score.is_nan() {
            0.0
        } else {
            f1score
        }
    }

    fn get_name(&self) -> String {
        format!("F1score@{}", self.length)
    }

    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }

    fn compute(&self, recommendations: &Vec<Scored>, next_items: &Vec<Scored>) -> f64 {
        let precision_score = self.precision.compute(recommendations, next_items);
        let recall_score = self.recall.compute(recommendations, next_items);
        let f1score = if precision_score.abs() < f64::EPSILON || recall_score.abs() < f64::EPSILON {
            0.0
        } else {
            (1.0 + self.beta_squared) * (precision_score * recall_score)
                / (self.beta_squared * precision_score + recall_score)
        };
        if f1score.is_nan() {
            0.0
        } else {
            f1score
        }
    }
}

#[cfg(test)]
mod f1score_test {
    use super::*;

    #[test]
    fn should_happyflow_f1score() {
        let mut undertest = F1score::new(20);
        let recommendations: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let actual_next_items: Vec<Scored> = vec![Scored::new(2_u32, 1.0), Scored::new(3_u32, 1.0)];
        undertest.add(&recommendations, &actual_next_items);
        assert!((0.09090909090909091 - undertest.result()).abs() < f64::EPSILON);
        assert_eq!("F1score@20", undertest.get_name());
    }

    #[test]
    fn should_emphasis_precision_f1score() {
        let mut undertest = F1score::with_beta(20, 0.5);
        let recommendations: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let actual_next_items: Vec<Scored> = vec![Scored::new(2_u32, 1.0), Scored::new(3_u32, 1.0)];
        undertest.add(&recommendations, &actual_next_items);
        assert!((0.06097560975609757 - undertest.result()).abs() < f64::EPSILON);
        assert_eq!("F1score@20", undertest.get_name());
    }

    #[test]
    fn should_emphasis_recall_f1score() {
        let mut undertest = F1score::with_beta(20, 2.0);
        let recommendations: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let actual_next_items: Vec<Scored> = vec![Scored::new(2_u32, 1.0), Scored::new(3_u32, 1.0)];
        undertest.add(&recommendations, &actual_next_items);
        assert!((0.17857142857142858 - undertest.result()).abs() < f64::EPSILON);
        assert_eq!("F1score@20", undertest.get_name());
    }
    #[test]
    fn should_handle_divide_by_zero() {
        let undertest = F1score::new(20);
        assert!((0.0 - undertest.result()).abs() < f64::EPSILON);
    }
    #[test]
    fn handle_empty_recommendations() {
        let mymetric = F1score::new(20);
        let recommendations: Vec<Scored> = vec![];
        let actual_next_items: Vec<Scored> = vec![Scored::new(1_u32, 1.0), Scored::new(2_u32, 1.0)];
        let result = mymetric.compute(&recommendations, &actual_next_items);
        assert_eq!(0.0, result);
        assert_eq!(mymetric.result(), result);
    }
}
