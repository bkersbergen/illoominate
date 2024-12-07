use std::error::Error;

use crate::metrics::{MetricConfig, MetricType};
use config::{Config, File};
use log::LevelFilter;
use serde::Deserialize;

// Define a struct for general hyperparameters, which vary by model
#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct HpoConfig {
    pub k: Option<usize>,
    pub m: Option<usize>,
    pub r_b: Option<f64>,
    pub r_g: Option<f64>,
    pub alpha: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct Metric {
    pub name: String,
    pub length: usize,
}

#[derive(Debug, Deserialize)]
pub struct Nbr {
    pub validation_ratio: Option<f64>,
}

// Define a struct for the entire config structure
#[derive(Debug, Deserialize)]
pub struct IlloominateConfig {
    pub model: ModelConfig,
    pub hpo: HpoConfig,
    pub metric: Metric,
    pub nbr: Option<Nbr>,
    pub logging: Option<LoggingConfig>,
}

// Define a struct for logging configuration
#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    level: Option<String>,
}
impl IlloominateConfig {
    // Function to load the configuration from a file
    pub fn load(config_file_basename: &str) -> Result<IlloominateConfig, Box<dyn Error>> {
        // Build the configuration from the file
        let config = Config::builder()
            .add_source(File::with_name(config_file_basename))
            .build()?;

        // Deserialize the configuration into the IlloominateConfig struct
        let app_config: IlloominateConfig = config.try_deserialize()?;

        // Set the logging level based on the configuration, default to 'info' if not specified
        let log_level = app_config
            .logging
            .as_ref()
            .and_then(|logging| logging.level.clone())
            .unwrap_or_else(|| "info".to_string());

        let level: LevelFilter = match log_level.to_lowercase().as_str() {
            "trace" => LevelFilter::Trace,
            "debug" => LevelFilter::Debug,
            "info" => LevelFilter::Info,
            "warn" => LevelFilter::Warn,
            "error" => LevelFilter::Error,
            _ => LevelFilter::Info, // Default to `info` if unrecognized
        };

        log::set_max_level(level);
        Ok(app_config)
    }
}

pub fn create_metric_config(app_config: &IlloominateConfig) -> MetricConfig {
    let metric_type = match app_config.metric.name.trim().to_lowercase().as_str() {
        "f1score" => MetricType::F1score,
        "hitrate" => MetricType::HitRate,
        "mrr" => MetricType::MRR,
        "precision" => MetricType::Precision,
        "recall" => MetricType::Recall,
        "sustainablemrr" => MetricType::SustainableMrr,
        "sustainabilitycoverage" => MetricType::SustainabilityCoverageTerm,
        "ndcg" => MetricType::Ndcg,
        invalid => panic!("Invalid metric type: {}", invalid), // Include invalid value in panic message
    };

    
    MetricConfig {
        importance_metric: metric_type.clone(),
        evaluation_metrics: vec![metric_type.clone()],
        length: app_config.metric.length,
        alpha: 0.8,
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use super::*;

    fn create_test_config(file_path: &str, content: &str) -> std::io::Result<()> {
        let mut file = File::create(file_path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    #[test]
    fn test_config_loading() {
        // Create a temporary config file for testing
        let test_config_file = "test_config.toml";
        let config_content = r#"
        [model]
        name = "test_model"

        [hpo]
        k = 5
        m = 10
        r_b = 0.2
        r_g = 0.4
        alpha = 0.01

        [metric]
        name="MRR"
        length=21

        [nbr]
        validation_ratio=0.8
        "#;
        create_test_config(test_config_file, config_content)
            .expect("Failed to create test config file");

        // Attempt to load the config
        let config = IlloominateConfig::load(test_config_file).expect("Failed to load config");

        // Check if the values were loaded correctly
        assert_eq!(config.model.name, "test_model");
        assert_eq!(config.hpo.k, Some(5));
        assert_eq!(config.hpo.m, Some(10));
        assert_eq!(config.hpo.r_b, Some(0.2));
        assert_eq!(config.hpo.r_g, Some(0.4));
        assert_eq!(config.hpo.alpha, Some(0.01));
        assert_eq!(config.nbr.unwrap().validation_ratio, Some(0.8));

        // Clean up test config file
        std::fs::remove_file(test_config_file).expect("Failed to delete test config file");
    }

    #[test]
    fn test_config_with_missing_optional_values() {
        let test_config_file = "test_config_missing_optional.toml";
        let config_content = r#"
        [model]
        name = "test_model"

        [hpo]
        k = 5

        [metric]
        name="MRR"
        length=21
        "#;
        create_test_config(test_config_file, config_content)
            .expect("Failed to create test config file");

        let config = IlloominateConfig::load(test_config_file).expect("Failed to load config");

        assert_eq!(config.model.name, "test_model");
        assert_eq!(config.hpo.k, Some(5));
        assert_eq!(config.hpo.m, None);
        assert_eq!(config.hpo.r_b, None);
        assert_eq!(config.hpo.r_g, None);
        assert_eq!(config.hpo.alpha, None);

        std::fs::remove_file(test_config_file).expect("Failed to delete test config file");
    }

    #[test]
    fn test_config_loading_with_optional_logging_debug() {
        let test_config_file = "test_config_with_logging.toml";
        let config_content = r#"
        [model]
        name = "test_model"

        [hpo]
        k = 5
        m = 10
        r_b = 0.2
        r_g = 0.4
        alpha = 0.01

        [metric]
        name="MRR"
        length=21

        [logging]
        level = "debug"
        "#;
        create_test_config(test_config_file, config_content)
            .expect("Failed to create test config file");

        let config = IlloominateConfig::load(test_config_file).expect("Failed to load config");

        assert_eq!(log::max_level(), LevelFilter::Debug);

        std::fs::remove_file(test_config_file).expect("Failed to delete test config file");
    }

    #[test]
    fn test_config_loading_with_metric_type() {
        let test_config_file = "test_config_with_metric_type.toml";
        let config_content = r#"
        [model]
        name = "test_model"

        [hpo]
        k = 5
        m = 10
        r_b = 0.2
        r_g = 0.4
        alpha = 0.01

        [metric]
        name="MRR"
        length=33

        [logging]
        level = "debug"
        "#;
        create_test_config(test_config_file, config_content)
            .expect("Failed to create test config file");

        let config = IlloominateConfig::load(test_config_file).expect("Failed to load config");

        assert_eq!(config.metric.name, "MRR");
        assert_eq!(config.metric.length, 33);
        assert_eq!(log::max_level(), LevelFilter::Debug);

        std::fs::remove_file(test_config_file).expect("Failed to delete test config file");
    }
}
