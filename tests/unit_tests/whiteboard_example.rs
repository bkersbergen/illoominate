//
// #[cfg(test)]
// mod localtmcshapley_test {
//     use std::collections::HashSet;
//     use std::env;
//     use illoominate::importance::{localtmcshapley, localtmcshapley_opt};
//     use illoominate::sessrec::io::read_sustainable_products_info;
//     use illoominate::sessrec::metrics::{MetricConfig, MetricFactory, MetricType};
//     use illoominate::sessrec::metrics::product_info::ProductInfo;
//     use illoominate::sessrec::types::{Interaction, ItemId};
//     use illoominate::sessrec::types::SessionDataset;
//     use illoominate::sessrec::vmisknn::VMISKNN;
//     use crate::unit_tests::whiteboard_example::assert_maps_equal;
//
//     #[test]
//     fn whiteboard_example() {
//         env::set_var("RUST_LOG", "debug");
//         env_logger::init();
//
//         let config = MetricConfig {
//             importance_metric: MetricType::MRR,
//             evaluation_metrics: vec![MetricType::MRR],
//             length: 21,
//             mrr_alpha: 0.8,
//         };
//         let product_info = ProductInfo::new(HashSet::new());
//         let metric_factory = MetricFactory::new(&config, &product_info);
//
//         let train = create_training_dataset();
//         let valid = create_validation_dataset();
//         let tolerance = 0.01;
//         let mean_score = 0.2;
//         let random_score = 0.006;
//         let k = 500;
//         let m = 500;
//         let seed = 42;
//         let iteration = 0;
//
//         let model = VMISKNN::fit_dataset(&train, m, k, 1.0);
//
//         let expected = illoominate::baselines::tmcshapley::one_iteration_dataset(
//             &train, &valid, tolerance, mean_score, random_score, k, m, seed, iteration);
//         let actual_noopt = localtmcshapley::one_iteration_dataset(
//             &model, &train, &valid, tolerance, mean_score, random_score, seed, iteration);
//         let actual_opt = localtmcshapley_opt::one_iteration_dataset(
//             &model, &train, &valid, tolerance, mean_score, random_score, seed, iteration);
//
//         log::info!("Expected Importances:\t{:?}", as_pretty_string_from(&expected));
//         log::info!("Actual Importances No-Opt:\t{:?}", as_pretty_string_from(&actual_noopt));
//         log::info!("Actual Importances Opt:\t{:?}", as_pretty_string_from(&actual_opt));
//
//         assert_maps_equal(&actual_noopt, &expected, 0.0001);
//         //TODO Adding this fails the test
//         //assert_maps_equal(&actual_opt, &expected, 0.0001);
//     }
//
//     fn as_pretty_string_from(input_map: &Vec<f64>) -> String {
//         // Convert the HashMap into a vector of key-value pairs and sort it by keys
//         // Construct the string and print it on one line
//         let result = input_map
//             .iter()
//             .enumerate()
//             .map(|(key, value)| format!("{}: {:.4}", key, value))
//             .collect::<Vec<_>>()
//             .join(", ");
//         result
//     }
//
//
//     fn create_training_dataset() -> SessionDataset {
//         SessionDataset::new(vec![
//             Interaction {session_id: 0, item_id: 91, timestamp: 20},
//             Interaction {session_id: 0, item_id: 95, timestamp: 19},
//             Interaction {session_id: 1, item_id: 91, timestamp: 8},
//             Interaction {session_id: 1, item_id: 95, timestamp: 7},
//             Interaction {session_id: 2, item_id: 92, timestamp: 16},
//             Interaction {session_id: 3, item_id: 95, timestamp: 15},
//             Interaction {session_id: 3, item_id: 91, timestamp: 14},
//             Interaction {session_id: 3, item_id: 95, timestamp: 13},
//             Interaction {session_id: 4, item_id: 92, timestamp: 12},
//             Interaction {session_id: 4, item_id: 95, timestamp: 11},
//             Interaction {session_id: 5, item_id: 91, timestamp: 10},
//             Interaction {session_id: 5, item_id: 95, timestamp: 9},
//             Interaction {session_id: 6, item_id: 92, timestamp: 18},
//             Interaction {session_id: 6, item_id: 95, timestamp: 17},
//             ])
//     }
//
//     fn create_validation_dataset() -> SessionDataset {
//         SessionDataset::new(vec![
//             Interaction {session_id: 999, item_id: 92, timestamp: 10000000},
//             Interaction {session_id: 999, item_id: 95, timestamp: 10000010},
//             ])
//     }
// }
//
//
// fn assert_maps_equal(expected: &Vec<f64>, actual: &Vec<f64>, margin: f64) {
//     assert_eq!(expected.len(), actual.len(), "HashMaps have different lengths");
//
//     for (idx, value1) in expected.iter().enumerate() {
//         if let Some(value2) = actual.get(idx) {
//             assert!((value1 - value2).abs() < margin);
//         } else {
//             panic!("Key '{}' not found in actual", idx);
//         }
//     }
// }
//
