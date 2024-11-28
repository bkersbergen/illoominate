use csv::ReaderBuilder;
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use polars_core::datatypes::AnyValue;
use pyo3::PyErr;
use polars::prelude::*;
use crate::sessrec::types::{Interaction, ItemId, SessionDataset, SessionId, Time};

impl SessionDataset {
    pub fn new(interactions: Vec<Interaction>) -> SessionDataset {
        let sessions: HashMap<SessionId, (Vec<ItemId>, Time)> = interactions
            .into_iter()
            .map(|datapoint| {
                (
                    datapoint.session_id,
                    (datapoint.item_id, datapoint.timestamp),
                )
            })
            .into_group_map()
            .into_iter()
            .map(|(session_id, mut item_ids_with_order)| {
                item_ids_with_order.sort_by(|(item_a, time_a), (item_b, time_b)| {
                    let ordering_by_time = time_a.cmp(time_b);

                    if ordering_by_time == Ordering::Equal {
                        item_a.cmp(item_b)
                    } else {
                        ordering_by_time
                    }
                });

                let (_item_id, max_timestamp) = *item_ids_with_order.last().unwrap();
                let session_items: Vec<ItemId> = item_ids_with_order
                    .into_iter()
                    .map(|(item, _order)| item)
                    .collect();

                (session_id, (session_items, max_timestamp))
            })
            .collect();

        SessionDataset { sessions }
    }
}

pub fn read_data(path_to_csvfile: &str) -> Vec<Interaction> {
    let file = File::open(path_to_csvfile).expect("Unable to read input file");
    let mut line_iterator = io::BufReader::new(file).lines();
    line_iterator.next(); // skip header
    let training_data = line_iterator.filter_map(move |result| {
        if let Ok(rawline) = result {
            let parts = rawline.split_whitespace().take(3).collect::<Vec<_>>();
            let (session_id, item_id, timestamp) = (
                parts.first().unwrap().parse::<SessionId>().unwrap(),
                parts.get(1).unwrap().parse::<ItemId>().unwrap(),
                parts.get(2).unwrap().parse::<f64>().unwrap(),
            );
            Some(Interaction::new(
                session_id,
                item_id,
                timestamp.round() as Time,
            ))
        } else {
            log::debug!(
                "Error parsing line: {:?} in path_to_csvfile: {:?}",
                result,
                path_to_csvfile
            );
            None
        }
    });
    training_data.collect()
}



pub fn polars_to_interactions(df: DataFrame) -> Result<Vec<Interaction>, PyErr> {

    // Pre-fetch columns to avoid repeated lookups
    let session_id_col = df.column("session_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let item_id_col = df.column("item_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let timestamp_col = df.column("timestamp")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Pre-allocate memory for results
    let mut results = Vec::with_capacity(df.height());

    // Iterate by row index, directly accessing each column
    for i in 0..df.height() {
        let session_id = match session_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in session_id column")),
        };
        let item_id = match item_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in item_id column")),
        };
        let timestamp = match timestamp_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in timestamp column")),
        };

        // Create an Interaction instance and store it
        let interaction = Interaction::new(session_id, item_id, timestamp);
        results.push(interaction);
    }
    Ok(results)
}

pub fn read_sustainable_products_info(path_to_csvfile: &str) -> HashSet<ItemId> {
    let file = File::open(path_to_csvfile)
        .unwrap_or_else(|_| panic!("Failed to open file: {}", path_to_csvfile));
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_reader(file);

    let mut result = HashSet::new();

    for string_record in reader.records() {
        if let Ok(record) = string_record {
            if let Some(value) = record.get(0) {
                if let Some(flag) = record.get(1) {
                    if flag == "True" {
                        if let Ok(num) = value.parse::<ItemId>() {
                            result.insert(num);
                        }
                    }
                }
            }
        }
    }
    result
}


#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_polars_to_success() {
        // Create a sample DataFrame with valid data
        let df = df![
            "session_id" => &[1, 2, 3],
            "item_id" => &[101, 102, 103],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ].expect("Failed to create DataFrame");

        // Call the polars_to function
        let result = polars_to_interactions(df);

        // Assert that the result is Ok
        assert!(result.is_ok());

        // Unwrap the result and check contents
        let interactions = result.unwrap();
        assert_eq!(interactions.len(), 3);

        assert_eq!(interactions[0].session_id, 1);
        assert_eq!(interactions[0].item_id, 101);
        assert_eq!(interactions[0].timestamp, 1000000000);

        assert_eq!(interactions[1].session_id, 2);
        assert_eq!(interactions[1].item_id, 102);
        assert_eq!(interactions[1].timestamp, 1000000010);

        assert_eq!(interactions[2].session_id, 3);
        assert_eq!(interactions[2].item_id, 103);
        assert_eq!(interactions[2].timestamp, 1000000020);
    }

    #[test]
    fn test_polars_to_missing_column() {
        // Create a DataFrame with a missing "item_id" column
        let df = df![
            "session_id" => &[1, 2, 3],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ].expect("Failed to create DataFrame");

        // Call the polars_to function
        let result = polars_to_interactions(df);

        // Assert that the result is an error
        assert!(result.is_err());

        // Optionally, check the specific error message
        let error_message = format!("{}", result.unwrap_err());
        assert!(error_message.contains("item_id"));
    }

    #[test]
    fn test_polars_to_invalid_data_type() {
        // Create a DataFrame with an invalid data type in "session_id" column
        let df = df![
            "session_id" => &["a", "b", "c"],  // invalid string type
            "item_id" => &[101, 102, 103],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ].expect("Failed to create DataFrame");

        // Call the polars_to function
        let result = polars_to_interactions(df);

        // Assert that the result is an error
        assert!(result.is_err());

        // Optionally, check the specific error message
        let error_message = format!("{}", result.unwrap_err());
        assert!(error_message.contains("Expected u64 in session_id column"));
    }
}