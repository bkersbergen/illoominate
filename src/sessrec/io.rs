use crate::sessrec::types::{Interaction, ItemId, SessionDataset, SessionId, Time};
use csv::ReaderBuilder;
use itertools::Itertools;
use polars::prelude::*;
use pyo3::PyErr;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};

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

fn build_session_dataset_from_grouped(
    grouped: HashMap<SessionId, Vec<(ItemId, Time)>>,
) -> SessionDataset {
    let sessions = grouped
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
            let session_items = item_ids_with_order
                .into_iter()
                .map(|(item, _)| item)
                .collect();

            (session_id, (session_items, max_timestamp))
        })
        .collect();

    SessionDataset { sessions }
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

pub fn get_sustainable_items(df: DataFrame) -> HashSet<ItemId> {
    cast_i64_column(&df, "item_id")
        .ok()
        .and_then(|column| {
            column.i64().ok().map(|values| {
                values
                    .into_no_null_iter()
                    .filter_map(|value| value.try_into().ok())
                    .collect()
            })
        })
        .unwrap_or_default()
}

pub fn polars_to_interactions(df: DataFrame) -> Result<Vec<Interaction>, PyErr> {
    let session_id_col = cast_i64_column(&df, "session_id")?;
    let item_id_col = cast_i64_column(&df, "item_id")?;
    let timestamp_col = cast_i64_column(&df, "timestamp")?;
    let mut results = Vec::with_capacity(df.height());

    for ((session_id, item_id), timestamp) in session_id_col
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .into_no_null_iter()
        .zip(
            item_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
        .zip(
            timestamp_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
    {
        let interaction = Interaction::new(
            session_id.try_into().map_err(|_| {
                py_value_error("Expected u32-compatible values in session_id column")
            })?,
            item_id
                .try_into()
                .map_err(|_| py_value_error("Expected u64-compatible values in item_id column"))?,
            timestamp.try_into().map_err(|_| {
                py_value_error("Expected usize-compatible values in timestamp column")
            })?,
        );
        results.push(interaction);
    }

    Ok(results)
}

pub fn session_dataset_from_polars(df: &DataFrame) -> Result<SessionDataset, PyErr> {
    let session_id_col = cast_i64_column(df, "session_id")?;
    let item_id_col = cast_i64_column(df, "item_id")?;
    let timestamp_col = cast_i64_column(df, "timestamp")?;

    let mut grouped: HashMap<SessionId, Vec<(ItemId, Time)>> = HashMap::new();

    for ((session_id, item_id), timestamp) in session_id_col
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .into_no_null_iter()
        .zip(
            item_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
        .zip(
            timestamp_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
    {
        grouped
            .entry(session_id.try_into().map_err(|_| {
                py_value_error("Expected u32-compatible values in session_id column")
            })?)
            .or_default()
            .push((
                item_id.try_into().map_err(|_| {
                    py_value_error("Expected u64-compatible values in item_id column")
                })?,
                timestamp.try_into().map_err(|_| {
                    py_value_error("Expected usize-compatible values in timestamp column")
                })?,
            ));
    }

    Ok(build_session_dataset_from_grouped(grouped))
}

pub fn dense_session_dataset_from_polars(df: &DataFrame) -> Result<SessionDataset, PyErr> {
    let session_id_col = cast_i64_column(df, "session_id")?;
    let item_id_col = cast_i64_column(df, "item_id")?;
    let timestamp_col = cast_i64_column(df, "timestamp")?;

    let mut grouped: HashMap<SessionId, Vec<(ItemId, Time)>> = HashMap::new();
    let mut session_mapping: HashMap<i64, SessionId> = HashMap::new();
    let mut next_session_id: SessionId = 0;

    for ((session_id, item_id), timestamp) in session_id_col
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .into_no_null_iter()
        .zip(
            item_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
        .zip(
            timestamp_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
    {
        let dense_session_id = match session_mapping.get(&session_id) {
            Some(&dense_id) => dense_id,
            None => {
                let dense_id = next_session_id;
                next_session_id = next_session_id
                    .checked_add(1)
                    .ok_or_else(|| py_value_error("Too many sessions to fit in u32"))?;
                session_mapping.insert(session_id, dense_id);
                dense_id
            }
        };

        grouped.entry(dense_session_id).or_default().push((
            item_id
                .try_into()
                .map_err(|_| py_value_error("Expected u64-compatible values in item_id column"))?,
            timestamp.try_into().map_err(|_| {
                py_value_error("Expected usize-compatible values in timestamp column")
            })?,
        ));
    }

    Ok(build_session_dataset_from_grouped(grouped))
}

pub fn dense_session_dataset_from_file(
    path_to_csvfile: &str,
    delimiter: u8,
) -> io::Result<SessionDataset> {
    let file = File::open(path_to_csvfile)?;
    let mut line_iterator = io::BufReader::new(file).lines();
    let separator = delimiter as char;

    line_iterator.next();

    let mut grouped: HashMap<SessionId, Vec<(ItemId, Time)>> = HashMap::new();
    let mut session_mapping: HashMap<u64, SessionId> = HashMap::new();
    let mut next_session_id: SessionId = 0;

    for line_result in line_iterator {
        let rawline = match line_result {
            Ok(rawline) => rawline,
            Err(_) => continue,
        };

        let mut parts = rawline.splitn(3, separator);
        let raw_session_id = match parts.next().and_then(|value| value.parse::<u64>().ok()) {
            Some(value) => value,
            None => continue,
        };
        let item_id = match parts.next().and_then(|value| value.parse::<ItemId>().ok()) {
            Some(value) => value,
            None => continue,
        };
        let timestamp = match parts.next().and_then(parse_time_value) {
            Some(value) => value,
            None => continue,
        };

        let dense_session_id = match session_mapping.get(&raw_session_id) {
            Some(&dense_id) => dense_id,
            None => {
                let dense_id = next_session_id;
                next_session_id = next_session_id.checked_add(1).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Too many sessions to fit in u32",
                    )
                })?;
                session_mapping.insert(raw_session_id, dense_id);
                dense_id
            }
        };

        grouped
            .entry(dense_session_id)
            .or_default()
            .push((item_id, timestamp));
    }

    Ok(build_session_dataset_from_grouped(grouped))
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

fn cast_i64_column(df: &DataFrame, column_name: &str) -> Result<Column, PyErr> {
    let column = df
        .column(column_name)
        .map_err(|e| py_value_error(e.to_string()))?
        .cast(&DataType::Int64)
        .map_err(|e| py_value_error(e.to_string()))?;

    if column
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .null_count()
        > 0
    {
        return Err(py_value_error(format!(
            "Column '{column_name}' must not contain null values"
        )));
    }

    Ok(column)
}

fn parse_time_value(raw: &str) -> Option<Time> {
    raw.parse::<Time>()
        .ok()
        .or_else(|| raw.parse::<f64>().ok().map(|value| value.round() as Time))
}

fn py_value_error(message: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepare_python() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn test_polars_to_success() {
        prepare_python();
        // Create a sample DataFrame with valid data
        let df = df![
            "session_id" => &[1, 2, 3],
            "item_id" => &[101, 102, 103],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ]
        .expect("Failed to create DataFrame");

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
        prepare_python();
        // Create a DataFrame with a missing "item_id" column
        let df = df![
            "session_id" => &[1, 2, 3],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ]
        .expect("Failed to create DataFrame");

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
        prepare_python();
        // Create a DataFrame with an invalid data type in "session_id" column
        let df = df![
            "session_id" => &["a", "b", "c"],  // invalid string type
            "item_id" => &[101, 102, 103],
            "timestamp" => &[1000000000, 1000000010, 1000000020]
        ]
        .expect("Failed to create DataFrame");

        // Call the polars_to function
        let result = polars_to_interactions(df);

        // Assert that the result is an error
        assert!(result.is_err());

        // Optionally, check the specific error message
        let error_message = format!("{}", result.unwrap_err());
        assert!(!error_message.is_empty());
    }
}
