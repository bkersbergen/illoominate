use crate::nbr::tifuknn::types::{Basket, UserId};
use grouping_by::GroupingBy;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use polars_core::datatypes::AnyValue;
use polars_core::frame::DataFrame;
use pyo3::PyErr;
use crate::nbr::types::NextBasketDataset;

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

#[derive(Debug, PartialEq)]
pub struct Purchase {
    pub(crate) user: UserId,
    pub(crate) basket: usize,
    pub(crate) item: usize,
}


pub fn polars_to_purchases(df: DataFrame) -> Result<Vec<Purchase>, PyErr> {

    // Pre-fetch columns to avoid repeated lookups
    let user_id_col = df.column("user_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let basket_id_col = df.column("basket_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let item_id_col = df.column("item_id")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Pre-allocate memory for results
    let mut results = Vec::with_capacity(df.height());

    // Iterate by row index, directly accessing each column
    for i in 0..df.height() {
        let user_id = match user_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in user_id column")),
        };
        let basket_id = match basket_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in basket_id column")),
        };
        let item_id = match item_id_col.get(i) {
            Ok(AnyValue::Int64(val)) => val.try_into().unwrap(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected u64 in item_id column")),
        };

        // Create an instance and store it
        results.push(Purchase {
            user: user_id,
            basket: basket_id,
            item: item_id,
        });
    }
    Ok(results)
}


pub fn read_baskets_file(dataset_file: &str) -> NextBasketDataset {
    let mut purchases: Vec<Purchase> = Vec::new();

    if let Ok(lines) = read_lines(dataset_file) {
        for line in lines.skip(1).flatten() {
            let triple: Vec<usize> = line
                .split('\t')
                .map(|s| s.parse::<usize>().unwrap())
                .collect();

            purchases.push(Purchase {
                user: triple[0] as UserId,
                basket: triple[1],
                item: triple[2],
            });
        }
    }

    crate::nbr::types::NextBasketDataset::from(&purchases)
}
