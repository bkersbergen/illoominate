use crate::nbr::tifuknn::types::Basket;
use crate::nbr::tifuknn::types::UserId;
use crate::nbr::types::NextBasketDataset;
use polars::prelude::*;
use pyo3::PyErr;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;

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
    let user_id_col = cast_i64_column(&df, "user_id")?;
    let basket_id_col = cast_i64_column(&df, "basket_id")?;
    let item_id_col = cast_i64_column(&df, "item_id")?;
    let mut results = Vec::with_capacity(df.height());

    for ((user_id, basket_id), item_id) in user_id_col
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .into_no_null_iter()
        .zip(
            basket_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
        .zip(
            item_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
    {
        results.push(Purchase {
            user: user_id
                .try_into()
                .map_err(|_| py_value_error("Expected u32-compatible values in user_id column"))?,
            basket: basket_id.try_into().map_err(|_| {
                py_value_error("Expected usize-compatible values in basket_id column")
            })?,
            item: item_id.try_into().map_err(|_| {
                py_value_error("Expected usize-compatible values in item_id column")
            })?,
        });
    }

    Ok(results)
}

pub fn next_basket_dataset_from_polars(df: &DataFrame) -> Result<NextBasketDataset, PyErr> {
    let user_id_col = cast_i64_column(df, "user_id")?;
    let basket_id_col = cast_i64_column(df, "basket_id")?;
    let item_id_col = cast_i64_column(df, "item_id")?;

    let mut baskets_by_user: std::collections::HashMap<
        UserId,
        std::collections::HashMap<usize, Vec<usize>>,
    > = std::collections::HashMap::new();

    for ((user_id, basket_id), item_id) in user_id_col
        .i64()
        .map_err(|e| py_value_error(e.to_string()))?
        .into_no_null_iter()
        .zip(
            basket_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
        .zip(
            item_id_col
                .i64()
                .map_err(|e| py_value_error(e.to_string()))?
                .into_no_null_iter(),
        )
    {
        let user_id: UserId = user_id
            .try_into()
            .map_err(|_| py_value_error("Expected u32-compatible values in user_id column"))?;
        let basket_id: usize = basket_id
            .try_into()
            .map_err(|_| py_value_error("Expected usize-compatible values in basket_id column"))?;
        let item_id: usize = item_id
            .try_into()
            .map_err(|_| py_value_error("Expected usize-compatible values in item_id column"))?;

        baskets_by_user
            .entry(user_id)
            .or_default()
            .entry(basket_id)
            .or_default()
            .push(item_id);
    }

    let user_baskets = baskets_by_user
        .into_iter()
        .map(|(user_id, baskets)| {
            let mut baskets: Vec<Basket> = baskets
                .into_iter()
                .map(|(basket_id, items)| Basket::new(basket_id, items))
                .collect();
            baskets.sort_by_key(|basket| basket.id);
            (user_id, baskets)
        })
        .collect();

    Ok(NextBasketDataset { user_baskets })
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

fn py_value_error(message: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.to_string())
}
