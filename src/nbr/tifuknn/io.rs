use crate::nbr::tifuknn::types::{Basket, UserId};
use grouping_by::GroupingBy;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use crate::nbr::types::NextBasketDataset;

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

#[derive(Debug, PartialEq)]
struct Purchase {
    user: UserId,
    basket: usize,
    item: usize,
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

    let baskets_by_user: HashMap<UserId, Vec<Basket>> = purchases
        .into_iter()
        .grouping_by(|p| p.user)
        .into_iter()
        .map(|(user, user_purchases)| {
            let mut baskets: Vec<Basket> = user_purchases
                .into_iter()
                .grouping_by(|p| p.basket)
                .into_iter()
                .map(|(basket_id, basket_purchases)| {
                    let items = basket_purchases.into_iter().map(|p| p.item).collect();
                    Basket::new(basket_id, items)
                })
                .collect();

            baskets.sort_by_key(|b| b.id);

            (user, baskets)
        })
        .collect();

    NextBasketDataset {
        user_baskets: baskets_by_user,
    }
}
