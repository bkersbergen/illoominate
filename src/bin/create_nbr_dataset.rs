use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::io::Write;
use std::path::Path;
use std::{env, fs, io};

#[derive(Clone)]
struct RawPurchase {
    raw_user_id: usize,
    raw_basket_id: usize,
    raw_item_id: usize,
}

//TODO refactor to remove code duplication
fn mapping_to_csv<T: Display>(mapping: HashMap<T, T>, output_path: &str) {
    let path = Path::new(output_path);
    if path.exists() {
        fs::remove_file(path).expect("Cannot delete file...");
    }

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(false)
        .open(output_path)
        .expect("Failed to open file");

    writeln!(output_file, "original\tmapped_to").expect("Failed to write to file");

    for (key, value) in mapping {
        writeln!(output_file, "{}\t{}", key, value).expect("Failed to write to file");
    }
}

fn read_data(
    path_to_csvfile: &str,
    customer_col: usize,
    basket_col: usize,
    item_col: usize,
) -> Vec<RawPurchase> {
    println!("{:?}", env::current_dir());
    println!("{:?}", path_to_csvfile);
    if Path::new(path_to_csvfile).exists() {
        println!("yeah file found!");
    }

    let file = File::open(path_to_csvfile).expect("Unable to read input file");

    let mut line_iterator = io::BufReader::new(file).lines();
    line_iterator.next(); // skip header
    let raw_purchases = line_iterator.skip(1).filter_map(move |result| {
        if let Ok(rawline) = result {
            let parts = rawline.split(',').collect::<Vec<_>>();
            let (raw_basket_id, raw_user_id, raw_item_id) = (
                parts.get(basket_col).unwrap().parse::<usize>().unwrap(),
                parts.get(customer_col).unwrap().parse::<usize>().unwrap(),
                parts.get(item_col).unwrap().parse::<usize>().unwrap(),
            );
            Some(RawPurchase {
                raw_user_id,
                raw_basket_id,
                raw_item_id,
            })
        } else {
            log::debug!(
                "Error parsing line: {:?} in path_to_csvfile: {:?}",
                result,
                path_to_csvfile
            );
            None
        }
    });
    raw_purchases.collect()
}

fn map_ids(mut raw_ids: Vec<usize>) -> HashMap<usize, usize> {
    raw_ids.sort();
    raw_ids.dedup();

    let id_mapping: HashMap<usize, usize> = raw_ids
        .iter()
        .enumerate()
        .map(|(index, user_id)| (*user_id, index))
        .collect();

    id_mapping
}

fn main() {
    let enable_tifu_preprocessing = false;
    let customer_col: usize = 0;
    let basket_col: usize = 1;
    let item_col: usize = 2;
    let path = "data/nbr/tafeng1k_experiments";
    let train_filename = "TaFeng_history_NB.csv";
    let valid_filename = "TaFeng_future_NB.csv";
    let train_purchases = read_data(
        format!("{}/{}", path, train_filename).as_str(),
        customer_col,
        basket_col,
        item_col,
    );
    let valid_purchases = read_data(
        format!("{}/{}", path, valid_filename).as_str(),
        customer_col,
        basket_col,
        item_col,
    );

    // println!("{}", train_purchases.len());
    // println!("{}", valid_purchases.len());

    let mut all_purchases: Vec<RawPurchase> = train_purchases
        .iter()
        .chain(valid_purchases.iter())
        .cloned()
        .collect();

    if enable_tifu_preprocessing {
        if train_filename.to_lowercase().contains("instacart")
            || train_filename.to_lowercase().contains("tafeng")
        {
            log::info!("Step 1: Remove least frequent items because train_filename contains instacart or tafeng");
            all_purchases = remove_least_frequent_items(all_purchases);
        }

        log::info!("Step 2: Remove customers with less than 4 unique items");
        all_purchases = remove_customers_with_less_than_n_unique_items(all_purchases, 4);

        log::info!("Step 3: Remove baskets with less than 4 items");
        all_purchases = remove_baskets_with_less_than_n_items(all_purchases, 4);

        log::info!("Step 4: Remove customers with less than 3 baskets");
        all_purchases = remove_customers_with_less_than_n_baskets(all_purchases, 3);
    }

    let all_user_ids: Vec<usize> = all_purchases
        .clone()
        .into_iter()
        .map(|raw_purchase| raw_purchase.raw_user_id)
        .collect();

    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed as u64);

    let mut user_ids_to_sample_from = all_user_ids.clone();
    user_ids_to_sample_from.sort();
    user_ids_to_sample_from.dedup();
    user_ids_to_sample_from.shuffle(&mut rng);

    let sampled_user_ids: HashSet<usize> = user_ids_to_sample_from.into_iter().take(1000).collect();

    let all_basket_ids: Vec<usize> = all_purchases
        .iter() // Convert Vec<RawPurchase> into an iterator over references
        .filter(|raw_purchase| sampled_user_ids.contains(&raw_purchase.raw_user_id))
        .map(|raw_purchase| raw_purchase.raw_basket_id)
        .collect();

    let all_item_ids: Vec<usize> = all_purchases
        .iter() // Convert Vec<RawPurchase> into an iterator over references
        .filter(|raw_purchase| sampled_user_ids.contains(&raw_purchase.raw_user_id))
        .map(|raw_purchase| raw_purchase.raw_item_id)
        .collect();

    // TODO refactor
    let sampled_user_ids_vec: Vec<usize> = sampled_user_ids.iter().copied().collect();

    let user_id_mapping = map_ids(sampled_user_ids_vec);
    let basket_id_mapping = map_ids(all_basket_ids);
    let item_id_mapping = map_ids(all_item_ids);

    println!(
        "{}, {}, {}",
        user_id_mapping.len(),
        basket_id_mapping.len(),
        item_id_mapping.len()
    );

    let output_file_binding = format!("{}/{}", path, "baskets.csv").clone();
    let purchases_path = Path::new(output_file_binding.as_str());
    if purchases_path.exists() {
        fs::remove_file(purchases_path).expect("Cannot delete file...");
    }

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(false)
        .open(purchases_path)
        .expect("Failed to open file");

    writeln!(output_file, "user_id\tbasket_id\titem_id").expect("Failed to write to file");

    all_purchases.iter().for_each(|raw_purchase| {
        if sampled_user_ids.contains(&raw_purchase.raw_user_id) {
            let user_id = user_id_mapping.get(&raw_purchase.raw_user_id).unwrap();
            let basket_id = basket_id_mapping.get(&raw_purchase.raw_basket_id).unwrap();
            let item_id = item_id_mapping.get(&raw_purchase.raw_item_id).unwrap();
            writeln!(output_file, "{}\t{}\t{}", user_id, basket_id, item_id)
                .expect("Failed to write to file");
        }
    });

    mapping_to_csv(
        user_id_mapping,
        format!("{}/{}", path, "__user_id_mapping.csv").as_str(),
    );
    mapping_to_csv(
        basket_id_mapping,
        format!("{}/{}", path, "__basket_id_mapping.csv").as_str(),
    );
    mapping_to_csv(
        item_id_mapping,
        format!("{}/{}", path, "__item_id_mapping.csv").as_str(),
    );
}

fn remove_least_frequent_items(purchases: Vec<RawPurchase>) -> Vec<RawPurchase> {
    let mut item_count: HashMap<usize, usize> = HashMap::new();
    for purchase in &purchases {
        *item_count.entry(purchase.raw_item_id).or_insert(0) += 1;
    }
    let threshold = 1; // Define your threshold for "least frequent"
    purchases
        .into_iter()
        .filter(|purchase| item_count[&purchase.raw_item_id] > threshold)
        .collect()
}

fn remove_customers_with_less_than_n_unique_items(
    purchases: Vec<RawPurchase>,
    n: usize,
) -> Vec<RawPurchase> {
    let mut customer_item_count: HashMap<usize, HashSet<usize>> = HashMap::new();
    for purchase in &purchases {
        customer_item_count
            .entry(purchase.raw_user_id)
            .or_default()
            .insert(purchase.raw_item_id);
    }
    let allowed_customers: HashSet<usize> = customer_item_count
        .into_iter()
        .filter(|(_, items)| items.len() >= n)
        .map(|(user_id, _)| user_id)
        .collect();
    purchases
        .into_iter()
        .filter(|purchase| allowed_customers.contains(&purchase.raw_user_id))
        .collect()
}

fn remove_baskets_with_less_than_n_items(
    purchases: Vec<RawPurchase>,
    n: usize,
) -> Vec<RawPurchase> {
    let mut basket_item_count: HashMap<usize, usize> = HashMap::new();
    for purchase in &purchases {
        *basket_item_count.entry(purchase.raw_basket_id).or_insert(0) += 1;
    }
    let allowed_baskets: HashSet<usize> = basket_item_count
        .into_iter()
        .filter(|(_, count)| *count >= n)
        .map(|(basket_id, _)| basket_id)
        .collect();
    purchases
        .into_iter()
        .filter(|purchase| allowed_baskets.contains(&purchase.raw_basket_id))
        .collect()
}

fn remove_customers_with_less_than_n_baskets(
    purchases: Vec<RawPurchase>,
    n: usize,
) -> Vec<RawPurchase> {
    let mut customer_basket_count: HashMap<usize, HashSet<usize>> = HashMap::new();
    for purchase in &purchases {
        customer_basket_count
            .entry(purchase.raw_user_id)
            .or_default()
            .insert(purchase.raw_basket_id);
    }
    let allowed_customers: HashSet<usize> = customer_basket_count
        .into_iter()
        .filter(|(_, baskets)| baskets.len() >= n)
        .map(|(user_id, _)| user_id)
        .collect();
    purchases
        .into_iter()
        .filter(|purchase| allowed_customers.contains(&purchase.raw_user_id))
        .collect()
}
