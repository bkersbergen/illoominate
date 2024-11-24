use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use illoominate::sessrec::io;
use illoominate::sessrec::types::{Interaction, ItemId, SessionId};

fn to_csv(mut dataset: Vec<Interaction>, output_path: &str) {
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

    writeln!(output_file, "session_id\titem_id\ttimestamp").expect("Failed to write to file");

    dataset.sort_by(|a, b| {
        a.session_id
            .cmp(&b.session_id)
            .then_with(|| a.timestamp.cmp(&b.timestamp))
            .then_with(|| a.item_id.cmp(&b.item_id))
    });

    for datapoint in dataset {
        writeln!(
            output_file,
            "{}\t{}\t{}",
            datapoint.session_id, datapoint.item_id, datapoint.timestamp
        )
        .expect("Failed to write to file");
    }
}

fn mapping_to_csv<T: Display>(mapping: HashMap<T, T>, output_path: &str) {
    let path = Path::new(output_path);
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).expect("Failed to create directories");
        }
    }
    if path.exists() {
        fs::remove_file(path).expect("Cannot delete file...");
    }

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(false)
        .open(output_path)
        .unwrap_or_else(|_| panic!("Failed to create file {output_path}"));

    writeln!(output_file, "original\tmapped_to").expect("Failed to write to file");

    for (key, value) in mapping {
        writeln!(output_file, "{}\t{}", key, value).expect("Failed to write to file");
    }
}

fn filter_and_save_validation_and_test(
    path: &str,
    training_sessions: &HashSet<SessionId>,
    validation_sessions: &HashSet<SessionId>,
    test_sessions: &HashSet<SessionId>,
    item_ids_in_training: &HashSet<ItemId>,
    mapped_interactions: &[Interaction],
) {
    let mut validation_dataset = Vec::new();
    let mut test_dataset = Vec::new();

    let mut session_interactions: HashMap<SessionId, Vec<Interaction>> = HashMap::new();

    // Collect interactions, filtering out items not in training data
    for interaction in mapped_interactions {
        if !training_sessions.contains(&interaction.session_id) && item_ids_in_training.contains(&interaction.item_id) {
            session_interactions
                .entry(interaction.session_id)
                .or_default()
                .push(interaction.clone());
        }
    }

    // Filter out sessions with fewer than 2 interactions
    for (session_id, interactions) in session_interactions {
        if interactions.len() >= 2 {
            if validation_sessions.contains(&session_id) {
                validation_dataset.extend(interactions.clone());
            } else if test_sessions.contains(&session_id) {
                test_dataset.extend(interactions.clone());
            }
        }
    }

    // Save the filtered datasets to CSV files
    to_csv(validation_dataset, &format!("{}/valid.csv", path));
    to_csv(test_dataset, &format!("{}/test.csv", path));
}
fn save_training_set(
    path: &str,
    training_sessions: &HashSet<SessionId>,
    mapped_interactions: &[Interaction],
) -> HashSet<ItemId> {
    let mut training_dataset = Vec::new();
    let mut item_ids_in_training = HashSet::new();
    let mut session_reindex_map = HashMap::new();
    let mut new_session_id = 0;

    for interaction in mapped_interactions {
        if training_sessions.contains(&interaction.session_id) {
            // Re-index the session_id
            let new_id = session_reindex_map
                .entry(interaction.session_id)
                .or_insert_with(|| {
                    let id = new_session_id;
                    new_session_id += 1;
                    id
                });

            // Update the interaction with the new session_id
            let mut reindexed_interaction = interaction.clone();
            reindexed_interaction.session_id = *new_id;

            item_ids_in_training.insert(reindexed_interaction.item_id);
            training_dataset.push(reindexed_interaction);
        }
    }

    to_csv(training_dataset, &format!("{}/train.csv", path));

    item_ids_in_training
}

fn map_interactions(
    path: &str,
    mut raw_interactions: Vec<Interaction>,
) -> (Vec<SessionId>, Vec<Interaction>) {
    // TODO Maybe we should sort first?
    raw_interactions.sort_by_key(|datapoint| datapoint.timestamp);

    let mut session_id_mapping: HashMap<SessionId, SessionId> = HashMap::new();
    let mut item_id_mapping: HashMap<ItemId, ItemId> = HashMap::new();

    let mut session_index_to_assign: SessionId = 0;
    let mut item_index_to_assign: ItemId = 0;

    let mut sessions_seen = HashSet::new();
    let mut sessions_by_start_time = Vec::new();

    let mut mapped_interactions = Vec::new();

    for interaction in &raw_interactions {
        let mapped_session_id = session_id_mapping
            .entry(interaction.session_id)
            .or_insert_with(|| {
                let mapped_session_id = session_index_to_assign;
                session_index_to_assign += 1;
                mapped_session_id
            });

        let mapped_item_id = item_id_mapping
            .entry(interaction.item_id)
            .or_insert_with(|| {
                let mapped_item_id = item_index_to_assign;
                item_index_to_assign += 1;
                mapped_item_id
            });

        if !sessions_seen.contains(mapped_session_id) {
            sessions_seen.insert(*mapped_session_id);
            sessions_by_start_time.push(*mapped_session_id);
        }

        let mapped_interaction =
            Interaction::new(*mapped_session_id, *mapped_item_id, interaction.timestamp);
        mapped_interactions.push(mapped_interaction);
    }

    mapping_to_csv(
        session_id_mapping,
        &format!("{}/__session_id_mapping.csv", path),
    );
    mapping_to_csv(item_id_mapping, &format!("{}/__item_id_mapping.csv", path));

    (sessions_by_start_time, mapped_interactions)
}

fn split_sessions(
    mut sessions_by_start_time: Vec<SessionId>,
    seed: usize,
    train_fraction: f64,
    validation_fraction: f64,
) -> (HashSet<SessionId>, HashSet<SessionId>, HashSet<SessionId>) {
    let train_split_index = (sessions_by_start_time.len() as f64 * train_fraction) as usize;

    let (training_sessions, valtest_sessions) =
        sessions_by_start_time.split_at_mut(train_split_index);

    let valtest_split_index = (valtest_sessions.len() as f64 * validation_fraction) as usize;

    let mut rng = StdRng::seed_from_u64(seed as u64);
    valtest_sessions.shuffle(&mut rng);
    let (validation_sessions, test_sessions) = valtest_sessions.split_at(valtest_split_index);

    let training_sessions: HashSet<SessionId> = training_sessions.iter_mut().map(|x| *x).collect();
    let validation_sessions: HashSet<SessionId> =
        validation_sessions.iter().copied().collect();
    let test_sessions: HashSet<SessionId> = test_sessions.iter().copied().collect();

    (training_sessions, validation_sessions, test_sessions)
}

pub struct DatasetParameters<'a> {
    data_path: &'a str,
    num_interactions: usize,
    seed: usize,
    train_fraction: f64,
    validation_fraction: f64,
    dataset_name: &'a str,
}

pub const PERFORMANCE_BOL1M: DatasetParameters = DatasetParameters {
    data_path: "bolcom-clicks-50m_train.txt",
    num_interactions: 1_100_000,
    seed: 42,
    train_fraction: 0.95,
    validation_fraction: 0.5,
    dataset_name: "performance/bol1m",
};

pub const PERFORMANCE_BOL2M: DatasetParameters = DatasetParameters {
    data_path: "bolcom-clicks-50m_train.txt",
    num_interactions: 2_200_000,
    seed: 42,
    train_fraction: 0.95, // 1 - ((2xtest size) 1000 / (num_interactions) 2_000_000)
    validation_fraction: 0.5,
    dataset_name: "performance/bol2m",
};

pub const PERFORMANCE_BOL5M: DatasetParameters = DatasetParameters {
    data_path: "bolcom-clicks-50m_train.txt",
    num_interactions: 5_500_000,
    seed: 42,
    train_fraction: 0.95,
    validation_fraction: 0.5,
    dataset_name: "performance/bol5m",
};

pub const PERFORMANCE_BOL10M: DatasetParameters = DatasetParameters {
    data_path: "bolcom-clicks-50m_train.txt",
    num_interactions: 11_000_000,
    seed: 42,
    train_fraction: 0.95,
    validation_fraction: 0.5,
    dataset_name: "performance/bol10m",
};

pub const NOW_PLAYING_500K: DatasetParameters = DatasetParameters {
    data_path: "nowplaying_sessions.csv",
    num_interactions: 607_000,
    seed: 42,
    train_fraction: 0.8,
    validation_fraction: 0.5,
    dataset_name: "nowplaying500k",
};

pub const NOW_PLAYING_250K: DatasetParameters = DatasetParameters {
    data_path: "nowplaying_sessions.csv",
    num_interactions: 250_000,
    seed: 42,
    train_fraction: 0.8,
    validation_fraction: 0.5,
    dataset_name: "nowplaying250k",
};

pub const NOW_PLAYING_1DOT5M: DatasetParameters = DatasetParameters {
    data_path: "/Users/ssc/projects/datasets/session-based-recommendation/nowplaying/raw/nowplaying_sessions.csv",
    num_interactions: 1_500_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "nowplaying1dot5m",
};

pub const ECOM_1M: DatasetParameters = DatasetParameters {
    // bolcom-clicks-1m_train_full 20210826.txt is strings
    // "data/bolcom1m/bolcom-clicks-1m_train20201207.txt" model performance drops when reducing training data
    data_path: "bolcom-clicks-1m_train_full 20231010.txt",
    num_interactions: 1_500_000,
    seed: 42,
    train_fraction: 0.7,
    validation_fraction: 0.3,
    dataset_name: "ecom1m",
};

pub const RETAIL_ROCKET: DatasetParameters = DatasetParameters {
    data_path: "retailrocket9_train.txt",
    num_interactions: 10_500_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "retailrocket",
};

pub const RSC15: DatasetParameters = DatasetParameters {
    data_path: "rsc15-clicks_train_full.9.txt",
    num_interactions: 100_500_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "rsc15",
};

pub const TMALL: DatasetParameters = DatasetParameters {
    data_path: "tmall_train_full.0.txt",
    num_interactions: 100_500_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "tmall",
};

pub const TMALL_SAMPLE: DatasetParameters = DatasetParameters {
    data_path: "tmall_train_full.0.txt",
    num_interactions: 200_000,
    seed: 42,
    train_fraction: 0.90,
    validation_fraction: 0.5,
    dataset_name: "tmall_sample",
};
pub const ECOM_50M: DatasetParameters = DatasetParameters {
    data_path: "bolcom-clicks-50m_train.txt",
    num_interactions: 55_000_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "ecom50m",
};

pub const ECOM_100K: DatasetParameters = DatasetParameters {
    data_path: "ecom-clicks-100k.txt",
    num_interactions: 100_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "ecom100k",
};

pub const INSTACART_50K: DatasetParameters = DatasetParameters {
    data_path: "instacart-train.csv",
    num_interactions: 50_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "instacart50k",
};

pub const INSTACART_1M: DatasetParameters = DatasetParameters {
    data_path: "/Users/ssc/projects/datasets/session-based-recommendation/instacart-train.csv",
    num_interactions: 1_000_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "instacart1m",
};

pub const INSTACART_250K: DatasetParameters = DatasetParameters {
    data_path: "/Users/ssc/projects/datasets/session-based-recommendation/instacart-train.csv",
    num_interactions: 250_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "instacart250k",
};

pub const NUTRICART_50K: DatasetParameters = DatasetParameters {
    data_path: "nutricart.csv",
    num_interactions: 50_000,
    seed: 42,
    train_fraction: 0.5,
    validation_fraction: 0.5,
    dataset_name: "nutricart50k",
};

fn main() {
    let params = PERFORMANCE_BOL10M;

    let path = format!("{}/{}", "data/sbr", params.dataset_name);
    let path = path.as_str();
    let train_filename = params.data_path;
    let raw_interactions = io::read_data(format!("{}/{}", path, train_filename).as_str())
        .into_iter()
        .take(params.num_interactions)
        .collect();

    let (sessions_by_start_time, mapped_interactions) = map_interactions(path, raw_interactions);

    let (training_sessions, validation_sessions, test_sessions) = split_sessions(
        sessions_by_start_time,
        params.seed,
        params.train_fraction,
        params.validation_fraction,
    );

    let item_ids_in_training = save_training_set(path, &training_sessions, &mapped_interactions);

    filter_and_save_validation_and_test(
        path,
        &training_sessions,
        &validation_sessions,
        &test_sessions,
        &item_ids_in_training,
        &mapped_interactions,
    );
}
