use chrono::Local;
use env_logger::Builder;

use illoominate::sessrec::io;


use illoominate::sessrec::types::SessionDataset;
use illoominate::sessrec::vmisknn::{Scored, VMISKNN};
use itertools::Itertools;
use log::LevelFilter;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::{env};

fn main() {
    init_logger();

    let data_location = match env::var("DATA_LOCATION") {
        Ok(val) => val,
        Err(_) => {
            log::error!("Environment variable DATA_LOCATION is not set");
            std::process::exit(1); // Exit with an error code
        }
    };

    let mrr_positive_train = SessionDataset::new(io::read_data(&format!(
        "{}/mrr_positive_train.csv",
        data_location
    )));
    let train_sustainable_mrr = SessionDataset::new(io::read_data(&format!(
        "{}/responsiblemrr_positive_train.csv",
        data_location
    )));
    let valid = SessionDataset::new(io::read_data(&format!("{}/valid.csv", data_location)));

    let k = 500;
    let m = 500;
    let mrr_model = VMISKNN::fit_dataset(&mrr_positive_train, m, k, 1.0);
    let sustainable_model = VMISKNN::fit_dataset(&train_sustainable_mrr, m, k, 1.0);

    let output_filename = "sustainability_predictions.csv";
    let file = match File::create(output_filename) {
        Ok(file) => file,
        Err(err) => {
            log::error!(
                "Error: Unable to create file '{}': {}",
                output_filename,
                err
            );
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    // write header to output file
    if let Err(err) = writeln!(
        writer,
        "session_id,query_session_items,mrr_recommendations,sustainable_recommendations"
    ) {
        log::error!(
            "Error: writing header to output file: '{}': {}",
            output_filename,
            err
        );
        return;
    }

    for (session_id, (session_items, _max_timestamp)) in valid.sessions.iter() {
        for pos in 1..session_items.len() {
            let query_session = &session_items[0..pos]
                .iter()
                .map(|&id| Scored::new(id as u32, 1.0))
                .collect_vec();
            // let actual_next_items = &session_items[pos..].to_vec();
            let mrr_recommendations = mrr_model.predict(query_session);
            let sustainable_recommendations = sustainable_model.predict(query_session);
            if let Err(_err) = writeln!(
                writer,
                "{},\"{:?}\",\"{:?}\",\"{:?}\"",
                session_id, query_session, mrr_recommendations, sustainable_recommendations
            ) {
                return;
            }
        }
    }
    writer.flush().expect("error closing the output file");
}

fn init_logger() {
    Builder::new()
        .filter_level(LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {}] {} {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.module_path().unwrap_or_else(|| "-".into()),
                record.level(),
                record.args()
            )
        })
        .init();
}
