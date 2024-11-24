use std::io::Write;
use chrono::Local;
use env_logger::Builder;
use log::LevelFilter;

pub fn init_logging() {
    Builder::new()
        .filter_level(LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {}] {} {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.module_path().unwrap_or("-"),
                record.level(),
                record.args()
            )
        })
        .init();
}