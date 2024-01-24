#![allow(unused, dead_code)]
use rfd::*;
use anyhow::Result;
use gcode_bandage::utils::*;

fn main() -> Result<()> {

    let files = FileDialog::new()
        .set_title("Select PRG to process")
        .add_filter("PRG", &["prg"])
        .set_directory(std::env::current_dir()?)
        .pick_files()
        .expect("no files selected");

    for file in files {
        let result = process_file(&file);
        if let Err(err) = result {
            MessageDialog::new()
                .set_title("Bandage Error")
                .set_level(MessageLevel::Error)
                .set_buttons(MessageButtons::Ok)
                .set_description(err.to_string())
                .show();
        };
    }

    Ok(())
}

fn process_file(path: &std::path::PathBuf) -> Result<()> {

    let source_stem = path.file_stem().expect("dialog read error");
    let mut target_name = source_stem.to_os_string();
    target_name.push("-tabs.prg");
    let mut target_file = path.clone();
    target_file.set_file_name(target_name);

    let config = Config::read("config.toml")?;

    let gcode = std::fs::read_to_string(path)?;

    let mut dom = DOM::parse(&gcode, &config)?;

    dom.add_tabs(&config)?;

    let gcode = dom.serialize();

    std::fs::write(target_file, &gcode)?;

    Ok(())
}
