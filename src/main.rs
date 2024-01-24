#![allow(unused, dead_code)]
use gcode_bandage::utils::*;
use anyhow::Result;

fn main() -> Result<()> {

    let config = Config::read("config.toml")?;

    let gcode = std::fs::read_to_string("gcode.prg")?;

    let mut dom = DOM::parse(&gcode, &config)?;

    dom.add_tabs(&config)?;

    let gcode = dom.serialize();

    std::fs::write("tabbed.prg", &gcode)?;

    Ok(())
}

