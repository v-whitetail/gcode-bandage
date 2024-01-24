#![allow(unused, dead_code)]
#![feature(iter_intersperse)]
pub mod utils {
    use anyhow::{ Result, anyhow, bail, ensure, };
    use serde::{ Serialize, Deserialize, };
    use toml::{ ser::to_string, de::from_str, };
    use std::sync::Arc;
    use std::collections::BTreeMap;
    use std::ops::{ Range, RangeFrom, RangeInclusive, Sub, };
    use nom::{
        IResult,
        multi::*,
        branch::*,
        sequence::*,
        combinator::*,
        bytes::complete::*,
        character::complete::*,
    };


    #[derive(Debug, Serialize, Deserialize)]
    pub struct Size2D { length: f32, height: f32 }
    #[derive(Debug, Serialize, Deserialize)]
    pub struct Config {
        pub tab_size: Size2D,
        pub tab_quantity: u8,
        pub limit_x: RangeInclusive<f32>,
        pub limit_y: RangeInclusive<f32>,
        pub limit_z: RangeFrom<f32>,
        pub bottom_layer: RangeInclusive<f32>,
    }
    impl Default for Config {
        fn default() -> Self {
            let limit_x = (0.0..=47.5);
            let limit_y = (0.0..=143.5);
            let limit_z = (-0.0625..);
            let tab_size = Size2D{length:0.75, height:0.25};
            let tab_quantity = 2;
            let bottom_layer = (-0.0625..=0.0625);
            Self{
                limit_x,
                limit_y,
                limit_z,
                tab_size,
                tab_quantity,
                bottom_layer,
            }
        }
    }
    impl Config {
        pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self> {
            let config = from_str(&std::fs::read_to_string(path)?)?;
            Ok(config)
        }
        pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
            std::fs::write(path, &to_string(self)?)?;
            Ok(())
        }
    }



    #[derive(Debug, Clone, Copy)]
    pub struct Point3D { x: f32, y: f32, z: f32, }
    impl Default for Point3D{
        fn default() -> Self { Self{ x: 1.0, y: 1.0, z: 1.0 } }
    }
    impl Point3D {
        fn to_bottom_layer(&self, config: &Config) -> bool {
            config.bottom_layer.contains(&self.z)
        }
        fn to_higher_layer(&self, config: &Config) -> bool {
            !config.bottom_layer.contains(&self.z)
        }
        fn target_in_bounds(&self, config: &Config) -> bool {
            config.limit_x.contains(&self.x)
            && config.limit_y.contains(&self.y)
            && config.limit_z.contains(&self.z)
        }
        fn scale(&self, s: &f32) -> Self {
            Self {
                x: s * self.x,
                y: s * self.y,
                z: s * self.z,
            }
        }
        fn rotate_ccw(&self, a: &f32) -> Self {
            Self {
                x: (self.x * a.cos()) - (self.y * a.sin()),
                y: (self.x * a.sin()) + (self.y * a.cos()),
                z: self.z,
            }
        }
        fn rotate_cw(&self, a: &f32) -> Self {
            Self {
                x: (self.x * (-a).cos()) - (self.y * (-a).sin()),
                y: (self.x * (-a).sin()) + (self.y * (-a).cos()),
                z: self.z,
            }
        }
        fn add(&self, other: &Self) -> Self {
            Self {
                x: (other.x + self.x),
                y: (other.y + self.y),
                z: (other.z + self.z),
            }
        }
        fn unit_vector(&self, other: &Self) -> Self {
            let d_lin = self.linear_distance(other);
            Self {
                x: (other.x - self.x) / d_lin,
                y: (other.y - self.y) / d_lin,
                z: (other.z - self.z) / d_lin,
            }
        }
        fn unit_vector_perp_left(&self, other: &Self) -> Self {
            let unit_vector = self.unit_vector(other);
            Self{
                x:-unit_vector.y,
                y: unit_vector.x,
                z: unit_vector.z,
            }
        }
        fn unit_vector_perp_right(&self, other: &Self) -> Self {
            let unit_vector = self.unit_vector(other);
            Self{
                x: unit_vector.y,
                y:-unit_vector.x,
                z: unit_vector.z,
            }
        }
        fn arc_distance(&self, other: &Self, r: &f32) -> f32 {
            let d_lin = self.linear_distance(other);
            let theta = 2.0 * (d_lin / r).asin();
            r * theta
        }
        fn linear_distance(&self, other: &Self) -> f32 {
            let x_norm = (other.x - self.x).powi(2);
            let y_norm = (other.y - self.y).powi(2);
            let z_norm = (other.z - self.z).powi(2);
            (x_norm + y_norm + z_norm).sqrt()
        }
        fn linear_midpoint(&self, other: &Self) -> Self {
            let x = 0.5 * (other.x + self.x);
            let y = 0.5 * (other.y + self.y);
            let z = 0.5 * (other.z + self.z);
            Self{x, y, z}
        }
        fn linear_target_offset(&self, other: &Self, o: &f32) -> Self {
            let offset = self.linear_distance(other) - o;
            let d_lin = self.unit_vector(other).scale(&offset);
            self.add(&d_lin)
        }
        fn linear_source_offset(&self, other: &Self, o: &f32) -> Self {
            let d_lin = self.unit_vector(other).scale(o);
            self.add(&d_lin)
        }
        fn arc_center(&self, other: &Self, r: &f32) -> Self {
            let d_lin = self.linear_distance(other);
            let p_mid = self.linear_midpoint(other);
            let s_to_center = (r.powi(2) - 0.25*(d_lin.powi(2))).sqrt();
            let v_to_center = match r.is_sign_positive() {
                true => self.unit_vector_perp_left(other).scale(&s_to_center),
                false => self.unit_vector_perp_right(other).scale(&s_to_center),
            };
            p_mid.add(&v_to_center)
        }
        fn arc_midpoint(&self, other: &Self, r: &f32) -> Self {
            let d_lin = self.linear_distance(other);
            let p_mid = self.linear_midpoint(other);
            let s_to_mid = r - (r.powi(2) - 0.25*(d_lin.powi(2))).sqrt();
            let v_to_mid = match r.is_sign_positive() {
                true => self.unit_vector_perp_right(other).scale(&s_to_mid),
                false => self.unit_vector_perp_left(other).scale(&s_to_mid),
            };
            p_mid.add(&v_to_mid)
        }
        fn arc_target_offset(&self, other: &Self, r: &f32, o: &f32) -> Self {
            let theta = o / r;
            let center = self.arc_center(other, r);
            let v_to_offset = match r.is_sign_positive() {
                true => self.unit_vector(other).rotate_cw(&theta).scale(r),
                false => self.unit_vector(other).rotate_ccw(&theta).scale(r),
            };
            center.add(&v_to_offset)
        }
        fn arc_source_offset(&self, other: &Self, r: &f32, o: &f32) -> Self {
            let theta = o / r;
            let center = self.arc_center(other, r);
            let v_to_offset = match r.is_sign_positive() {
                true => other.unit_vector(self).rotate_ccw(&theta).scale(r),
                false => other.unit_vector(self).rotate_cw(&theta).scale(r),
            };
            center.add(&v_to_offset)
        }
    }
    #[derive(Debug, Clone, Copy)]
    pub struct DomKey { i: usize }
    impl DomKey {
        fn increment(mut self) -> Self { self.i += 10; self }
        fn to_arc(self) -> Arc<usize> { self.i.into() }
    }
    impl Default for DomKey{
        fn default() -> Self { Self{ i: 10 } }
    }
    #[derive(Default, Debug, Clone, Copy)]
    pub struct ParserIter { point: Point3D, key: DomKey }
    impl ParserIter {
        fn increment(mut self) -> Self {
            Self {point: self.point, key: self.key.increment()}
        }
    }



    #[derive(Debug, Clone)]
    pub struct DOM<'a> {
        pub line_tree: BTreeMap<Arc<usize>, Line<'a>>,
        pub positions: BTreeMap<Arc<usize>, Point3D>,
    }
    impl<'a> DOM<'a> {
        pub fn serialize(&self) -> String {
            self.line_tree
                .iter()
                .map( |(_, line)| line.raw.to_string() )
                .collect()
        }
        pub fn parse(s: &'a str, config: &Config) -> Result<Self> {
            let mut line_tree = BTreeMap::default();
            let mut positions = BTreeMap::default();
            fold_many1(
                terminated(Line::parse, line_ending),
                ParserIter::default,
                |mut iter, line| {
                    let source = iter.point;
                    let target = line.targets();
                    iter.point.x = target[0].unwrap_or(source.x);
                    iter.point.y = target[1].unwrap_or(source.y);
                    iter.point.z = target[2].unwrap_or(source.z);
                    let key = iter.key.to_arc();
                    line_tree.insert(key.clone(), line);
                    positions.insert(key.clone(), iter.point);
                    iter.increment()
                })(s)
            .map_err(|err| anyhow!("parse error: {:#?}", err))?;
            let dom = Self{line_tree, positions};
            dom.check_bounds(config);
            Ok(dom)
        }
        pub fn add_tabs(&mut self, config: &Config) -> Result<()> {
            let loops = (&*self).get_loops(config)?;
            for pass in loops {
                let mut pass = pass.clone();
                pass.sort_by_key(
                    |key| (&*self).distance(&key).unwrap_or(0.0) as isize
                    );
                for key in pass.iter().rev().take(config.tab_quantity as usize) {
                    match self.line_tree[key].move_type() {
                        Some(G::G01) => {
                            let source = self.positions[self.previous_key(key)?];
                            let target = self.positions[key];

                            let front_key= Arc::new(*key.clone()-3);
                            let peak_key = Arc::new(*key.clone()-2);
                            let back_key = Arc::new(*key.clone()-1);

                            let mut peak_target = source.linear_midpoint(&target);
                            let front_target = source.linear_target_offset(
                                &peak_target, &config.tab_size.length);
                            let back_target = peak_target.linear_source_offset(
                                &target, &config.tab_size.length);

                            peak_target.z = config.tab_size.height;

                            let front_line= self.line_tree[key].clone().retarget(&front_target,3);
                            let peak_line = self.line_tree[key].clone().retarget(&peak_target, 2);
                            let back_line = self.line_tree[key].clone().retarget(&back_target, 1);
                            self.line_tree.insert(front_key, front_line);
                            self.line_tree.insert(peak_key, peak_line);
                            self.line_tree.insert(back_key, back_line);
                        },
                        Some(G::G02) => {
//                            let source = self.positions[self.previous_key(key)?];
//                            let target = self.positions[key];
//                            let radius =-self.line_tree[key].move_radius().unwrap_or(0.001);
//
//                            let front_key= Arc::new(*key.clone()-3);
//                            let peak_key = Arc::new(*key.clone()-2);
//                            let back_key = Arc::new(*key.clone()-1);
//
//                            let mut peak_target = source.arc_midpoint(&target, &radius);
//                            let front_target = source.arc_target_offset(
//                                &peak_target, &radius, &config.tab_size.length);
//                            let back_target = peak_target.arc_source_offset(
//                                &target, &radius, &config.tab_size.length);
//
//                            peak_target.z = config.tab_size.height;
//
//                            let front_line= self.line_tree[key].clone().retarget(&front_target,3);
//                            let peak_line = self.line_tree[key].clone().retarget(&peak_target, 2);
//                            let back_line = self.line_tree[key].clone().retarget(&back_target, 1);
//                            self.line_tree.insert(front_key, front_line);
//                            self.line_tree.insert(peak_key, peak_line);
//                            self.line_tree.insert(back_key, back_line);
                        },
                        Some(G::G03) => {
//                            let source = self.positions[self.previous_key(key)?];
//                            let target = self.positions[key];
//                            let radius = self.line_tree[key].move_radius().unwrap_or(0.001);
//
//                            let front_key= Arc::new(*key.clone()-3);
//                            let peak_key = Arc::new(*key.clone()-2);
//                            let back_key = Arc::new(*key.clone()-1);
//
//                            let mut peak_target = source.arc_midpoint(&target, &radius);
//                            let front_target = source.arc_target_offset(
//                                &peak_target, &radius, &config.tab_size.length);
//                            let back_target = peak_target.arc_source_offset(
//                                &target, &radius, &config.tab_size.length);
//
//                            peak_target.z = config.tab_size.height;
//
//                            let front_line= self.line_tree[key].clone().retarget(&front_target,3);
//                            let peak_line = self.line_tree[key].clone().retarget(&peak_target, 2);
//                            let back_line = self.line_tree[key].clone().retarget(&back_target, 1);
//                            self.line_tree.insert(front_key, front_line);
//                            self.line_tree.insert(peak_key, peak_line);
//                            self.line_tree.insert(back_key, back_line);
                        },
                        _ => {}
                    }
                }
            }; Ok(())
        }
        pub fn distance(&self, key: &Arc<usize>) -> Result<f32> {
            let line = &self.line_tree[key];
            let prev_key = self.previous_key(key)?;
            let source = self.positions[prev_key];
            let target = self.positions[key];
            match line.move_type() {
                Some(G::G01) => {
                    return Ok(source.linear_distance(&target));
                },
//                Some(G::G02) => {
//                    let rad = line.move_radius();
//                    ensure!(rad.is_some(), "invalid radius at line {:?}", line);
//                    return Ok(source.arc_distance(&target, &rad.unwrap()));
//                },
//                Some(G::G03) => {
//                    let rad = line.move_radius();
//                    ensure!(rad.is_some(), "invalid radius at line {:?}", line);
//                    return Ok(source.arc_distance(&target, &rad.unwrap()));
//                },
                _ => { bail!("canot calculate distance at line {:?}", line); },
            }
        }
        pub fn get_loops(&self, config: &Config) -> Result<Vec<Vec<Arc<usize>>>> {
            let mut loops = Vec::new();
            let mut walker = self.positions.iter().peekable();
            while let Some(source) = walker.next() {
                if let Some(target) = walker.peek() {
                    let lhs = config.bottom_layer.contains(&source.1.z);
                    let rhs = config.bottom_layer.contains(&target.1.z);
                    match (lhs, rhs) {
                        (false, true) => {
                            let mut path = Vec::new();
                            path.push(target.0.clone());
                            loops.push(path);
                        },
                        (true, true) => {
                            let path = loops.last_mut();
                            ensure!(
                                path.is_some(),
                                "plunge to last pass was not found at line {:?}",
                                self.line_tree[source.0]
                                );
                            path.unwrap().push(target.0.clone());
                        },
                        _ => {},
                    }
                }
            }
            loops.retain( |pass| {
                let line = &self.line_tree[&pass[0]];
                !line.commands.contains(&Command::G(G::G00))
            });
            Ok(loops)
        }
        fn previous_key(&self, key: &Arc<usize>) -> Result<&Arc<usize>> {
            let prev_key = self.positions
                .range(..key.clone())
                .next_back();
            ensure!(prev_key.is_some(), "previous key not found from line {:?}", self.line_tree[key]);
            Ok(prev_key.unwrap().0)
        }
        fn check_bounds(&self, config: &Config) -> Result<()> {
            for (i, target) in self.positions.iter() {
                if !target.target_in_bounds(config) {
                    let line = &self.line_tree[i];
                    bail!("out of bounds at: {:#?}", line);
                };
            }; Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct Line<'a> {
        raw: Box<str>,
        commands: Box<[Command<'a>]>,
    }
    impl<'a> Line<'a> {
        pub fn parse(s: &'a str) -> IResult<&str, Self> {
            map(
                separated_list1(space1, Command::parse),
                |c| Self{
                    raw: s.split_inclusive('\n')
                        .next().unwrap_or(s).into(),
                    commands: c.into()
                }
               )(s)
        }
        pub fn retarget(&self, target: &Point3D, line_offset: u64) -> Self {
            let mut line_number = 0;
            let mut retarget = self.commands
                .into_iter()
                .filter_map( |command| match command {
                    Command::X(_) => None,
                    Command::Y(_) => None,
                    Command::Z(_) => None,
                    Command::N(n) => { line_number = n - line_offset; None },
                    _ => Some(command),
                }).cloned().collect::<Vec<_>>();
            retarget.push(Command::N(line_number));
            retarget.push(Command::X(target.x));
            retarget.push(Command::Y(target.y));
            retarget.push(Command::Z(target.z));
            retarget.push(Command::Comment("\r\n(^generated by gcode-bandage)\r\n"));
            retarget.sort();
            let commands = retarget.into_boxed_slice();
            let mut raw = commands
                .iter()
                .map( |cmd| cmd.serialize() )
                .intersperse(" ".into())
                .collect::<String>()
                .into();
            Self{raw, commands}
        }
        pub fn move_type(&self) -> Option<G> {
            for command in self.commands.iter() {
                match command {
                    Command::G(G::G01) => { return Some(G::G01); },
                    Command::G(G::G02) => { return Some(G::G02); },
                    Command::G(G::G03) => { return Some(G::G03); },
                    _ => {},
                };
            };
            return None;
        }
        pub fn move_radius(&self) -> Option<f32> {
            for command in self.commands.iter() {
                if let Command::R(radius) = command { return Some(*radius) };
            };
            return None;
        }
        pub fn z_target(&self) -> Option<&f32> {
            for command in self.commands.iter() {
                if let Command::G(G::G00) = command { return None; }
                if let Command::Z(z) = command { return Some(z); }
            };
            return None;
        }
        pub fn weak_targets(&self) -> Option<Box<[[&f32; 3]]>> {
            let targets = self.commands.iter().filter_map(
                |command| match command {
                    Command::X(x) => Some([x, &0.0, &0.0]),
                    Command::Y(y) => Some([&0.0, y, &0.0]),
                    Command::Z(z) => Some([&0.0, &0.0, z]),
                    _ => None,
                }).collect::<Box<[_]>>();
            if 0 < targets.len() { return Some(targets) };
            return None;
        }
        pub fn targets(&self) -> [Option<f32>;3] {
            self.commands
                .iter()
                .map( |c| match c {
                    Command::X(x) => [Some(x.clone()), None, None],
                    Command::Y(y) => [None, Some(y.clone()), None],
                    Command::Z(z) => [None, None, Some(z.clone())],
                    _ => [None, None, None],
                }).reduce( |lhs, rhs| [
                           lhs[0].or(rhs[0]),
                           lhs[1].or(rhs[1]),
                           lhs[2].or(rhs[2]),
                ]).unwrap_or_default()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Command<'a> {
        G(G),
        M(M),
        X(f32),
        Y(f32),
        Z(f32),
        F(f32), // Feed Rate
        T(u16), // Tool Index
        D(u16), // Tool Offset
        R(f32), // Curve Radius
        N(u64), // Command Index
        S(f32), // Spindle Speed
        Comment(&'a str),
    }
    impl<'a> Command<'a> {
        fn parse(s: &'a str) -> IResult<&str, Self> {
            let n_parser = |s| preceded(tag("N"), u64)(s);
            let s_parser = |s| preceded(tag("S"), f32)(s);
            let d_parser = |s| preceded(tag("D"), u16)(s);
            let t_parser = |s| preceded(tag("T"), u16)(s);
            let x_parser = |s| preceded(tag("X"), f32)(s);
            let y_parser = |s| preceded(tag("Y"), f32)(s);
            let z_parser = |s| preceded(tag("Z"), f32)(s);
            let f_parser = |s| preceded(tag("F"), f32)(s);
            let r_parser = |s| preceded(tag("R="),f32)(s);
            let comments = |s| not_line_ending(s);
            alt((
                    map(n_parser, |v| Self::N(v)), // Command Index
                    map(G::parse, |v| Self::G(v)),
                    map(M::parse, |v| Self::M(v)),
                    map(x_parser, |v| Self::X(v)), // X
                    map(y_parser, |v| Self::Y(v)), // Y
                    map(z_parser, |v| Self::Z(v)), // Z
                    map(r_parser, |v| Self::R(v)), // R
                    map(f_parser, |v| Self::F(v)), // Feed Rate
                    map(t_parser, |v| Self::T(v)), // Tool Index
                    map(d_parser, |v| Self::D(v)), // Tool Offset
                    map(s_parser, |v| Self::S(v)), // Spindle Speed
                    map(comments, |v| Self::Comment(v)),
                ))(s)
        }
        fn serialize(&self) -> Box<str> {
            match self {
                Self::G(g) => g.serialize(),
                Self::M(m) => m.serialize(),
                Self::X(call) => format!("X{:?}",call).into(),
                Self::Y(call) => format!("Y{:?}",call).into(),
                Self::Z(call) => format!("Z{:?}",call).into(),
                Self::F(call) => format!("F{:?}",call).into(),
                Self::T(call) => format!("T{:?}",call).into(),
                Self::D(call) => format!("D{:?}",call).into(),
                Self::R(call) => format!("R{:?}",call).into(),
                Self::N(call) => format!("N{:?}",call).into(),
                Self::S(call) => format!("S{:?}",call).into(),
                Self::Comment(call) => call.to_owned().into(),
            }
        }
        fn discriminant(&self) -> u8 {
            match self {
                Self::N(_) => 0,
                Self::G(_) => 1,
                Self::M(_) => 2,
                Self::X(_) => 10,
                Self::Y(_) => 11,
                Self::Z(_) => 12,
                Self::R(_) => 13,
                Self::F(_) => 20,
                Self::T(_) => 21,
                Self::D(_) => 22,
                Self::S(_) => 24,
                Self::Comment(_) => 255,
            }
        }
    }
    impl<'a> std::cmp::Eq for Command<'a> { }
    impl<'a> std::cmp::PartialOrd for Command<'a> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.discriminant().partial_cmp(&other.discriminant())
        }
    }
    impl<'a> std::cmp::Ord for Command<'a> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.discriminant().cmp(&other.discriminant())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum G {
        G00, // Rapid
        G01, // Linear
        G02, // Circular CW
        G03, // Circular CCW
        G70, // Units: in
        G71, // Units: mm
        Misc(u16),
    }
    impl G {
        fn parse(s: &str) -> IResult<&str, Self> {
            map( preceded(tag("G"), u16),
            |call| match call {
                00 => Self::G00,
                01 => Self::G01,
                02 => Self::G02,
                03 => Self::G03,
                70 => Self::G70,
                71 => Self::G71,
                __ => Self::Misc(call)
            })(s)
        }
        fn serialize(&self) -> Box<str> {
            match self {
                Self::G00 => "G00".into(),
                Self::G01 => "G01".into(),
                Self::G02 => "G02".into(),
                Self::G03 => "G03".into(),
                Self::G70 => "G70".into(),
                Self::G71 => "G71".into(),
                Self::Misc(call) => format!("G{:?}",call).into(),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum M {
        M03, // Spindle ON
        M05, // Spindle OFF
        M06, // Tool Change
        Misc(u16),
    }
    impl M {
        fn parse(s: &str) -> IResult<&str, Self> {
            map( preceded(tag("G"), u16),
            |call| match call {
                03 => Self::M03,
                05 => Self::M05,
                06 => Self::M06,
                __ => Self::Misc(call)
            })(s)
        }
        fn serialize(&self) -> Box<str> {
            match self{
                Self::M03 => "M03".into(),
                Self::M05 => "M05".into(),
                Self::M06 => "M06".into(),
                Self::Misc(call) => format!("M{:?}",call).into(),
            }
        }
    }

    fn f32(s: &str) -> IResult<&str, f32> {
        map(
            is_a(".-1234567890"),
            |s: &str| s.parse::<f32>().expect("invalid f32 format")
           )(s)
    }
}
