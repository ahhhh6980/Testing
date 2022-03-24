#![allow(dead_code)]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use rand::Rng;
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use rustdct::{Dct3, DctPlanner};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::{
    error::Error,
    fs,
    io::{self, Write},
    ops::{Range, RangeBounds},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use transpose::transpose;
#[allow(dead_code)]
enum FindType {
    File,
    Dir,
}

fn list_dir<P: AsRef<Path>>(dir: P, find_dirs: FindType) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::<PathBuf>::new();
    for item in fs::read_dir(dir)? {
        let item = item?;
        match &find_dirs {
            FindType::File => {
                if item.file_type()?.is_file() {
                    files.push(item.path());
                }
            }
            FindType::Dir => {
                if item.file_type()?.is_dir() {
                    files.push(item.path());
                }
            }
        }
    }
    Ok(files)
}

fn prompt_number(bounds: Range<u32>, message: &str, def: i32) -> io::Result<u32> {
    let stdin = io::stdin();
    let mut buffer = String::new();
    // Tell the user to enter a value within the bounds
    if !message.is_empty() {
        if def >= 0 {
            println!(
                "{} in the range [{}:{}] (default: {})",
                message,
                bounds.start,
                bounds.end - 1,
                def
            );
        } else {
            println!(
                "{} in the range [{}:{}]",
                message,
                bounds.start,
                bounds.end - 1
            );
        }
    }
    buffer.clear();
    // Keep prompting until the user passes a value within the bounds
    Ok(loop {
        stdin.read_line(&mut buffer)?;
        print!("\r\u{8}");
        io::stdout().flush().unwrap();
        if let Ok(value) = buffer.trim().parse() {
            if bounds.contains(&value) {
                break value;
            }
        } else if def >= 0 {
            print!("\r\u{8}");
            println!("{}", &def);
            io::stdout().flush().unwrap();
            break def as u32;
        }
        buffer.clear();
    })
}

fn input_prompt<P: AsRef<Path>>(
    dir: P,
    find_dirs: FindType,
    message: &str,
) -> std::io::Result<PathBuf> {
    // Get files/dirs in dir
    let files = list_dir(&dir, find_dirs)?;
    // Inform the user that they will need to enter a value
    if !message.is_empty() {
        println!("{}", message);
    }
    // Enumerate the names of the files/dirs
    for (i, e) in files.iter().enumerate() {
        println!("{}: {}", i, e.display());
    }
    // This is the range of values they can pick
    let bound: Range<u32> = Range {
        start: 0,
        end: files.len() as u32,
    };
    // Return the path they picked
    Ok((&files[prompt_number(bound, "", -1)? as usize]).clone())
}

struct FFT2Interface {
    fft_planner: FftPlanner<f32>,
    dct_planner: DctPlanner<f32>,
    size: (usize, usize),
    data: Vec<Vec<Complex<f32>>>,
    channels: usize,
}

enum Align {
    Center,
    Front,
    Back,
}

fn next_power_2(value: u32) -> u32 {
    let mut value = value - 1;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value + 1
}

impl FFT2Interface {
    fn from_vec(data: &[f32], size: (usize, usize), divisor: f32) -> FFT2Interface {
        let channels = data.len() / (size.0 * size.1);
        let mut new_data = vec![vec![Complex::new(0.0, 0.0); 0]; channels];
        let mut j = 0;
        for item in data {
            j += 1;
            if j == channels {
                j = 0;
            }
            new_data[j].push(Complex {
                re: *item / divisor,
                im: 0.0,
            })
        }
        FFT2Interface {
            fft_planner: FftPlanner::new(),
            dct_planner: DctPlanner::new(),
            size,
            data: new_data,
            channels,
        }
    }
    fn fft2(&mut self, direction: FftDirection) {
        for i in 0..self.channels {
            let fft = self.fft_planner.plan_fft(self.size.0, direction);
            self.data[i]
                .par_chunks_exact_mut(self.size.0)
                .for_each_with(&fft, |fft, row| {
                    fft.process(row);
                });

            let mut new_data = vec![Complex::new(0.0, 0.0); self.size.0 * self.size.1];
            transpose::<Complex<f32>>(&self.data[i], &mut new_data, self.size.0, self.size.1);
            self.data[i] = new_data.clone();
            self.size = (self.size.1, self.size.0);

            let fft = self.fft_planner.plan_fft(self.size.0, direction);
            self.data[i]
                .par_chunks_exact_mut(self.size.0)
                .for_each_with(&fft, |fft, row| {
                    fft.process(row);
                });
        }
    }
    fn into_vec(self, scalar: f32) -> Vec<f32> {
        let mut out_vec = Vec::new();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for channel in self.data.iter() {
            for e in channel {
                let v = e.re;
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        // let (min, max) = (min.sqrt(), max.sqrt());
        for j in 0..self.size.0 * self.size.1 {
            for i in 0..self.channels {
                out_vec.push(
                    ((self.data[(i + 1) % self.channels][j].re - min) / (max - min)) * scalar,
                );
            }
        }
        out_vec
    }
    fn into_vec_copy_channels(self, channels: usize, scalar: f32) -> Vec<f32> {
        let mut out_vec = Vec::new();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for i in 0..self.channels {
            for j in 0..(self.size.0 * self.size.1) {
                let v = self.data[i][j].norm_sqr();
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        let (min, max) = (min.sqrt(), max.sqrt());
        for j in 0..self.size.0 * self.size.1 {
            for i in 0..(self.channels * channels) {
                out_vec.push(
                    ((self.data[(i + 1) % self.channels][j].re - min) / (max - min)) * scalar,
                );
            }
        }
        out_vec
    }
    fn pad(&mut self, x: (usize, usize), y: (usize, usize)) {
        let mut new_data =
            vec![vec![Complex::new(0.0, 0.0); (self.size.0 + x.0 + x.1) * y.0]; self.channels];

        for (channel, new_row) in self.data.iter().zip(&mut new_data) {
            for row in channel.chunks_exact(self.size.0) {
                new_row.extend(vec![Complex::new(0.0, 0.0); x.0].into_iter());
                new_row.extend_from_slice(row);
                new_row.extend(vec![Complex::new(0.0, 0.0); x.1].into_iter());
            }
        }
        for item in new_data.iter_mut() {
            item.extend(vec![
                Complex::new(0.0, 0.0);
                (self.size.0 + x.0 + x.1) * y.1
            ]);
        }
        self.data = new_data;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
        self.size = (self.size.0 + x.0 + x.1, self.size.1 + y.0 + y.1);
    }
    fn pad_to_square(&mut self, mode: Align) -> ((usize, usize), Align) {
        let old_size = self.size;
        let size_diff = (self.size.0 as i32 - self.size.1 as i32).abs() as usize;
        let resize = match mode {
            Align::Center => (size_diff / 2, size_diff / 2),
            Align::Front => (0, size_diff),
            Align::Back => (size_diff, 0),
        };
        if self.size.0 > self.size.1 {
            self.pad((0, 0), resize);
        } else {
            self.pad(resize, (0, 0));
        }
        (old_size, mode)
    }
    fn remove_square_padding(&mut self, method: ((usize, usize), Align)) {
        let size_diff = ((method.0 .0 as i32) - (method.0 .1 as i32)).abs() as usize;
        let paddings = match method.1 {
            Align::Center => (size_diff / 2, size_diff / 2),
            Align::Front => (0, size_diff),
            Align::Back => (size_diff, 0),
        };
        let (mut x_range, mut y_range) = (0..method.0 .0, 0..method.0 .1);
        if method.0 .0 > method.0 .1 {
            y_range = paddings.0..((self.size.0 * self.size.1) - paddings.1);
        } else {
            x_range = paddings.0..(self.size.0 - paddings.1);
        }
        let mut new_data = vec![Vec::<Complex<f32>>::new(); 4];
        for (channel, new_channel) in self.data.iter_mut().zip(&mut new_data) {
            for row in channel[y_range.clone()].chunks_exact_mut(method.0 .0) {
                new_channel.extend(&row.to_vec()[x_range.clone()]);
            }
        }
        self.data = new_data;
        self.size = method.0;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
    }
    fn crop(&mut self, offset: (usize, usize), size: (usize, usize)) {
        let (x_range, y_range) = (
            offset.0..(size.0 + offset.0),
            (offset.1 * self.size.0)..((offset.1 * self.size.0) + (self.size.0 * size.1)),
        );
        let mut new_data = vec![Vec::<Complex<f32>>::new(); 4];
        for (channel, new_channel) in self.data.iter_mut().zip(&mut new_data) {
            for row in channel[y_range.clone()].chunks_exact_mut(self.size.0) {
                new_channel.extend(&row.to_vec()[x_range.clone()]);
            }
        }
        self.data = new_data;
        self.size = size;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
    }
    fn crop_align(&mut self, mode: (Align, Align), size: (usize, usize)) {
        let offset = (
            match mode.0 {
                Align::Center => (self.size.0 / 2) - (size.0 / 2),
                Align::Front => 0,
                Align::Back => self.size.0 - size.0,
            },
            match mode.1 {
                Align::Center => (self.size.1 / 2) - (size.1 / 2),
                Align::Front => 0,
                Align::Back => self.size.1 - size.1,
            },
        );
        //
        self.crop(offset, size);
    }
    #[allow(unused_variables)]
    fn convolve(&mut self, kernel: &mut FFT2Interface) {
        kernel.pad((0, self.size.0 + 3), (0, self.size.1 + 3));
        self.pad((3, 3), (3, 3));

        kernel.fft2(FftDirection::Forward);
        self.fft2(FftDirection::Forward);
        for channel in self.data.iter_mut() {
            for (var, kernel_var) in channel.iter_mut().zip(&kernel.data[0]) {
                *var *= kernel_var;
            }
        }

        kernel.fft2(FftDirection::Inverse);
        self.fft2(FftDirection::Inverse);
    }
    fn deconvolve(&mut self, kernel: &mut FFT2Interface) {
        let dirac: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut dirac = FFT2Interface::from_vec(&dirac, (3, 3), 1.0);

        dirac.pad((0, self.size.0 - 3), (0, self.size.1 - 3));
        // kernel.pad((0, self.size.0 + 3), (0, self.size.1 + 3));
        // self.pad((3, 3), (3, 3));

        dirac.fft2(FftDirection::Forward);
        kernel.fft2(FftDirection::Forward);
        self.fft2(FftDirection::Forward);

        let s = 1.0 / ((self.size.0) as f32 * (self.size.1) as f32);

        for channel in self.data.iter_mut() {
            for ((var, kernel_var), dirac_var) in
                channel.iter_mut().zip(&kernel.data[0]).zip(&dirac.data[0])
            {
                *var *= dirac_var / (kernel_var + s);
            }
        }

        kernel.fft2(FftDirection::Inverse);
        self.fft2(FftDirection::Inverse);
    }
}

// struct lab(f32, f32, f32);

#[allow(dead_code, unused_variables)]
fn main() -> Result<(), Box<dyn Error>> {
    let fname = input_prompt("input", FindType::File, "Please select an image")?;
    let image = image::open(fname)?;
    let (w, h) = (image.width(), image.height());

    println!("ok");
    let v = 1;
    let mut time = Duration::ZERO;
    for i in 0..v {
        let data: Vec<f32> = image
            .clone()
            .to_rgb8()
            .into_vec()
            .iter()
            .map(|x| *x as f32)
            .collect();
        let kernel: Vec<f32> = vec![0., 1., 0., 1., -4., 1., 0., 1., 0.];
        let mut kernel = FFT2Interface::from_vec(&kernel, (3, 3), 1.0);
        let mut fft_thing = FFT2Interface::from_vec(&data, (w as usize, h as usize), 1.0);

        let now = Instant::now();
        // let method = fft_thing.pad_to_square(Align::Front);
        let flag = w != h;

        let mut method = ((w as usize, h as usize), Align::Front);
        if flag {
            method = fft_thing.pad_to_square(Align::Front);
        }
        fft_thing.convolve(&mut kernel);
        fft_thing.deconvolve(&mut kernel);
        // let c = 2f32.powf((w as f32).min(h as f32).log(2.0).floor());
        // fft_thing.crop((3, 3), method.0);
        // fft_thing.apply_kernel(&mut kernel);
        // fft_thing.square(Align::Front);
        // fft_thing.fft2(FftDirection::Forward);
        // fft_thing.fft2(FftDirection::Inverse);
        time += now.elapsed();

        let (w, h) = fft_thing.size;
        let new_data = fft_thing.into_vec(256.0);
        let new_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
            w as u32,
            h as u32,
            new_data.iter().map(|x| *x as u8).collect(),
        )
        .unwrap();
        new_image.save("omgwow.png")?;
    }
    println!("{:?}", time / v);
    Ok(())
}
