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
    ops::Range,
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

// #[allow(dead_code, unused_variables)]
// fn apply_kernel(
//     channels: &Vec<Vec<Vec<f32>>>,
//     kernel: &Vec<Vec<f32>>,
//     w: usize,
//     h: usize,
//     kernel_scale: f32,
// ) -> Result<Vec<Vec<Vec<u8>>>, Box<dyn Error>> {
//     let mut applied_channels = Vec::new();

//     for k in 0..3 {
//         let channel = channels[k].clone();
//         let mut padded_kernel = vec![vec![0.0f32; w + 6]; h + 6];
//         let mut padded_image = padded_kernel.clone();

//         let mut j = 0;
//         for row in kernel {
//             let mut i = 0;
//             for column in row {
//                 padded_kernel[j + 3][i + 3] = kernel[j][i].clone() * kernel_scale;
//                 i += 1;
//             }
//             j += 1;
//         }

//         for y in 0..h {
//             for x in 0..w {
//                 padded_image[y + 3][x + 3] = channel[y][x].clone();
//             }
//         }

//         let mut img_fft = fft_2d(&padded_image, w + 6, h + 6)?;
//         let kernel_fft = fft_2d(&padded_kernel, w + 6, h + 6)?;

//         for y in 0..h + 6 {
//             for x in 0..w + 6 {
//                 img_fft[y][x] *= kernel_fft[y][x];
//             }
//         }

//         applied_channels.push(img_fft);
//     }

//     let mut new_channels = Vec::new();
//     for i in 0..3 {
//         new_channels.push(ifft_2d(&applied_channels[i], w + 6, h + 6)?);
//     }
//     let mut output = fft_to_channels(&applied_channels, w + 6, h + 6)?;
//     output.push(vec![vec![255; w]; h]);
//     Ok(output)
// }

struct FFT2Interface {
    fft_planner: FftPlanner<f32>,
    dct_planner: DctPlanner<f32>,
    size: (usize, usize),
    data: Vec<Vec<Complex<f32>>>,
    channels: usize,
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

            let fft = self.fft_planner.plan_fft(self.size.1, direction);
            self.data[i]
                .par_chunks_exact_mut(self.size.1)
                .for_each_with(&fft, |fft, row| {
                    fft.process(row);
                });
            self.size = (self.size.1, self.size.0);
        }
    }
    fn into_vec(self, scalar: f32) -> Vec<f32> {
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
            for i in 0..self.channels {
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
    fn apply_kernel(&mut self, kernel: &mut FFT2Interface) {
        kernel.pad((0, self.size.0), (0, self.size.1));
        self.pad((3, 3), (3, 3));

        kernel.fft2(FftDirection::Forward);
        self.fft2(FftDirection::Forward);
        if self.channels == kernel.channels {
            for (channel, rhs_channel) in self.data.iter_mut().zip(&kernel.data) {
                for (var, rhs_var) in channel.iter_mut().zip(rhs_channel) {
                    *var *= *rhs_var;
                }
            }
        } else {
            for channel in self.data.iter_mut() {
                for (var, rhs_var) in channel.iter_mut().zip(&kernel.data[0]) {
                    *var *= *rhs_var;
                }
            }
        }

        kernel.fft2(FftDirection::Inverse);
        self.fft2(FftDirection::Inverse);
    }
}

// // (z * xMax * yMax) + (y * xMax) + x
// struct FFT2Object {
//     fft_planner: FftPlanner<f32>,
//     dct_planner: DctPlanner<f32>,
//     scratch: Vec<Complex<f32>>,
//     size: (usize, usize),
//     data: Vec<Array2<Complex<f32>>>,
//     channels: usize,
// }

// #[allow(dead_code, unused_variables)]
// impl FFT2Object {
//     pub fn from_array(array: Vec<Array2<f32>>) -> FFT2Object {
//         let s = array[0].shape();
//         let mut data = Vec::new();
//         let channels = array.len();
//         for i in 0..channels {
//             data.push(Array2::<Complex<f32>>::zeros((s[0], s[1])))
//         }
//         for i in 0..channels {
//             for y in 0..s[1] {
//                 for x in 0..s[0] {
//                     data[i][[x, y]] = Complex {
//                         re: array[i][[x, y]] as f32,
//                         im: 0.0,
//                     }
//                 }
//             }
//         }
//         FFT2Object {
//             fft_planner: FftPlanner::<f32>::new(),
//             dct_planner: DctPlanner::<f32>::new(),
//             scratch: vec![Complex { re: 0.0, im: 0.0 }; s[0].min(s[1])],
//             size: (s[0], s[1]),
//             data: data,
//             channels: channels,
//         }
//     }
//     pub fn from_img(img: &DynamicImage) -> FFT2Object {
//         let s = (img.width() as usize, img.height() as usize);
//         let mut data = vec![
//             Array2::<Complex<f32>>::zeros(s),
//             Array2::<Complex<f32>>::zeros(s),
//             Array2::<Complex<f32>>::zeros(s),
//         ];
//         let img = img.to_rgba8();
//         for i in 0..3 {
//             for y in 0..s.1 {
//                 for x in 0..s.0 {
//                     data[i][[x, y]] = Complex {
//                         re: img.get_pixel(x as u32, y as u32).0[i].clone() as f32,
//                         im: 0.0,
//                     }
//                 }
//             }
//         }
//         FFT2Object {
//             fft_planner: FftPlanner::<f32>::new(),
//             dct_planner: DctPlanner::<f32>::new(),
//             scratch: vec![Complex { re: 0.0, im: 0.0 }; s.0.min(s.1)],
//             size: s,
//             data: data,
//             channels: 3,
//         }
//     }
//     fn get_normalized(
//         &self,
//         scale: f32,
//     ) -> Vec<ArrayBase<OwnedRepr<Complex<f32>>, Dim<[usize; 2]>>> {
//         let mut max = vec![f32::MIN; self.channels];
//         let mut min = vec![f32::MAX; self.channels];
//         for i in 0..self.channels {
//             self.data[i].iter().for_each(|value| {
//                 let v = value.norm_sqr();
//                 if v > max[i] {
//                     max[i] = v;
//                 }
//                 if v < min[i] {
//                     min[i] = v;
//                 }
//             });
//             min[i] = min[i].sqrt();
//             max[i] = max[i].sqrt();
//         }
//         let mut out = Vec::new();
//         for i in 0..self.channels {
//             out.push(((self.data[i].clone() - min[0]) / (max[0] - min[0])) * scale)
//         }
//         out
//     }
//     fn image(&self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
//         let scaled = self.get_normalized(255.0);
//         let mut out = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(self.size.0 as u32, self.size.1 as u32);
//         for (x, y, pixel) in out.enumerate_pixels_mut() {
//             let (i, j) = (x as usize, y as usize);
//             *pixel = if self.channels == 3 {
//                 Rgba {
//                     0: [
//                         scaled[0][[i, j]].re as u8,
//                         scaled[1][[i, j]].re as u8,
//                         scaled[2][[i, j]].re as u8,
//                         255,
//                     ],
//                 }
//             } else {
//                 Rgba {
//                     0: [
//                         scaled[0][[i, j]].re as u8,
//                         scaled[0][[i, j]].re as u8,
//                         scaled[0][[i, j]].re as u8,
//                         255,
//                     ],
//                 }
//             }
//         }
//         out
//     }
//     fn fft2(&mut self, direction: FftDirection) {
//         for i in 0..self.channels {
//             for d in match direction {
//                 FftDirection::Forward => [(false, self.size.0, false), (true, self.size.1, true)],
//                 FftDirection::Inverse => [(true, self.size.1, false), (false, self.size.0, true)],
//             } {
//                 if d.2 {
//                     self.data[i] = self.data[i].clone().reversed_axes();
//                 }
//                 let fft = self.fft_planner.plan_fft(d.1, direction);
//                 for mut row in (match d.0 {
//                     true => ArrayBase::rows_mut,
//                     false => ArrayBase::columns_mut,
//                 })(&mut self.data[i])
//                 {
//                     if let Some(r) = row.as_slice_mut() {
//                         fft.process_with_scratch(r, &mut self.scratch);
//                     } else {
//                         println!("LMAO");
//                     }
//                 }
//             }
//         }
//     }
//     fn dct2(&mut self, inverse: bool) {
//         let mut dct_scratch = vec![0.0f32; self.size.1 * self.size.0];
//         let mut new_data = Vec::new();
//         for i in 0..self.channels {
//             new_data.push(Array2::<f32>::zeros(self.size));
//             for y in 0..self.size.1 {
//                 for x in 0..self.size.0 {
//                     new_data[i][[x, y]] = self.data[i][[x, y]].re;
//                 }
//             }
//         }
//         if !inverse {
//             for i in 0..self.channels {
//                 for d in [(false, self.size.1), (true, self.size.0)] {
//                     let dct = self.dct_planner.plan_dct2(d.1);
//                     for mut row in (match d.0 {
//                         false => ArrayBase::rows_mut,
//                         true => ArrayBase::columns_mut,
//                     })(&mut new_data[i])
//                     {
//                         if let Some(r) = row.as_slice_mut() {
//                             dct.process_dct2_with_scratch(r, &mut dct_scratch);
//                         }
//                     }
//                 }
//             }
//         } else {
//             for i in 0..self.channels {
//                 for d in [(true, self.size.0), (false, self.size.1)] {
//                     let dct = self.dct_planner.plan_dct3(d.1);
//                     for mut row in (match d.0 {
//                         false => ArrayBase::rows_mut,
//                         true => ArrayBase::columns_mut,
//                     })(&mut new_data[i])
//                     {
//                         if let Some(r) = row.as_slice_mut() {
//                             dct.process_dct3_with_scratch(r, &mut dct_scratch);
//                         }
//                     }
//                 }
//             }
//         }
//         for i in 0..self.channels {
//             for y in 0..self.size.1 {
//                 for x in 0..self.size.0 {
//                     self.data[i][[x, y]] = Complex {
//                         re: new_data[i][[x, y]],
//                         im: 0.0,
//                     };
//                 }
//             }
//         }
//     }
//     fn pad(&mut self, count: usize) {
//         let new_size = (self.size.0 + (count * 2), self.size.1 + (count * 2));
//         let mut pad = Vec::new();
//         for i in 0..self.channels {
//             pad.push(Array2::<Complex<f32>>::zeros(new_size));
//         }
//         for i in 0..self.channels {
//             for y in 0..new_size.1 {
//                 for x in 0..new_size.0 {
//                     if (count..(new_size.0 - count)).contains(&x) {
//                         if (count..(new_size.1 - count)).contains(&y) {
//                             pad[i][[x, y]] = self.data[i][[x - count, y - count]];
//                         }
//                     }
//                 }
//             }
//         }

//         self.data = pad;
//         self.size = new_size;
//         self.scratch = vec![Complex { re: 0.0, im: 0.0 }; self.size.0.min(self.size.1)];
//         self.fft_planner = FftPlanner::<f32>::new();
//         self.dct_planner = DctPlanner::<f32>::new();
//     }
//     fn update(&mut self) {
//         self.scratch = vec![Complex { re: 0.0, im: 0.0 }; self.size.0.min(self.size.1)];
//         self.fft_planner = FftPlanner::<f32>::new();
//         self.dct_planner = DctPlanner::<f32>::new();
//     }
//     fn kernel(&mut self, kernel: Array2<f32>) {
//         let kshape = kernel.shape();
//         let mut kernel_padded = FFT2Object::from_array(vec![Array2::<f32>::zeros(self.size)]);
//         for j in 0..3 {
//             for i in 0..3 {
//                 kernel_padded.data[0][[i, j]] = Complex {
//                     re: kernel[[i, j]],
//                     im: 0.0,
//                 };
//             }
//         }
//         println!("ok2");
//         kernel_padded.pad(3);
//         self.pad(3);

//         kernel_padded.fft2(FftDirection::Forward);
//         self.fft2(FftDirection::Forward);
//         kernel_padded.image().save("testt.png");
//         self.image().save("testt2.png");
//         println!("{}:{}", self.size.0, self.size.1);
//         for i in 0..self.channels {
//             // let a = self.data[i].dot(&kernel_padded.data[0]);
//             for y in 0..self.size.1 {
//                 for x in 0..self.size.0 {
//                     self.data[i][[x, y]] = self.data[i][[x, y]] * kernel_padded.data[0][[x, y]];
//                 }
//             }
//         }
//         self.fft2(FftDirection::Inverse);
//     }
// }

// struct lab(f32, f32, f32);

#[allow(dead_code, unused_variables)]
fn main() -> Result<(), Box<dyn Error>> {
    let fname = input_prompt("input", FindType::File, "Please select an image")?;
    let image = image::open(fname)?;
    let (w, h) = (image.width(), image.height());
    // let s = (w as usize, h as usize);
    // println!("Opened");

    // let kernel_vec = Vec::from([
    //     vec![0.0, 0.0, 0.0],
    //     vec![0.0, 1.0, 0.0],
    //     vec![0.0, 0.0, 0.0],
    // ]);
    // let mut kernel = Array2::<f32>::zeros((3,3));
    // for i in 0..3 {
    //     for j in 0..3 {
    //         kernel[[j,i]] = kernel_vec[i][j];
    //     }
    // }
    println!("ok");
    let v = 1;
    let mut time = Duration::ZERO;
    // let (w,h) = (767,1535);
    // let (w, h) = (3073, 1537);
    // let (w, h) = (1024, 2048);
    // let mut rng = rand::thread_rng();
    for i in 0..v {
        // let mut data = FFT2Object::from_img(&image)
        // for e in data.iter_mut() {
        //     *e = Complex{re:rng.gen_range(0.0..255.0), im:rng.gen_range(0.0..255.0)};
        // }

        // let mut data = vec![0u8; w * h as usize];
        // for e in data.iter_mut() {
        //     *e = rng.gen_range(0u8..255u8);
        // }

        let data: Vec<f32> = image
            .clone()
            .to_rgb8()
            .into_vec()
            .iter()
            .map(|x| *x as f32)
            .collect();
        let kernel: Vec<f32> = vec![0., 0., 0., 0., 1., 0., 0., 0., 0.];
        let mut kernel = FFT2Interface::from_vec(&kernel, (3, 3), 1.0);
        let mut fft_thing = FFT2Interface::from_vec(&data, (w as usize, h as usize), 1.0);

        let now = Instant::now();
        fft_thing.apply_kernel(&mut kernel);
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
        // data.fft2(FftDirection::Forward);
        // let temp = data.clone();
        // transpose_inplace::<Complex<f32>>(&mut data, &mut scratch, w, h);
        // data.dct2();
        // data.kernel(kernel.clone());
        // let new_image = data.image();
        // new_image.save("FAST.png")?;
    }
    //.as_nanos() as f32 / (1_000_000.0 * v as f32)
    println!("{:?}", time / v);
    // println!("{}ms per fft", now.elapsed as f32 / (1000000.0 * v as f32));
    // new_image.save("test_NEW.png")?;
    // let mut test;
    // let s = (w as usize, h as usize);
    // let mut time = 0;
    // let mut rng = rand::thread_rng();
    // let v = 500;
    // let mut planner = FftPlanner::<f32>::new();
    // let direction = FftDirection::Forward;
    // let mut scratch = vec![Complex { re: 0.0, im: 0.0 }; w.min(h) as usize];

    // for i in 0..v {
    //     test = Array2::<Complex<f32>>::zeros(s);
    //     for ((x, y), value) in test.indexed_iter_mut() {
    //         *value = Complex {
    //             re: rng.gen_range(0.0..255.0),
    //             im: 0.0,
    //         };
    //     }
    //     let now = Instant::now();
    //     apply_2d_fft(&mut test, s, direction, &mut scratch, &mut planner);
    //     time += now.elapsed().as_nanos();
    // }
    // println!("{}ms per fft", time as f32 / (1000000.0 * v as f32));
    // let fft_channels = channels_to_fft(&channels, w as usize, h as usize)?;
    // let new_channels = fft_to_channels(&fft_channels, w as usize, h as usize)?;
    // let new_img = channels_to_image(&new_channels, w as usize, h as usize);

    // let kernel: Vec<Vec<f32>> = Vec::from([
    //     vec![1.0, 2.0, 1.0],
    //     vec![2.0, 4.0, 2.0],
    //     vec![1.0, 2.0, 1.0],
    // ]);
    // let now = Instant::now();
    // for i in 0..60 {
    //     let kernel: Vec<Vec<f32>> = Vec::from([
    //         vec![-1.0,-1.0,-1.0],
    //         vec![-1.0, 8.0,-1.0],
    //         vec![-1.0,-1.0,-1.0],
    //     ]);

    //     let new_channels = apply_kernel(&channels, &kernel, w as usize, h as usize, 1.0)?;
    //     let new_img = channels_to_image(&new_channels, w as usize, h as usize);
    // }
    // println!("Finished in: {}ms", now.elapsed().as_millis());
    // let channels = image_to_color_vec2s(&new_img, w, h)?;

    // let kernel: Vec<Vec<f32>> = Vec::from([
    //     vec![0.0, 1.0, 0.0],
    //     vec![1.0, -4.0, 1.0],
    //     vec![0.0, 1.0, 0.0],
    // ]);

    // let new_channels = apply_kernel(&channels, &kernel, w as usize, h as usize, 1.0)?;
    // let new_img = channels_to_image(&new_channels, w as usize, h as usize);

    // new_img.save("test4.png")?;
    Ok(())
}
