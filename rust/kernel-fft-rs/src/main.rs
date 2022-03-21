use std::{
    error::Error,
    fs,
    io::{self, Write},
    ops::Range,
    path::{Path, PathBuf},
};

use image::{ImageBuffer, Rgba};
use rustfft::{num_complex::Complex, FftPlanner};

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
    if message != "" {
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
            print!("{}\n", &def);
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
    if message != "" {
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

#[allow(dead_code, unused_variables)]
fn image_to_color_vec2s(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    w: u32,
    h: u32,
) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn Error>> {
    let mut imgvec = vec![vec![vec![0.0f32; w as usize]; h as usize]; 4];
    for (x, y, pixel) in img.enumerate_pixels() {
        for i in 0..4 {
            imgvec[i][y as usize][x as usize] = pixel.0[i] as f32;
        }
    }
    Ok(imgvec)
}

fn to_complex(img: &Vec<Vec<f32>>, w: usize, h: usize) -> Vec<Vec<Complex<f32>>> {
    let mut new_img = vec![vec![Complex { re: 0.0, im: 0.0 }; w]; h];
    for y in 0..h {
        for x in 0..w {
            new_img[y][x] = Complex {
                re: img[y][x].clone() as f32,
                im: 0.0,
            };
        }
    }
    new_img
}

#[allow(dead_code, unused_variables)]
fn transpose_array(
    arr: &Vec<Vec<Complex<f32>>>,
    w: usize,
    h: usize,
    out: &mut Vec<Vec<Complex<f32>>>,
) {
    for y in 0..h {
        for x in 0..w {
            out[x][y] = arr[y][x];
        }
    }
}

#[allow(dead_code, unused_variables)]
fn fft_2d(
    img: &Vec<Vec<f32>>,
    w: usize,
    h: usize,
) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(w);
    let ffth = planner.plan_fft_forward(h);
    let mut fft_img = vec![vec![Complex { re: 0.0, im: 0.0 }; h]; w];
    let mut out_img = to_complex(img, w, h);

    for y in 0..h {
        let mut row = out_img[y].clone();
        fft.process(&mut row);
        out_img[y] = row;
    }

    transpose_array(&out_img, w, h, &mut fft_img);
    for x in 0..w {
        let mut row = fft_img[x].clone();
        ffth.process(&mut row);
        fft_img[x] = row;
    }

    transpose_array(&fft_img, h, w, &mut out_img);
    Ok(out_img)
}

#[allow(dead_code, unused_variables)]
fn ifft_2d(
    img: &Vec<Vec<Complex<f32>>>,
    w: usize,
    h: usize,
) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(w);
    let ffth = planner.plan_fft_forward(h);
    let mut fft_img = vec![vec![Complex { re: 0.0, im: 0.0 }; h]; w];
    let mut out_img = img.clone();

    transpose_array(&out_img, w, h, &mut fft_img);
    for x in 0..w {
        let mut row = fft_img[x].clone();
        ffth.process(&mut row);
        fft_img[x] = row;
    }

    transpose_array(&fft_img, h, w, &mut out_img);
    for y in 0..h {
        let mut row = out_img[y].clone();
        fft.process(&mut row);
        out_img[y] = row;
    }

    Ok(out_img)
}

#[allow(dead_code, unused_variables)]
fn normalize_fft(
    fft_img: &Vec<Vec<Complex<f32>>>,
    w: usize,
    h: usize,
    out_img: &mut Vec<Vec<Complex<f32>>>,
    scalar: f32,
) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for y in 0..h {
        for x in 0..w {
            let v = fft_img[y][x].re;
            if v > max {
                max = v;
            }
            if v < min {
                min = v;
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            out_img[y][x] = ((fft_img[y][x] - min) / (max - min)) * scalar;
        }
    }
}

#[allow(dead_code, unused_variables)]
fn channels_to_fft(
    channels: &Vec<Vec<Vec<f32>>>,
    w: usize,
    h: usize,
) -> Result<Vec<Vec<Vec<Complex<f32>>>>, Box<dyn Error>> {
    let mut fft_channels: Vec<Vec<Vec<Complex<f32>>>> = Vec::new();
    let mut float_channel = vec![vec![0.0f32; w]; h];
    for channel in channels {
        for y in 0..h {
            for x in 0..w {
                float_channel[y][x] = channel[y][x] as f32;
            }
        }
        let ftt = fft_2d(&float_channel, w, h)?;
        fft_channels.push(ftt);
    }
    Ok(fft_channels)
}

#[allow(dead_code, unused_variables)]
fn fft_to_channels(
    fft_channels: &Vec<Vec<Vec<Complex<f32>>>>,
    w: usize,
    h: usize,
) -> Result<Vec<Vec<Vec<u8>>>, Box<dyn Error>> {
    let mut channels: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut u8_channel = vec![vec![0u8; w]; h];
    let mut normalized_image = vec![vec![Complex { re: 0.0, im: 0.0 }; w as usize]; h as usize];
    //     normalize_fft(&ifft_image, w as usize, h as usize, &mut normalized_image, 255.0);
    for channel in fft_channels {
        let iftt = ifft_2d(&channel, w, h)?;
        normalize_fft(&iftt, w as usize, h as usize, &mut normalized_image, 255.0);
        for y in 0..h {
            for x in 0..w {
                u8_channel[y][x] = normalized_image[y][x].re.abs() as u8;
            }
        }
        channels.push(u8_channel.clone());
    }
    Ok(channels)
}

#[allow(dead_code, unused_variables)]
fn channels_to_image(
    channels: &Vec<Vec<Vec<u8>>>,
    w: usize,
    h: usize,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut out = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let mut col = [0u8; 4];
            for c in 0..4 {
                col[c] = channels[c][y][(w - x) % w];
            }
            out.put_pixel(x as u32, ((h - 1) - y) as u32, Rgba { 0: col });
        }
    }
    out
}

#[allow(dead_code, unused_variables)]
fn apply_kernel(
    channels: &Vec<Vec<Vec<f32>>>,
    kernel: &Vec<Vec<f32>>,
    w: usize,
    h: usize,
    kernel_scale: f32,
) -> Result<Vec<Vec<Vec<u8>>>, Box<dyn Error>> {
    let mut applied_channels = Vec::new();

    for k in 0..3 {
        let channel = channels[k].clone();
        let mut padded_kernel = vec![vec![0.0f32; w + 6]; h + 6];
        let mut padded_image = padded_kernel.clone();

        let mut j = 0;
        for row in kernel {
            let mut i = 0;
            for column in row {
                padded_kernel[j + 3][i + 3] = kernel[j][i].clone() * kernel_scale;
                i += 1;
            }
            j += 1;
        }

        for y in 0..h {
            for x in 0..w {
                padded_image[y + 3][x + 3] = channel[y][x].clone();
            }
        }

        let mut img_fft = fft_2d(&padded_image, w + 6, h + 6)?;
        let kernel_fft = fft_2d(&padded_kernel, w + 6, h + 6)?;

        for y in 0..h + 6 {
            for x in 0..w + 6 {
                img_fft[y][x] *= kernel_fft[y][x];
            }
        }

        applied_channels.push(img_fft);
    }

    let mut new_channels = Vec::new();
    for i in 0..3 {
        new_channels.push(ifft_2d(&applied_channels[i], w + 6, h + 6)?);
    }
    let mut output = fft_to_channels(&applied_channels, w + 6, h + 6)?;
    output.push(vec![vec![255; w]; h]);
    Ok(output)
}

#[allow(dead_code, unused_variables)]
fn main() -> Result<(), Box<dyn Error>> {
    let fname = input_prompt("input", FindType::File, "Please select an image")?;
    let image = image::open(fname)?;
    let (w, h) = (image.width(), image.height());
    let image = image.into_rgba8();
    let channels = image_to_color_vec2s(&image, w, h)?;
    // let fft_channels = channels_to_fft(&channels, w as usize, h as usize)?;
    // let new_channels = fft_to_channels(&fft_channels, w as usize, h as usize)?;
    // let new_img = channels_to_image(&new_channels, w as usize, h as usize);

    let kernel: Vec<Vec<f32>> = Vec::from([
        vec![1.0, 2.0, 1.0],
        vec![2.0, 4.0, 2.0],
        vec![1.0, 2.0, 1.0],
    ]);

    // let mut kernel: Vec<Vec<f32>> = Vec::from([
    //     vec![0.0, 0.0, 0.0],
    //     vec![1.0, 0.0,-1.0],
    //     vec![0.0, 0.0, 0.0],
    // ]);

    let new_channels = apply_kernel(&channels, &kernel, w as usize, h as usize, 1.0 / 16.0)?;
    let new_img = channels_to_image(&new_channels, w as usize, h as usize);

    let channels = image_to_color_vec2s(&new_img, w, h)?;

    let kernel: Vec<Vec<f32>> = Vec::from([
        vec![0.0, 1.0, 0.0],
        vec![1.0, -4.0, 1.0],
        vec![0.0, 1.0, 0.0],
    ]);

    let new_channels = apply_kernel(&channels, &kernel, w as usize, h as usize, 1.0)?;
    let new_img = channels_to_image(&new_channels, w as usize, h as usize);

    new_img.save("test.png")?;
    Ok(())
}
