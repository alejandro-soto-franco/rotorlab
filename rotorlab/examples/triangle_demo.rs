//! 5-second 1080p mp4 of a tricolor triangle, animated only by clear-color hue cycling.
//!
//! Run with: `cargo run --release --example triangle_demo`
//! Output: `triangle_demo.mp4` in the current working directory.

use rotorlab::encode::FfmpegEncoder;
use rotorlab::render::frame::FrameRecorder;
use rotorlab::render::pipelines::triangle::{TrianglePipeline, TriangleVertexBuffer};
use rotorlab::render::render_target::HeadlessRenderTarget;
use rotorlab::render::{Device, Instance, RenderPass};
use std::path::Path;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FPS: u32 = 60;
const SECONDS: u32 = 5;

fn main() {
    let total_frames = FPS * SECONDS;
    let instance = Instance::new().expect("create instance");
    let device = Device::new(instance).expect("create device");
    let render_pass = RenderPass::new(device.clone()).expect("render pass");
    let target = HeadlessRenderTarget::new(device.clone(), WIDTH, HEIGHT).expect("target");
    let pipeline =
        TrianglePipeline::new(device.clone(), &render_pass, WIDTH, HEIGHT).expect("pipeline");
    let vbo = TriangleVertexBuffer::new(device.clone()).expect("vbo");
    let recorder = FrameRecorder::new(device.clone(), &render_pass, &target).expect("recorder");

    let mut encoder =
        FfmpegEncoder::new(Path::new("triangle_demo.mp4"), WIDTH, HEIGHT, FPS).expect("ffmpeg");
    let mut host = vec![0u8; (WIDTH * HEIGHT * 4) as usize];

    for f in 0..total_frames {
        let t = f as f32 / total_frames as f32;
        let r = 0.5 + 0.5 * (t * std::f32::consts::TAU).sin();
        let g = 0.5 + 0.5 * ((t + 0.33) * std::f32::consts::TAU).sin();
        let b = 0.5 + 0.5 * ((t + 0.66) * std::f32::consts::TAU).sin();
        let clear = [r * 0.2, g * 0.2, b * 0.2, 1.0];
        recorder
            .render_triangle(&render_pass, &pipeline, &vbo, &target, clear)
            .expect("render");
        target.read_to_host(&mut host).expect("readback");
        encoder.submit_frame(&host).expect("submit");
    }
    let n = encoder.finish().expect("finish");
    println!("wrote {n} frames to triangle_demo.mp4");
}
