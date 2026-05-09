//! End-to-end single-frame headless render integration test.

use rotorlab::render::frame::FrameRecorder;
use rotorlab::render::pipelines::triangle::{TrianglePipeline, TriangleVertexBuffer};
use rotorlab::render::render_target::HeadlessRenderTarget;
use rotorlab::render::{Device, Instance, RenderPass};

#[test]
fn renders_one_triangle_frame() {
    let Ok(instance) = Instance::new() else {
        eprintln!("skip: no vulkan");
        return;
    };
    let Ok(device) = Device::new(instance) else {
        eprintln!("skip: no device");
        return;
    };
    let render_pass = RenderPass::new(device.clone()).unwrap();
    let target = HeadlessRenderTarget::new(device.clone(), 320, 180).unwrap();
    let pipeline = TrianglePipeline::new(device.clone(), &render_pass, 320, 180).unwrap();
    let vbo = TriangleVertexBuffer::new(device.clone()).unwrap();
    let recorder = FrameRecorder::new(device.clone(), &render_pass, &target).unwrap();
    recorder
        .render_triangle(
            &render_pass,
            &pipeline,
            &vbo,
            &target,
            [0.05, 0.05, 0.07, 1.0],
        )
        .unwrap();
    let mut out = vec![0u8; 320 * 180 * 4];
    target.read_to_host(&mut out).unwrap();
    // Sanity: at least one pixel is not the clear color.
    let clear_b = (0.07 * 255.0) as u8;
    let any_nontrivial = out
        .chunks(4)
        .any(|p| p[2].saturating_sub(clear_b) > 20 || p[1] > 30 || p[0] > 30);
    assert!(any_nontrivial, "the frame should contain triangle pixels");
}
