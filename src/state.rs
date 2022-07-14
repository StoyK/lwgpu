use wgpu::{
    Backends, Color, CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance,
    Limits, LoadOp, Operations, PowerPreference, PresentMode, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RequestAdapterOptions, Surface, SurfaceConfiguration, SurfaceError,
    TextureUsages, TextureViewDescriptor,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

pub struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
}

impl State {
    pub async fn new(window: &Window) -> Self {
        // Get the actual size of the window (without bar and OS buttons)
        let size = window.inner_size();
        // This represents a WGPU [Instance] and it serves as context for all of the other modules
        // All backends means - Vulkan, D3D12, OpenGL, WASM...
        let instance = Instance::new(Backends::all());
        // Create a [Surface] on the window to draw on (kind of like a sheet of paper on a desk)
        let surface = unsafe { instance.create_surface(window) };
        // An [Adapter] is a handle(software-hardware connection) to the GPU
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                // Meaning a low power GPU (like an integrated cpu one) or a high power GPU like a RTX 3090
                power_preference: PowerPreference::default(),
                // This doesnt create a surface, only makes sure that the GPU we're getting is compatible with the surface we have
                compatible_surface: Some(&surface),
                // Asks if we should force WGPU to use a software implementation of a GPU instead of the actual GPU
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // A [Device] is an open connection to the GPU that we have established using the aforementioned [Adapter]
        // A [Queue] represents the command buffer of the GPU a.k.a. the commands that the GPU will execute
        let (device, queue) = adapter
            .request_device(
                // A description of the device(GPU) we want from the computer
                &DeviceDescriptor {
                    // Not the acutal name of the GPU, the label is if we want to
                    // have some kind of name for debugging purposes
                    label: None,
                    // If we want to use some specific feature of a GPU, we need to mention it here
                    features: Features::empty(),
                    // Represents the graphical capability we want from a GPU
                    limits: if cfg!(target_arch = "wasm32") {
                        Limits::downlevel_webgl2_defaults()
                    } else {
                        Limits::default()
                    },
                },
                // Trace path for tracing (google if you dont know what tracing is)
                None,
            )
            .await
            .unwrap();

        // This config describes how the surface will create its [SurfaceTexture]s
        let config = SurfaceConfiguration {
            // The usage field describes how the textures will be used
            usage: TextureUsages::RENDER_ATTACHMENT,
            // The format describes how the textures will be stored on the GPU
            // with surface.get_supported_formats(&adapter) we get the best
            // and prefered format for our GPU
            format: surface.get_supported_formats(&adapter)[0],
            // The width and height describe the size of the textures
            width: size.width,
            height: size.height,
            // Present mode represents how to sync the [Surface] with the screen
            // [PresentMode::Fifo] will cap the fps to the refresh rate of the
            // monitor, a.k.a. VSync
            present_mode: PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config)
        }
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        // Get a frame(texture) to render to
        let output = self.surface.get_current_texture()?;
        // Get a [TextureView], because we want to be able to say
        // how the GPU should render textures
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());
        // Create a [CommandEncoder], because we need to be able to actually
        // send commands to the GPU. The encoder builds a command buffer that
        // the GPU expects
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Now we need to actually clear the screen, to do that we will use a
        // "render pass" or in simple terms a single execution of the render
        // pipeline that we've built
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        // The reason we drop the render_pass is because the command
        // encoder.begin_render_pass() borrows the encoder as &mut which
        // means that we can't call encoder.finish() until that borrow is given
        // back. So, we tell Rust to drop render_pass which holds a &mut to
        // the encoder
        drop(render_pass);

        // Call encoder[CommandEncoder].finish which returns a [CommmandBuffer]
        // and send it to the GPU
        self.queue.submit(std::iter::once(encoder.finish()));
        // Schedule the output[SurfaceTexture] to be displayed on the screen
        output.present();

        Ok(())
    }
}
