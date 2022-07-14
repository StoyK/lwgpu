use wgpu::{
    Backends, BlendState, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device,
    DeviceDescriptor, Face, Features, FragmentState, FrontFace, Instance, Limits, LoadOp,
    MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode, PowerPreference,
    PresentMode, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError,
    TextureUsages, TextureViewDescriptor, VertexState,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

pub struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
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

        // A [ShaderModule] represents a compiled shader program
        // that we include through the file "shader.wgsl"
        // NOTE: All of this can be replaced with a macro like so:
        // ```
        // let shader = device.create_shader_module(include_wgsl!("shader.wgsl"))
        // ```
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            // Label for debugging purposes
            label: Some("Shader"),
            // The source of the file a.k.a. where it's located related to "state.rs"
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // A [PipelineLayout] represents the mapping and placing in memory between all [BindGroup]s
        // [BindGroup]s are groups of resources bound together in a group (unexpected, right?)
        // these resources can be textures, samplers(encoders for transformations) and bindings
        // to buffers
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            // Label for debugging purposes
            label: Some("Render Pipeline Layout"),
            // [BindGroup]s that the pipeline uses
            bind_group_layouts: &[],
            // [PushConstantRange] is a set of push constants, which are small amounts of data
            // we can send directly to a [ShaderModule] and any of its stages(Vertex, Fragment...)
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            // Label for debugging purposes
            label: Some("Render Pipeline"),
            // Give the render pipeline a [PipelineLayout]
            layout: Some(&render_pipeline_layout),
            // Define a [VertexState] or a description of our vertex shader
            vertex: VertexState {
                // Give the actual variable holding the [ShaderModule]
                module: &shader,
                // Define the name(entry_point) of the function for the vertex shader
                // in the file
                entry_point: "vs_main",
                // Give the [VertexBufferLayout](buffers of vertices) we use if there are any
                // In our case we have defined them in "shader.wgsl" so there is no reason to
                // add any here
                buffers: &[],
            },
            // Define a [FragmentState] or a description of our fragment shader
            // It's technically optional, so we wrap it in Some()
            fragment: Some(FragmentState {
                // Give the actual variable holding the [ShaderModule]
                module: &shader,
                // Define the name(entry_point) of the function for the fragment
                // shader in the file
                entry_point: "fs_main",
                // A [ColorTargetState] defines how the GPU should render colors
                targets: &[Some(ColorTargetState {
                    // The [TextureFormat] for the [ColorTargetState] must be the same
                    // as the [TextureFormat] used in the [CommandEncoder]'s begin_render_pass function
                    // That's why we are using the [TextureFormat] from the config we previously used
                    format: config.format,
                    // The [BlendState] defines how colors from previous renders should blend, in our
                    // case we don't want them to blend so we tell the GPU to just replace the old colors
                    blend: Some(BlendState::REPLACE),
                    // The [ColorWrites] or write mask tells the GPU if it should ignore any parts of the
                    // RGBA colors, in our case we want to use all colors possible so we don't use a color mask
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // The [PrimitiveState] describes how the GPU should handle converting
            // our vertices to triangles
            primitive: PrimitiveState {
                // The [PrimitiveTopology] describes how to group different vertices
                // For example:
                // PrimitiveTopology::PointList tells the GPU to treat every verticie as a new point
                // PrimitiveTopology::TriangleList tells the GPU to treat vertices as a group of 3
                // a.k.a. triangles
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // [FrontFace] tells the GPU how to determine if a triangle is front facing or not
                // FrontFace::Ccw means that a triangle is front facing if the vertices of said
                // triangle are aranged in a counter-clockwise direction
                front_face: FrontFace::Ccw,
                // [Face] cull_mode tells the GPU that triangles that are not front facing are
                // culled or said in simpler terms - not included in the rendering
                cull_mode: Some(Face::Back),
                // [PolygonMode] tells the GPU how to render polygons either Fill, Line or Point
                // Fill renders the vertices, the lines between the vertices and what is inside the actual polygon
                // Line renders the vertices and the lines between the vertices
                // Point only renders the vertices
                polygon_mode: PolygonMode::Fill,
                // When rasterization occurs, the GPU has to decide what shapes are above others in a depth scale
                // unclipped_depth unclamps the depth for polygons from only 0-1 to any numbers
                // Enabling this requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // When rasterization occurs if this option is selected the GPU will overestimate which pixels
                // should be filled and which not. This only works with PolygonMode::Fill!
                // Enabling this requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            // We are not using a depth/stencil buffer so we don't need this for now
            depth_stencil: None,
            // Multisampling/Anti-aliasing are very complex topics: https://www.youtube.com/watch?v=pFKalA-fd34
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
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
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
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

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw(0..3, 0..1);
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
