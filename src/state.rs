use std::num::NonZeroU32;

use crate::vertex::Vertex;
use image::GenericImageView;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AddressMode, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device,
    DeviceDescriptor, Extent3d, Face, Features, FilterMode, FragmentState, FrontFace,
    ImageCopyTexture, ImageDataLayout, IndexFormat, Instance, Limits, LoadOp, MultisampleState,
    Operations, Origin3d, PipelineLayoutDescriptor, PolygonMode, PowerPreference, PresentMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface,
    SurfaceConfiguration, SurfaceError, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        tex_coords: [0.4131759, 0.00759614],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        tex_coords: [0.0048659444, 0.43041354],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        tex_coords: [0.28081453, 0.949397],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        tex_coords: [0.85967, 0.84732914],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        tex_coords: [0.9414737, 0.2652641],
    }, // E
];

// Indices(coming from the word index, like in an array) is basically an
// approach, where we store unique values in the vertex buffer, which can
// hold a lot of vertices and instead of repeating these complex values
// that take up a lot of memory, we can just make an index buffer, where
// we index or refrence these values instead of duplicating data
const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

pub struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    num_indices: u32,
    diffuse_bind_group: BindGroup,
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
                    // Label for debugging purposes
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

        // Let's create a texture!
        // We take the bytes from an image file and load them into a [DynamicImage]
        // which is then converted into a Vec of rgba bytes. We also save the image's
        // dimensions for when we create the actual Texture.
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image =
            image::load_from_memory(diffuse_bytes).expect("Couldn't load image from memory");
        let diffuse_rgba = diffuse_image.to_rgba8();
        let dimensions = diffuse_image.dimensions();

        // In graphics, all textures are stored as 3D. To represent
        // ours as 2D we set the field depth_or_array_layers to 1
        let texture_size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        // A diffuse texture is the most common kind of texture. It defines the color and pattern
        // of an object. Mapping the diffuse color is like painting an image on the surface of the
        // object. For example, if you want a wall to be made out of brick, you can choose an image
        // file with a photograph of bricks and put it on the object.
        let diffuse_texture = device.create_texture(&TextureDescriptor {
            // Label for debugging purposes
            label: Some("Diffuse Texture"),
            // The size of our texture
            size: texture_size,
            // See: https://en.wikipedia.org/wiki/Mipmap
            mip_level_count: 1,
            // See: https://www.khronos.org/opengl/wiki/Multisampling
            sample_count: 1,
            // What dimensions is our texture in
            dimension: TextureDimension::D2,
            // Most images use an sRGBA format of u8 so we use that
            format: TextureFormat::Rgba8UnormSrgb,
            // TextureUsages::TEXTURE_BINDING tells WGPU we want to use the
            // texture in shaders. TextureUsages::COPY_DST tells WGPU we want
            // to be able to copy data from this texture in the future
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });

        // The [Texture] struct has no actual methods to interact with the data
        // that it stores. However, we can use the queue we created earlier to
        // load the texture in
        queue.write_texture(
            ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &diffuse_rgba,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        // Now that we have data in our [Texture], we have to actually use it
        // The way we do that is using a [TextureView] and a [Sampler]

        // A [TextureView] contains a handle(view) to our texture and the
        // needed metadata for a [RenderPipeline] or a [BindGroup]
        let diffuse_texture_view = diffuse_texture.create_view(&TextureViewDescriptor::default());
        // A [Sampler] works the same way as the color picker tool in programs
        // like Photoshop/GIMP. The sampler goes through the texture and tells
        // us what each color is
        let diffuse_sampler = device.create_sampler(&SamplerDescriptor {
            // The address_mode_* parameters determine what to do if
            // the sampler gets a texture coordinate that's outside
            // the texture itself
            // AddressMode::ClampToEdge will make any coordinates outside
            // the texture use the color of the nearest pixel before
            // going out of bounds
            // AddressMode::Repeat will repeat the same texture as the
            // texture coordinates go out of bounds
            // AddressMode::MirroredRepeat is AddressMode::Repeat, but
            // the image gets mirrored upside down
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            // The mag_filter and min_filter options describe what to
            // do when a fragment covers multiple pixels, or there are
            // multiple fragments for a single pixel. This often comes
            // into play when viewing a surface from up close or far away
            // FilterMode::Linear attempts to blend in fragments
            // FilterMode::Nearest makes in-between fragments choose the
            // color of the closest pixel
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // The [TextureView] and [Sampler] are cool as resources, but if
        // we can't use them anywhere they don't matter. That's where
        // [BindGroupLayout] and [PipelineLayout] enter
        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                // Label for debugging purposes
                label: Some("Texture Bind Group Layout"),
                // The actual resources our [BindGroup] will hold
                entries: &[
                    // The first entry is for our [TextureView]
                    BindGroupLayoutEntry {
                        // This binding is correspondant to @location(0)
                        // in our fragment shader
                        binding: 0,
                        // This tells WGPU that the [TextureView] and [Sampler]
                        // will only be available in the fragment shader
                        visibility: ShaderStages::FRAGMENT,
                        // This is the type of the binding
                        ty: BindingType::Texture {
                            // See: https://www.khronos.org/opengl/wiki/Multisampling
                            multisampled: false,
                            // The dimensions of our texture view
                            view_dimension: TextureViewDimension::D2,
                            // The type of numbers we are using to hold information
                            // about the texture
                            sample_type: TextureSampleType::Float {
                                // If this field is set to false we can't sample our texture
                                // using a [Sampler]
                                filterable: true, // <--- this
                            },
                        },
                        // If this value is Some() it indicates that this is
                        // an array of textures. It isn't so we're keeping it
                        // as None
                        count: None,
                    },
                    // The second entry is for our [Sampler]
                    BindGroupLayoutEntry {
                        // This binding is correspondant to @location(1)
                        // in our fragment shader
                        binding: 1,
                        // Again, these bindings will only be available in
                        // our fragment shader
                        visibility: ShaderStages::FRAGMENT,
                        // The type of binding we are holding
                        ty: BindingType::Sampler(SamplerBindingType::Filtering), // <--- and this should match
                        // Same thing as the above entry
                        count: None,
                    },
                ],
            });

        // With our [BindGroupLayout] done, we can actually create the [BindGroup]
        let diffuse_bind_group = device.create_bind_group(&BindGroupDescriptor {
            // Label for debugging purposes
            label: Some("Bind Group"),
            // The [BindGroupLayout] we created for the [BindGroup]
            layout: &texture_bind_group_layout,
            // The actual data the [BindGroup] will hold
            entries: &[
                // This is out [TextureView] at @location(0)
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&diffuse_texture_view),
                },
                // This is our [Sampler] at @location(1)
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&diffuse_sampler),
                },
            ],
        });

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
        // that can be used by shaders. These resources can be textures, samplers and bindings to buffers
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            // Label for debugging purposes
            label: Some("Render Pipeline Layout"),
            // [BindGroup]s that the pipeline uses
            bind_group_layouts: &[&texture_bind_group_layout],
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
                // Give the [VertexBufferLayout] we use if there is any
                buffers: &[Vertex::desc()],
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

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
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
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
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
