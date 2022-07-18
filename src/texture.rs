use std::num::NonZeroU32;

use anyhow::*;
use image::{DynamicImage, GenericImageView};
use wgpu::{
    AddressMode, Device, Extent3d, FilterMode, ImageCopyTexture, ImageDataLayout, Origin3d, Queue,
    Sampler, SamplerDescriptor, Texture as WgpuTexture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

pub struct Texture {
    pub texture: WgpuTexture,
    pub view: TextureView,
    pub sampler: Sampler,
}

impl Texture {
    pub fn from_image(
        device: &Device,
        queue: &Queue,
        img: &DynamicImage,
        label: Option<&str>,
    ) -> Result<Texture> {
        // Let's create a texture!
        // We take the bytes from an image file and load them into a [DynamicImage]
        // which is then converted into a Vec of rgba bytes. We also save the image's
        // dimensions for when we create the actual Texture.

        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        // In graphics, all textures are stored as 3D. To represent
        // ours as 2D we set the field depth_or_array_layers to 1
        let size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        // A diffuse texture is the most common kind of texture. It defines the color and pattern
        // of an object. Mapping the diffuse color is like painting an image on the surface of the
        // object. For example, if you want a wall to be made out of brick, you can choose an image
        // file with a photograph of bricks and put it on the object.
        let texture = device.create_texture(&TextureDescriptor {
            // Label for debugging purposes
            label,
            // The size of our texture
            size,
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
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &rgba,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            size,
        );

        // Now that we have data in our [Texture], we have to actually use it
        // The way we do that is using a [TextureView] and a [Sampler]

        // A [TextureView] contains a handle(view) to our texture and the
        // needed metadata for a [RenderPipeline] or a [BindGroup]
        let view = texture.create_view(&TextureViewDescriptor::default());
        // A [Sampler] works the same way as the color picker tool in programs
        // like Photoshop/GIMP. The sampler goes through the texture and tells
        // us what each color is
        let sampler = device.create_sampler(&SamplerDescriptor {
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

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    pub fn from_bytes(
        device: &Device,
        queue: &Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Texture> {
        let img = image::load_from_memory(bytes)?;
        Texture::from_image(device, queue, &img, Some(label))
    }
}
