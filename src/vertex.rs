use bytemuck::{Pod, Zeroable};
use wgpu::{BufferAddress, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
/// A [Vertex] is a data structure that describes attributes for a point in 3D(or 2D) space.
/// It can contain position, color, reflectance, Texture(UV) coordinates and other
/// attributes
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// A [VertexBufferLayout] defines how a buffer is represented in memory.
    /// Without this, the [RenderPipeline] has no idea how to map the buffer in the shader.
    /// This function describes what a buffer full of [Vertex] would look like.
    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            // Array stride is how distanced are vertexes withing memory
            // Since we want them to be sequential and don't want any gaps
            // we say that every element is distanced by how big the element is
            // or in simpler terms that there isn't any space between vertices
            array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
            // Step mode describes how often the vertex buffer goes one "step"
            // a.k.a. stride forward, we want to stride forward after every vertex
            step_mode: VertexStepMode::Vertex,
            // Here we describe how the different attributes are positioned in memory
            // These are the same as the attributes we described in the [Vertex] struct
            // as well as the actual compiled "shader.wgsl"
            attributes: &[
                VertexAttribute {
                    // We say that the first attribute start without offset
                    offset: 0,
                    // This tells the shader what location to store this attribute at.
                    // For example @location(0) x: vec3<f32> in the vertex shader would
                    // correspond to the position field of the [Vertex] struct, while
                    // @location(1) x: vec3<f32> would be the color field.
                    shader_location: 0,
                    // The format field tells the shader the shape of the attribute
                    // Float32x3 corresponds to vec3<f32> in shader code. The max value
                    // we can store in an attribute is Float32x4 (Uint32x4, and Sint32x4 work as well)
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    // We say that the second attributes start one [f32; 3] forward in memory
                    // where [f32; 3] directly corresponds to the position field
                    offset: std::mem::size_of::<[f32; 3]>() as BufferAddress,
                    // Again, in the shader code @location(0) x: vec3<f32> would be the position field
                    // So @location(1) y: vec3<f32> would be the color field.
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
            ],
        }
    }
}
