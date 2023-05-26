use ash::vk::IndexType;

use crate::{
	buffer::Buffer,
	context::{DrawContext, HardwareContext},
};

pub struct Model {
	pub vertex_buffer: Buffer,
	pub index_buffer: Buffer,
	pub index_data_len: u32,
}

impl Model {
	pub unsafe fn draw(
		&self,
		hardware_context: &HardwareContext,
		draw_context: &DrawContext,
	) {
		hardware_context.device.cmd_bind_vertex_buffers(
			draw_context.command_buffer,
			0,
			&[self.vertex_buffer.buffer],
			&[0],
		);
		hardware_context.device.cmd_bind_index_buffer(
			draw_context.command_buffer,
			self.index_buffer.buffer,
			0,
			IndexType::UINT32,
		);
		hardware_context.device.cmd_draw_indexed(
			draw_context.command_buffer,
			self.index_data_len,
			1,
			0,
			0,
			1,
		);
	}
}
