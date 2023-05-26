use anyhow::{Context as _, Result};
use ash::{
	util::Align,
	vk::{self, BufferUsageFlags, DeviceMemory, MemoryRequirements},
};

use crate::{context::HardwareContext, find_memorytype_index};

pub struct Buffer {
	pub buffer: vk::Buffer,
	pub size: u64,
	pub memory_requirements: MemoryRequirements,
	pub memory_index: u32,
	pub memory: DeviceMemory,
}

impl Buffer {
	pub unsafe fn new(
		hardware_context: &HardwareContext,
		kind: BufferUsageFlags,
		size: u64,
	) -> Result<Self> {
		let buffer_info = vk::BufferCreateInfo {
			size,
			usage: kind,
			sharing_mode: vk::SharingMode::EXCLUSIVE,
			..Default::default()
		};

		let buffer = hardware_context
			.device
			.create_buffer(&buffer_info, None)
			.unwrap();

		let memory_requirements = hardware_context
			.device
			.get_buffer_memory_requirements(buffer);

		let memory_index = find_memorytype_index(
			&memory_requirements,
			&hardware_context.device_memory_properties,
			vk::MemoryPropertyFlags::HOST_VISIBLE
				| vk::MemoryPropertyFlags::HOST_COHERENT,
		)
		.context("Unable to find suitable memorytype for the vertex buffer.")?;

		let allocate_info = vk::MemoryAllocateInfo {
			allocation_size: memory_requirements.size,
			memory_type_index: memory_index,
			..Default::default()
		};
		let memory = hardware_context
			.device
			.allocate_memory(&allocate_info, None)?;

		Ok(Self {
			buffer,
			memory_requirements,
			memory_index,
			memory,
			size,
		})
	}

	pub unsafe fn bind_data<T: Copy>(
		&self,
		hardware_context: &HardwareContext,
		data: &[T],
		alignment_size: u64,
	) -> Result<()> {
		let pointer = hardware_context.device.map_memory(
			self.memory,
			0,
			self.memory_requirements.size,
			vk::MemoryMapFlags::empty(),
		)?;

		let mut vert_align =
			Align::new(pointer, alignment_size, self.memory_requirements.size);
		vert_align.copy_from_slice(data);
		hardware_context.device.unmap_memory(self.memory);
		hardware_context.device.bind_buffer_memory(
			self.buffer,
			self.memory,
			0,
		)?;

		Ok(())
	}
}
