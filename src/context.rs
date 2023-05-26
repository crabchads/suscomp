use ash::{
	vk::{CommandBuffer, PhysicalDevice, PhysicalDeviceMemoryProperties},
	Device,
};

pub struct HardwareContext<'a> {
	pub device: &'a Device,
	pub physical_device: &'a PhysicalDevice,
	pub device_memory_properties: PhysicalDeviceMemoryProperties,
}

#[derive(Debug)]
pub struct DrawContext {
	pub command_buffer: CommandBuffer,
}
