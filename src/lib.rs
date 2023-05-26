use anyhow::Result;
use ash::{
	extensions::khr::Display,
	vk::{
		self, DisplayPlaneAlphaFlagsKHR, DisplaySurfaceCreateInfoKHR,
		PhysicalDevice, SurfaceKHR, SurfaceTransformFlagsKHR,
	},
	Device,
};

pub unsafe fn get_surfaces_for_displays(
	physical_device: PhysicalDevice,
	display_loader: &Display,
) -> Result<Vec<SurfaceKHR>> {
	let mut surfaces = vec![];

	let planes = display_loader
		.get_physical_device_display_plane_properties(physical_device)?;

	for plane in planes {
		let supported_displays = display_loader
			.get_display_plane_supported_displays(
				physical_device,
				plane.current_stack_index,
			)?;

		// TODO: show display picker
		let supported_display = supported_displays[0];

		let modes = display_loader
			.get_display_mode_properties(physical_device, supported_display)?;

		for mode in modes {
			let surface = {
				let create_info = DisplaySurfaceCreateInfoKHR {
					display_mode: mode.display_mode,
					alpha_mode: DisplayPlaneAlphaFlagsKHR::PER_PIXEL,
					transform: SurfaceTransformFlagsKHR::IDENTITY,
					..Default::default()
				};

				display_loader
					.create_display_plane_surface(&create_info, None)?
			};

			surfaces.push(surface);
		}
	}

	Ok(surfaces)
}

pub fn find_memorytype_index(
	memory_req: &vk::MemoryRequirements,
	memory_prop: &vk::PhysicalDeviceMemoryProperties,
	flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
	memory_prop.memory_types[..memory_prop.memory_type_count as _]
		.iter()
		.enumerate()
		.find(|(index, memory_type)| {
			(1 << index) & memory_req.memory_type_bits != 0
				&& memory_type.property_flags & flags == flags
		})
		.map(|(index, _memory_type)| index as _)
}

#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	command_buffer_reuse_fence: vk::Fence,
	submit_queue: vk::Queue,
	wait_mask: &[vk::PipelineStageFlags],
	wait_semaphores: &[vk::Semaphore],
	signal_semaphores: &[vk::Semaphore],
	f: F,
) -> Result<()> {
	unsafe {
		device.wait_for_fences(
			&[command_buffer_reuse_fence],
			true,
			std::u64::MAX,
		)?;

		device.reset_fences(&[command_buffer_reuse_fence])?;

		device.reset_command_buffer(
			command_buffer,
			vk::CommandBufferResetFlags::RELEASE_RESOURCES,
		)?;

		let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

		device
			.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
		f(device, command_buffer);
		device.end_command_buffer(command_buffer)?;

		let command_buffers = vec![command_buffer];

		let submit_info = vk::SubmitInfo::default()
			.wait_semaphores(wait_semaphores)
			.wait_dst_stage_mask(wait_mask)
			.command_buffers(&command_buffers)
			.signal_semaphores(signal_semaphores);

		device.queue_submit(
			submit_queue,
			&[submit_info],
			command_buffer_reuse_fence,
		)?;
	}

	Ok(())
}
