#![feature(offset_of)]

use std::{ffi::CStr, io::Cursor, mem::offset_of};

use anyhow::{Context, Ok, Result};
use ash::{
	extensions::{
		ext::DebugUtils,
		khr::{Display, Surface, Swapchain},
	},
	util::read_spv,
	vk::{self, BufferUsageFlags, ImageView},
	Device, Entry, Instance,
};
use suscomp::{
	context::{DrawContext, HardwareContext},
	find_memorytype_index, get_surfaces_for_displays,
	model::Model,
	record_submit_commandbuffer, Vertex,
};

fn main() -> Result<()> {
	unsafe {
		tracing_subscriber::fmt::init();

		let entry = Entry::load()?;
		let app_name = CStr::from_bytes_with_nul_unchecked(b"bruh");
		let appinfo = vk::ApplicationInfo::default()
			.application_name(app_name)
			.application_version(0)
			.engine_name(app_name)
			.engine_version(0)
			.api_version(vk::make_api_version(0, 1, 0, 0));

		let create_flags = vk::InstanceCreateFlags::default();

		let layer_names = [CStr::from_bytes_with_nul_unchecked(
			b"VK_LAYER_KHRONOS_validation\0",
		)];
		let layers_names_raw: Vec<*const i8> = layer_names
			.iter()
			.map(|raw_name| raw_name.as_ptr())
			.collect();

		let extension_names = vec![
			DebugUtils::NAME.as_ptr(),
			"VK_KHR_display\0".as_ptr() as *const _,
			"VK_KHR_surface\0".as_ptr() as *const _,
		];

		let create_info = vk::InstanceCreateInfo::default()
			.application_info(&appinfo)
			.enabled_layer_names(&layers_names_raw)
			.enabled_extension_names(&extension_names)
			.flags(create_flags);

		let instance: Instance = entry
			.create_instance(&create_info, None)
			.context("Instance creation error")?;

		let physical_devices = instance
			.enumerate_physical_devices()
			.context("Physical device error")?;

		let (physical_device, queue_family_index) = physical_devices
			.iter()
			.find_map(|physical_device| {
				instance
					.get_physical_device_queue_family_properties(
						*physical_device,
					)
					.iter()
					.enumerate()
					.map(|(index, _info)| (*physical_device, index))
					.next()
			})
			.context("Couldn't find suitable device.")?;
		let queue_family_index = queue_family_index as u32;
		let device_extension_names_raw = [Swapchain::NAME.as_ptr()];
		let features = vk::PhysicalDeviceFeatures {
			shader_clip_distance: 1,
			..Default::default()
		};

		let priorities = [1.0];

		let queue_info = vk::DeviceQueueCreateInfo::default()
			.queue_family_index(queue_family_index)
			.queue_priorities(&priorities);

		let device_create_info = vk::DeviceCreateInfo::default()
			.queue_create_infos(std::slice::from_ref(&queue_info))
			.enabled_extension_names(&device_extension_names_raw)
			.enabled_features(&features);

		let device: Device = instance.create_device(
			physical_device,
			&device_create_info,
			None,
		)?;

		let display_loader = Display::new(&entry, &instance);
		let surfaces =
			get_surfaces_for_displays(physical_device, &display_loader)?;

		let surface = surfaces[0];

		let surface_loader = Surface::new(&entry, &instance);

		let present_queue = device.get_device_queue(queue_family_index, 0);

		let surface_format = surface_loader
			.get_physical_device_surface_formats(physical_device, surface)?[0];

		let surface_capabilities = surface_loader
			.get_physical_device_surface_capabilities(
				physical_device,
				surface,
			)?;
		let mut desired_image_count = surface_capabilities.min_image_count + 1;
		if surface_capabilities.max_image_count > 0
			&& desired_image_count > surface_capabilities.max_image_count
		{
			desired_image_count = surface_capabilities.max_image_count;
		}
		let surface_resolution = match surface_capabilities.current_extent.width
		{
			std::u32::MAX => vk::Extent2D {
				width: 1920,
				height: 1080,
			},
			_ => surface_capabilities.current_extent,
		};
		let pre_transform = if surface_capabilities
			.supported_transforms
			.contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
		{
			vk::SurfaceTransformFlagsKHR::IDENTITY
		} else {
			surface_capabilities.current_transform
		};
		let present_modes = surface_loader
			.get_physical_device_surface_present_modes(
				physical_device,
				surface,
			)?;
		let present_mode = present_modes
			.iter()
			.cloned()
			.find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
			.unwrap_or(vk::PresentModeKHR::FIFO);
		let swapchain_loader = Swapchain::new(&instance, &device);

		let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
			.surface(surface)
			.min_image_count(desired_image_count)
			.image_color_space(surface_format.color_space)
			.image_format(surface_format.format)
			.image_extent(surface_resolution)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
			.pre_transform(pre_transform)
			.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
			.present_mode(present_mode)
			.clipped(true)
			.image_array_layers(1);

		let swapchain =
			swapchain_loader.create_swapchain(&swapchain_create_info, None)?;

		let pool_create_info = vk::CommandPoolCreateInfo::default()
			.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
			.queue_family_index(queue_family_index);

		let pool = device.create_command_pool(&pool_create_info, None)?;

		let command_buffer_allocate_info =
			vk::CommandBufferAllocateInfo::default()
				.command_buffer_count(2)
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY);

		let command_buffers =
			device.allocate_command_buffers(&command_buffer_allocate_info)?;
		let setup_command_buffer = command_buffers[0];
		let draw_command_buffer = command_buffers[1];

		let present_images =
			swapchain_loader.get_swapchain_images(swapchain)?;
		let present_image_views: Vec<vk::ImageView> = present_images
			.iter()
			.map(|&image| {
				let create_view_info = vk::ImageViewCreateInfo::default()
					.view_type(vk::ImageViewType::TYPE_2D)
					.format(surface_format.format)
					.components(vk::ComponentMapping {
						r: vk::ComponentSwizzle::R,
						g: vk::ComponentSwizzle::G,
						b: vk::ComponentSwizzle::B,
						a: vk::ComponentSwizzle::A,
					})
					.subresource_range(vk::ImageSubresourceRange {
						aspect_mask: vk::ImageAspectFlags::COLOR,
						base_mip_level: 0,
						level_count: 1,
						base_array_layer: 0,
						layer_count: 1,
					})
					.image(image);
				Ok(device.create_image_view(&create_view_info, None)?)
			})
			.collect::<Result<Vec<ImageView>>>()?;

		let hardware_context = HardwareContext {
			device: &device,
			physical_device: &physical_device,
			device_memory_properties: instance
				.get_physical_device_memory_properties(physical_device),
		};

		let depth_image_create_info = vk::ImageCreateInfo::default()
			.image_type(vk::ImageType::TYPE_2D)
			.format(vk::Format::D16_UNORM)
			.extent(surface_resolution.into())
			.mip_levels(1)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.tiling(vk::ImageTiling::OPTIMAL)
			.usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
			.sharing_mode(vk::SharingMode::EXCLUSIVE);

		let depth_image =
			device.create_image(&depth_image_create_info, None)?;
		let depth_image_memory_req =
			device.get_image_memory_requirements(depth_image);
		let depth_image_memory_index = find_memorytype_index(
			&depth_image_memory_req,
			&hardware_context.device_memory_properties,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)
		.context("Unable to find suitable memory index for depth image.")?;

		let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
			.allocation_size(depth_image_memory_req.size)
			.memory_type_index(depth_image_memory_index);

		let depth_image_memory =
			device.allocate_memory(&depth_image_allocate_info, None)?;

		device
			.bind_image_memory(depth_image, depth_image_memory, 0)
			.context("Unable to bind depth image memory")?;

		let fence_create_info = vk::FenceCreateInfo::default()
			.flags(vk::FenceCreateFlags::SIGNALED);

		let draw_commands_reuse_fence = device
			.create_fence(&fence_create_info, None)
			.context("Create fence failed.")?;
		let setup_commands_reuse_fence = device
			.create_fence(&fence_create_info, None)
			.context("Create fence failed.")?;

		record_submit_commandbuffer(
			&device,
			setup_command_buffer,
			setup_commands_reuse_fence,
			present_queue,
			&[],
			&[],
			&[],
			|device, setup_command_buffer| {
				let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                        .image(depth_image)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1)
                                .level_count(1),
                        );

				device.cmd_pipeline_barrier(
					setup_command_buffer,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
					vk::DependencyFlags::empty(),
					&[],
					&[],
					&[layout_transition_barriers],
				);

				Ok(())
			},
		)?;

		let depth_image_view_info = vk::ImageViewCreateInfo::default()
			.subresource_range(
				vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::DEPTH)
					.level_count(1)
					.layer_count(1),
			)
			.image(depth_image)
			.format(depth_image_create_info.format)
			.view_type(vk::ImageViewType::TYPE_2D);

		let depth_image_view =
			device.create_image_view(&depth_image_view_info, None)?;

		let semaphore_create_info = vk::SemaphoreCreateInfo::default();

		let present_complete_semaphore =
			device.create_semaphore(&semaphore_create_info, None)?;
		let rendering_complete_semaphore =
			device.create_semaphore(&semaphore_create_info, None)?;

		let renderpass_attachments = [
			vk::AttachmentDescription {
				format: surface_format.format,
				samples: vk::SampleCountFlags::TYPE_1,
				load_op: vk::AttachmentLoadOp::CLEAR,
				store_op: vk::AttachmentStoreOp::STORE,
				final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
				..Default::default()
			},
			vk::AttachmentDescription {
				format: vk::Format::D16_UNORM,
				samples: vk::SampleCountFlags::TYPE_1,
				load_op: vk::AttachmentLoadOp::CLEAR,
				initial_layout:
					vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				..Default::default()
			},
		];
		let color_attachment_refs = [vk::AttachmentReference {
			attachment: 0,
			layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
		}];
		let depth_attachment_ref = vk::AttachmentReference {
			attachment: 1,
			layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};
		let dependencies = [vk::SubpassDependency {
			src_subpass: vk::SUBPASS_EXTERNAL,
			src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
			dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
				| vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
			dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
			..Default::default()
		}];

		let subpass = vk::SubpassDescription::default()
			.color_attachments(&color_attachment_refs)
			.depth_stencil_attachment(&depth_attachment_ref)
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

		let renderpass_create_info = vk::RenderPassCreateInfo::default()
			.attachments(&renderpass_attachments)
			.subpasses(std::slice::from_ref(&subpass))
			.dependencies(&dependencies);

		let renderpass =
			device.create_render_pass(&renderpass_create_info, None)?;

		let framebuffers: Vec<vk::Framebuffer> = present_image_views
			.iter()
			.map(|&present_image_view| {
				let framebuffer_attachments =
					[present_image_view, depth_image_view];
				let frame_buffer_create_info =
					vk::FramebufferCreateInfo::default()
						.render_pass(renderpass)
						.attachments(&framebuffer_attachments)
						.width(surface_resolution.width)
						.height(surface_resolution.height)
						.layers(1);

				device
					.create_framebuffer(&frame_buffer_create_info, None)
					.unwrap()
			})
			.collect();
		let index_buffer_data = &[0u32, 1, 2];
		let index_buffer = suscomp::buffer::Buffer::new(
			&hardware_context,
			BufferUsageFlags::INDEX_BUFFER,
			std::mem::size_of_val(index_buffer_data) as u64,
		)?;
		index_buffer.bind_data(
			&hardware_context,
			index_buffer_data,
			std::mem::align_of::<u32>() as u64,
		)?;

		let vertex_buffer = suscomp::buffer::Buffer::new(
			&hardware_context,
			BufferUsageFlags::VERTEX_BUFFER,
			3 * std::mem::size_of::<Vertex>() as u64,
		)?;
		vertex_buffer.bind_data(
			&hardware_context,
			&[
				Vertex {
					position: [-1.0, 1.0, 0.0, 1.0],
					color: [0.0, 1.0, 0.0, 1.0],
				},
				Vertex {
					position: [1.0, 1.0, 0.0, 1.0],
					color: [0.0, 0.0, 1.0, 1.0],
				},
				Vertex {
					position: [0.0, -1.0, 0.0, 1.0],
					color: [1.0, 0.0, 0.0, 1.0],
				},
			],
			std::mem::align_of::<Vertex>() as u64,
		)?;

		let model = Model {
			vertex_buffer,
			index_buffer,
			index_data_len: 3,
		};

		let mut vertex_shader_file =
			Cursor::new(&include_bytes!("../shader/triangle.vert.spv")[..]);
		let mut frag_shader_file =
			Cursor::new(&include_bytes!("../shader/triangle.frag.spv")[..]);

		let vertex_shader_code = read_spv(&mut vertex_shader_file)
			.expect("Failed to read vertex shader spv file");
		let vertex_shader_info =
			vk::ShaderModuleCreateInfo::default().code(&vertex_shader_code);

		let frag_code = read_spv(&mut frag_shader_file)
			.context("Failed to read fragment shader spv file")?;
		let frag_shader_info =
			vk::ShaderModuleCreateInfo::default().code(&frag_code);

		let vertex_shader_module = device
			.create_shader_module(&vertex_shader_info, None)
			.context("Vertex shader module error")?;

		let fragment_shader_module = device
			.create_shader_module(&frag_shader_info, None)
			.context("Fragment shader module error")?;

		let layout_create_info = vk::PipelineLayoutCreateInfo::default();

		let pipeline_layout =
			device.create_pipeline_layout(&layout_create_info, None)?;

		let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
		let shader_stage_create_infos = [
			vk::PipelineShaderStageCreateInfo {
				module: vertex_shader_module,
				p_name: shader_entry_name.as_ptr(),
				stage: vk::ShaderStageFlags::VERTEX,
				..Default::default()
			},
			vk::PipelineShaderStageCreateInfo {
				s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
				module: fragment_shader_module,
				p_name: shader_entry_name.as_ptr(),
				stage: vk::ShaderStageFlags::FRAGMENT,
				..Default::default()
			},
		];
		let vertex_input_binding_descriptions =
			[vk::VertexInputBindingDescription {
				binding: 0,
				stride: std::mem::size_of::<Vertex>() as u32,
				input_rate: vk::VertexInputRate::VERTEX,
			}];
		let vertex_input_attribute_descriptions = [
			vk::VertexInputAttributeDescription {
				location: 0,
				binding: 0,
				format: vk::Format::R32G32B32A32_SFLOAT,
				offset: offset_of!(Vertex, position) as u32,
			},
			vk::VertexInputAttributeDescription {
				location: 1,
				binding: 0,
				format: vk::Format::R32G32B32A32_SFLOAT,
				offset: offset_of!(Vertex, color) as u32,
			},
		];

		let vertex_input_state_info =
			vk::PipelineVertexInputStateCreateInfo::default()
				.vertex_attribute_descriptions(
					&vertex_input_attribute_descriptions,
				)
				.vertex_binding_descriptions(
					&vertex_input_binding_descriptions,
				);
		let vertex_input_assembly_state_info =
			vk::PipelineInputAssemblyStateCreateInfo {
				topology: vk::PrimitiveTopology::TRIANGLE_LIST,
				..Default::default()
			};
		let viewports = [vk::Viewport {
			x: 0.0,
			y: 0.0,
			width: surface_resolution.width as f32,
			height: surface_resolution.height as f32,
			min_depth: 0.0,
			max_depth: 1.0,
		}];
		let scissors = [surface_resolution.into()];
		let viewport_state_info =
			vk::PipelineViewportStateCreateInfo::default()
				.scissors(&scissors)
				.viewports(&viewports);

		let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
			front_face: vk::FrontFace::COUNTER_CLOCKWISE,
			line_width: 1.0,
			polygon_mode: vk::PolygonMode::FILL,
			..Default::default()
		};
		let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
			rasterization_samples: vk::SampleCountFlags::TYPE_1,
			..Default::default()
		};
		let noop_stencil_state = vk::StencilOpState {
			fail_op: vk::StencilOp::KEEP,
			pass_op: vk::StencilOp::KEEP,
			depth_fail_op: vk::StencilOp::KEEP,
			compare_op: vk::CompareOp::ALWAYS,
			..Default::default()
		};
		let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
			depth_test_enable: 1,
			depth_write_enable: 1,
			depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
			front: noop_stencil_state,
			back: noop_stencil_state,
			max_depth_bounds: 1.0,
			..Default::default()
		};
		let color_blend_attachment_states =
			[vk::PipelineColorBlendAttachmentState {
				blend_enable: 0,
				src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
				dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
				color_blend_op: vk::BlendOp::ADD,
				src_alpha_blend_factor: vk::BlendFactor::ZERO,
				dst_alpha_blend_factor: vk::BlendFactor::ZERO,
				alpha_blend_op: vk::BlendOp::ADD,
				color_write_mask: vk::ColorComponentFlags::RGBA,
			}];
		let color_blend_state =
			vk::PipelineColorBlendStateCreateInfo::default()
				.logic_op(vk::LogicOp::CLEAR)
				.attachments(&color_blend_attachment_states);

		let dynamic_state =
			[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
		let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
			.dynamic_states(&dynamic_state);

		let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
			.stages(&shader_stage_create_infos)
			.vertex_input_state(&vertex_input_state_info)
			.input_assembly_state(&vertex_input_assembly_state_info)
			.viewport_state(&viewport_state_info)
			.rasterization_state(&rasterization_info)
			.multisample_state(&multisample_state_info)
			.depth_stencil_state(&depth_state_info)
			.color_blend_state(&color_blend_state)
			.dynamic_state(&dynamic_state_info)
			.layout(pipeline_layout)
			.render_pass(renderpass);

		let graphics_pipelines = device
			.create_graphics_pipelines(
				vk::PipelineCache::null(),
				&[graphic_pipeline_info],
				None,
			)
			.expect("Unable to create graphics pipeline");

		let graphic_pipeline = graphics_pipelines[0];

		loop {
			let (present_index, _) = swapchain_loader.acquire_next_image(
				swapchain,
				std::u64::MAX,
				present_complete_semaphore,
				vk::Fence::null(),
			)?;
			let clear_values = [
				vk::ClearValue {
					color: vk::ClearColorValue {
						float32: [0.0, 0.0, 0.0, 0.0],
					},
				},
				vk::ClearValue {
					depth_stencil: vk::ClearDepthStencilValue {
						depth: 1.0,
						stencil: 0,
					},
				},
			];

			let render_pass_begin_info = vk::RenderPassBeginInfo::default()
				.render_pass(renderpass)
				.framebuffer(framebuffers[present_index as usize])
				.render_area(surface_resolution.into())
				.clear_values(&clear_values);

			record_submit_commandbuffer(
				&device,
				draw_command_buffer,
				draw_commands_reuse_fence,
				present_queue,
				&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
				&[present_complete_semaphore],
				&[rendering_complete_semaphore],
				|device, draw_command_buffer| {
					device.cmd_begin_render_pass(
						draw_command_buffer,
						&render_pass_begin_info,
						vk::SubpassContents::INLINE,
					);
					device.cmd_bind_pipeline(
						draw_command_buffer,
						vk::PipelineBindPoint::GRAPHICS,
						graphic_pipeline,
					);
					device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
					device.cmd_set_scissor(draw_command_buffer, 0, &scissors);

					let draw_context = DrawContext {
						command_buffer: draw_command_buffer,
					};

					model.draw(&hardware_context, &draw_context);

					device.cmd_end_render_pass(draw_command_buffer);

					Ok(())
				},
			)?;

			let wait_semaphors = [rendering_complete_semaphore];
			let swapchains = [swapchain];
			let image_indices = [present_index];
			let present_info = vk::PresentInfoKHR::default()
				.wait_semaphores(&wait_semaphors)
				.swapchains(&swapchains)
				.image_indices(&image_indices);

			swapchain_loader.queue_present(present_queue, &present_info)?;
		}
	}
}
