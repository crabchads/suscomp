#![allow(incomplete_features)]
#![feature(unsized_locals, unsized_fn_params)]

use anyhow::{bail, Context as _, Result};
use smithay::{
	reexports::wayland_server::ListeningSocket,
	wayland::{
		compositor::CompositorState, shell::xdg::XdgShellState, shm::ShmState,
	},
};
use std::sync::Arc;
use suscomp::{
	handle_compositor_events, load_texture, textured_quad, RenderContext,
	TexturedVertex, WaylandCompositor,
};
use tracing::{debug, warn};
use vulkano::{
	buffer::{Buffer, BufferCreateInfo, BufferUsage},
	command_buffer::{
		allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
		CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
	},
	descriptor_set::{
		allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
		WriteDescriptorSet,
	},
	device::{
		physical::PhysicalDeviceType, Device, DeviceCreateInfo,
		DeviceExtensions, Features, QueueCreateInfo, QueueFlags,
	},
	image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
	instance::{Instance, InstanceCreateInfo, InstanceExtensions},
	memory::allocator::{
		AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator,
	},
	pipeline::{
		graphics::{
			color_blend::ColorBlendState,
			input_assembly::InputAssemblyState,
			multisample::MultisampleState,
			rasterization::RasterizationState,
			vertex_input::{Vertex, VertexDefinition},
			viewport::{Viewport, ViewportState},
			GraphicsPipelineCreateInfo,
		},
		layout::PipelineDescriptorSetLayoutCreateInfo,
		GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
	},
	render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
	sampler::{Sampler, SamplerCreateInfo},
	shader::PipelineShaderStageCreateInfo,
	swapchain::{
		acquire_next_image,
		display::{Display, DisplayPlane},
		AcquireError, Surface, Swapchain, SwapchainCreateInfo,
		SwapchainCreationError, SwapchainPresentInfo,
	},
	sync::{self, FlushError, GpuFuture},
	Version, VulkanLibrary,
};

fn window_size_dependent_setup(
	images: &[Arc<SwapchainImage>],
	render_pass: Arc<RenderPass>,
	viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
	let dimensions = images[0].dimensions().width_height();
	viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

	images
		.iter()
		.map(|image| {
			let view = ImageView::new_default(image.clone()).unwrap();

			Framebuffer::new(
				render_pass.clone(),
				FramebufferCreateInfo {
					attachments: vec![view],
					..Default::default()
				},
			)
			.unwrap()
		})
		.collect::<Vec<_>>()
}

fn main() -> Result<()> {
	tracing_subscriber::fmt::init();

	let library = VulkanLibrary::new()?;

	let instance = Instance::new(
		library,
		InstanceCreateInfo {
			enabled_extensions: InstanceExtensions {
				khr_display: true,
				khr_surface: true,
				..InstanceExtensions::empty()
			},
			enumerate_portability: true,
			..Default::default()
		},
	)?;

	let mut device_extensions = DeviceExtensions {
		khr_swapchain: true,
		..DeviceExtensions::empty()
	};

	let (physical_device, queue_family_index) = instance
		.enumerate_physical_devices()?
		.filter(|p| {
			p.api_version() >= Version::V1_3
				|| p.supported_extensions().khr_dynamic_rendering
		})
		.filter(|p| p.supported_extensions().contains(&device_extensions))
		.filter_map(|p| {
			p.queue_family_properties()
				.iter()
				.enumerate()
				.position(|(_i, q)| {
					q.queue_flags.intersects(QueueFlags::GRAPHICS)
				})
				.map(|i| (p, i as u32))
		})
		.min_by_key(|(p, _)| match p.properties().device_type {
			PhysicalDeviceType::DiscreteGpu => 0,
			PhysicalDeviceType::IntegratedGpu => 1,
			PhysicalDeviceType::VirtualGpu => 2,
			PhysicalDeviceType::Cpu => 3,
			PhysicalDeviceType::Other => 4,
			_ => 5,
		})
		.context("Couldn't find a physical device")?;

	debug!(
		"Using device: {} (type: {:?})",
		physical_device.properties().device_name,
		physical_device.properties().device_type,
	);

	if physical_device.api_version() < Version::V1_3 {
		device_extensions.khr_dynamic_rendering = true;
	}

	let (device, mut queues) = Device::new(
		physical_device.clone(),
		DeviceCreateInfo {
			enabled_extensions: device_extensions,
			enabled_features: Features {
				dynamic_rendering: true,
				..Features::empty()
			},
			queue_create_infos: vec![QueueCreateInfo {
				queue_family_index,
				..Default::default()
			}],
			..Default::default()
		},
	)?;

	let display = Display::enumerate(physical_device.clone())
		.next()
		.context("Could not find a suitable display")?;
	let display_mode = display.display_modes().next().unwrap();
	let display_plane =
		DisplayPlane::enumerate(physical_device).next().unwrap();
	let surface = Surface::from_display_plane(&display_mode, &display_plane)?;

	let queue = queues.next().unwrap();

	let (mut swapchain, images) = {
		let surface_capabilities = device
			.physical_device()
			.surface_capabilities(&surface, Default::default())?;

		let image_format = Some(
			device
				.physical_device()
				.surface_formats(&surface, Default::default())?[0]
				.0,
		);
		debug!("image format: {:?}", image_format);

		Swapchain::new(
			device.clone(),
			surface,
			SwapchainCreateInfo {
				min_image_count: surface_capabilities.min_image_count,
				image_format,
				image_extent: display.physical_resolution(),
				image_usage: ImageUsage::COLOR_ATTACHMENT,
				composite_alpha: surface_capabilities
					.supported_composite_alpha
					.into_iter()
					.next()
					.unwrap(),

				..Default::default()
			},
		)?
	};

	let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

	let (vertices, indices) = textured_quad(2.0, 2.0);
	let vertex_buffer = Buffer::from_iter(
		&memory_allocator,
		BufferCreateInfo {
			usage: BufferUsage::VERTEX_BUFFER,
			..Default::default()
		},
		AllocationCreateInfo {
			usage: MemoryUsage::Upload,
			..Default::default()
		},
		vertices,
	)?;

	let index_buffer = Buffer::from_iter(
		&memory_allocator,
		BufferCreateInfo {
			usage: BufferUsage::INDEX_BUFFER,
			..Default::default()
		},
		AllocationCreateInfo {
			usage: MemoryUsage::Upload,
			..Default::default()
		},
		indices.clone(),
	)?;

	mod vs {
		vulkano_shaders::shader! {
			ty: "vertex",
			src: r"
				#version 450
				layout(location = 0) in vec4 position;
				layout(location = 1) in vec2 tex_coords;

				layout(location = 0) out vec2 f_tex_coords;

				void main() {
					gl_Position = position;
					f_tex_coords = tex_coords;
				}
			",
		}
	}

	mod fs {
		vulkano_shaders::shader! {
			ty: "fragment",
			src: r"
				#version 450
				layout(location = 0) in vec2 v_tex_coords;

				layout(location = 0) out vec4 f_color;

				layout(set = 0, binding = 0) uniform sampler2D tex;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0) + texture(tex, v_tex_coords);
				}
			",
		}
	}

	let render_pass = vulkano::single_pass_renderpass!(device.clone(),
		attachments: {
			color: {
				load: Clear,
				store: Store,
				format: swapchain.image_format(),
				samples: 1,
			},
		},
		pass: {
			color: [color],
			depth_stencil: {},
		},
	)?;

	let pipeline = {
		let vs = vs::load(device.clone())?.entry_point("main").unwrap();
		let fs = fs::load(device.clone())?.entry_point("main").unwrap();

		let vertex_input_state = TexturedVertex::per_vertex()
			.definition(&vs.info().input_interface)?;

		let stages = [
			PipelineShaderStageCreateInfo::entry_point(vs),
			PipelineShaderStageCreateInfo::entry_point(fs),
		];

		let layout = PipelineLayout::new(
			device.clone(),
			PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
				.into_pipeline_layout_create_info(device.clone())?,
		)?;

		let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

		GraphicsPipeline::new(
			device.clone(),
			None,
			GraphicsPipelineCreateInfo {
				stages: stages.into_iter().collect(),
				vertex_input_state: Some(vertex_input_state),
				input_assembly_state: Some(InputAssemblyState::default()),
				viewport_state: Some(
					ViewportState::viewport_dynamic_scissor_irrelevant(),
				),
				rasterization_state: Some(RasterizationState::default()),
				multisample_state: Some(MultisampleState::default()),
				color_blend_state: Some(
					ColorBlendState::new(subpass.num_color_attachments())
						.blend_alpha(),
				),
				subpass: Some(subpass.into()),
				..GraphicsPipelineCreateInfo::layout(layout)
			},
		)?
	};

	let dimensions = display.physical_resolution();

	let mut viewport = Viewport {
		origin: [0.0, 0.0],
		dimensions: [0.0, 0.0],
		depth_range: 0.0..1.0,
	};

	let mut attachment_image_views = window_size_dependent_setup(
		&images,
		render_pass.clone(),
		&mut viewport,
	);

	let command_buffer_allocator =
		StandardCommandBufferAllocator::new(device.clone(), Default::default());

	let mut recreate_swapchain = false;
	let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

	let mut wayland_display = smithay::reexports::wayland_server::Display::<
		WaylandCompositor,
	>::new()?;
	let wayland_display_handle = wayland_display.handle();
	let compositor_state =
		CompositorState::new::<WaylandCompositor>(&wayland_display_handle);
	let shm_state =
		ShmState::new::<WaylandCompositor>(&wayland_display_handle, vec![]);
	let socket = ListeningSocket::bind("wayland-0")?;

	let mut wayland_state = WaylandCompositor {
		compositor_state,
		shm_state,
		xdg_shell_state: XdgShellState::new::<WaylandCompositor>(
			&wayland_display_handle,
		),
		clients: Vec::new(),
	};

	let mut render_context = RenderContext {
		command_buffer_allocator,
		memory_allocator,
		queue,
		device,
	};

	let texture = load_texture(&mut render_context, [1280, 720])?;

	let sampler = Sampler::new(
		render_context.device.clone(),
		SamplerCreateInfo::simple_repeat_linear(),
	)
	.unwrap();

	let descriptor_set = {
		let layout = pipeline.layout().set_layouts().get(0).unwrap();
		let descriptor_set_allocator =
			StandardDescriptorSetAllocator::new(render_context.device.clone());

		PersistentDescriptorSet::new(
			&descriptor_set_allocator,
			layout.clone(),
			[WriteDescriptorSet::image_view_sampler(
				0,
				texture.clone(),
				sampler,
			)],
		)?
	};

	loop {
		previous_frame_end.as_mut().unwrap().cleanup_finished();

		if recreate_swapchain {
			let (new_swapchain, new_images) =
				match swapchain.recreate(SwapchainCreateInfo {
					image_extent: dimensions,
					..swapchain.create_info()
				}) {
					Ok(r) => r,
					Err(SwapchainCreationError::ImageExtentNotSupported {
						..
					}) => break,
					Err(e) => bail!("failed to recreate swapchain: {e}"),
				};

			swapchain = new_swapchain;

			attachment_image_views = window_size_dependent_setup(
				&new_images,
				render_pass.clone(),
				&mut viewport,
			);

			recreate_swapchain = false;
		}

		let (image_index, suboptimal, acquire_future) =
			match acquire_next_image(swapchain.clone(), None) {
				Ok(r) => r,
				Err(AcquireError::OutOfDate) => {
					recreate_swapchain = true;
					continue;
				}
				Err(e) => bail!("failed to acquire next image: {e}"),
			};

		if suboptimal {
			recreate_swapchain = true;
		}

		let mut builder = AutoCommandBufferBuilder::primary(
			&render_context.command_buffer_allocator,
			render_context.queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		handle_compositor_events(
			&socket,
			&mut wayland_display,
			&mut wayland_state,
			&mut render_context,
			texture.clone(),
		)?;

		builder
			.begin_render_pass(
				RenderPassBeginInfo {
					clear_values: vec![Some([0.0, 0.0, 0.0, 0.0].into())],
					..RenderPassBeginInfo::framebuffer(
						attachment_image_views[image_index as usize].clone(),
					)
				},
				SubpassContents::Inline,
			)?
			.set_viewport(0, [viewport.clone()].into_iter().collect())
			.bind_pipeline_graphics(pipeline.clone())
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				pipeline.layout().clone(),
				0,
				descriptor_set.clone(),
			)
			.bind_vertex_buffers(0, vertex_buffer.clone())
			.bind_index_buffer(index_buffer.clone())
			.draw_indexed(indices.len() as u32, 1, 0, 0, 0)?
			.end_render_pass()?;

		let command_buffer = builder.build()?;

		let future = previous_frame_end
			.take()
			.unwrap()
			.join(acquire_future)
			.then_execute(render_context.queue.clone(), command_buffer)?
			.then_swapchain_present(
				render_context.queue.clone(),
				SwapchainPresentInfo::swapchain_image_index(
					swapchain.clone(),
					image_index,
				),
			)
			.then_signal_fence_and_flush();

		match future {
			Ok(future) => {
				previous_frame_end = Some(future.boxed());
			}
			Err(FlushError::OutOfDate) => {
				recreate_swapchain = true;
				previous_frame_end =
					Some(sync::now(render_context.device.clone()).boxed());
			}
			Err(e) => {
				warn!("failed to flush future: {:#?}", e);
				previous_frame_end =
					Some(sync::now(render_context.device.clone()).boxed());
			}
		}
	}

	Ok(())
}
