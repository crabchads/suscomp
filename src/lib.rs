use std::{borrow::BorrowMut, sync::Arc};

use anyhow::Result;
use smithay::{
	backend::{
		allocator::Buffer as SmithayBuffer,
		renderer::utils::{
			with_renderer_surface_state, RendererSurfaceStateUserData,
		},
	},
	delegate_compositor, delegate_shm, delegate_xdg_shell,
	reexports::wayland_server::{
		protocol::{wl_buffer, wl_seat, wl_surface::WlSurface},
		Client, ListeningSocket,
	},
	reexports::{
		wayland_protocols::xdg::shell::server::xdg_toplevel,
		wayland_server::{
			backend::{ClientData, ClientId, DisconnectReason},
			Display,
		},
	},
	utils::Serial,
	wayland::{
		buffer::BufferHandler,
		compositor::{
			with_states, BufferAssignment, CompositorClientState,
			CompositorHandler, CompositorState, SurfaceAttributes,
		},
		shell::xdg::{
			PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler,
			XdgShellState,
		},
		shm::{with_buffer_contents, ShmHandler, ShmState},
	},
};
use tracing::debug;
use vulkano::{
	buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
	command_buffer::{
		allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
		BufferImageCopy, CommandBufferUsage, CopyBufferToImageInfo,
	},
	device::{Device, Queue},
	format::Format,
	image::{
		view::ImageView, AttachmentImage, ImageAccess, ImageDimensions,
		ImageUsage, ImageViewAbstract, ImmutableImage, StorageImage,
	},
	memory::allocator::{
		AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator,
	},
	pipeline::graphics::vertex_input::Vertex,
};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TexturedVertex {
	#[format(R32G32B32A32_SFLOAT)]
	pub position: [f32; 4],

	#[format(R32G32_SFLOAT)]
	pub tex_coords: [f32; 2],
}

pub fn textured_quad(
	width: f32,
	height: f32,
) -> (Vec<TexturedVertex>, Vec<u32>) {
	(
		vec![
			TexturedVertex {
				position: [-(width / 2.0), -(height / 2.0), 0.0, 1.0],
				tex_coords: [0.0, 1.0],
			},
			TexturedVertex {
				position: [-(width / 2.0), height / 2.0, 0.0, 1.0],
				tex_coords: [0.0, 0.0],
			},
			TexturedVertex {
				position: [width / 2.0, height / 2.0, 0.0, 1.0],
				tex_coords: [1.0, 0.0],
			},
			TexturedVertex {
				position: [width / 2.0, -(height / 2.0), 0.0, 1.0],
				tex_coords: [1.0, 1.0],
			},
		],
		vec![0, 2, 1, 0, 3, 2],
	)
}

pub struct RenderContext {
	pub command_buffer_allocator: StandardCommandBufferAllocator,
	pub memory_allocator: StandardMemoryAllocator,
	pub queue: Arc<Queue>,
	pub device: Arc<Device>,
}

pub fn load_texture(
	render_context: &mut RenderContext,
	dimensions: [u32; 2],
) -> Result<Arc<ImageView<AttachmentImage>>> {
	let texture = {
		let image = AttachmentImage::with_usage(
			&render_context.memory_allocator,
			dimensions,
			Format::B8G8R8A8_SRGB,
			ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
		)
		.unwrap();

		ImageView::new_default(image).unwrap()
	};

	Ok(texture)
}

pub struct WaylandCompositor {
	pub compositor_state: CompositorState,
	pub xdg_shell_state: XdgShellState,
	pub shm_state: ShmState,
	pub clients: Vec<Client>,
}

pub fn handle_compositor_events(
	socket: &ListeningSocket,
	display: &mut Display<WaylandCompositor>,
	state: &mut WaylandCompositor,
	render_context: &mut RenderContext,
	texture: Arc<ImageView<AttachmentImage>>,
) -> Result<()> {
	if let Some(stream) = socket.accept()? {
		let client = display
			.handle()
			.insert_client(stream, Arc::new(ClientState::default()))?;
		state.clients.push(client);
	}

	state
		.xdg_shell_state
		.toplevel_surfaces()
		.iter()
		.map(|surface| {
			with_states(surface.wl_surface(), |states| {
				let mut attrs =
					states.cached_state.current::<SurfaceAttributes>();

				if let Some(BufferAssignment::NewBuffer(buffer)) =
					attrs.buffer.take()
				{
					with_buffer_contents(&buffer, |slice, size, data| {
						debug!(
							"buffer {}, w: {}, h: {}, format: {:?}",
							size, data.width, data.height, data.format
						);

						let bytes =
							unsafe { std::slice::from_raw_parts(slice, size) };
						let rebuilt = Vec::from(bytes);

						let buffer = Buffer::from_iter(
							&render_context.memory_allocator,
							BufferCreateInfo {
								usage: BufferUsage::TRANSFER_SRC,
								..Default::default()
							},
							AllocationCreateInfo {
								usage: MemoryUsage::Upload,
								..Default::default()
							},
							rebuilt,
						)
						.unwrap();

						// Render surface
						let mut uploads = AutoCommandBufferBuilder::primary(
							&render_context.command_buffer_allocator,
							render_context.queue.queue_family_index(),
							CommandBufferUsage::OneTimeSubmit,
						)
						.unwrap();

						uploads
						.copy_buffer_to_image(
							vulkano::command_buffer::CopyBufferToImageInfo {
								regions: [BufferImageCopy {
									image_subresource: texture
										.image()
										.subresource_layers(),
									image_extent: [data.width as u32, data.height as u32, 1],
									..Default::default()
								}]
								.into(),
								..CopyBufferToImageInfo::buffer_image(
									buffer,
									texture.image().clone(),
								)
							},
						)
						.unwrap();
					})
					.unwrap();
				}
			});

			Ok(())
		})
		.collect::<Result<Vec<_>>>()?;

	display.dispatch_clients(state)?;
	display.flush_clients()?;

	Ok(())
}

impl ShmHandler for WaylandCompositor {
	fn shm_state(&self) -> &ShmState {
		&self.shm_state
	}
}

impl BufferHandler for WaylandCompositor {
	fn buffer_destroyed(&mut self, _buffer: &wl_buffer::WlBuffer) {}
}

impl CompositorHandler for WaylandCompositor {
	fn compositor_state(&mut self) -> &mut CompositorState {
		&mut self.compositor_state
	}

	fn client_compositor_state<'a>(
		&self,
		client: &'a Client,
	) -> &'a CompositorClientState {
		&client.get_data::<ClientState>().unwrap().compositor_state
	}

	fn commit(&mut self, surface: &WlSurface) {
		debug!("Requested commit for surface: {:?}", surface);
	}
}

#[derive(Default)]
struct ClientState {
	compositor_state: CompositorClientState,
}
impl ClientData for ClientState {
	fn initialized(&self, client_id: ClientId) {
		debug!("client initialized: {:?}", client_id);
	}

	fn disconnected(&self, client_id: ClientId, _reason: DisconnectReason) {
		debug!("client disconnected: {:?}", client_id);
	}
}

impl AsMut<CompositorState> for WaylandCompositor {
	fn as_mut(&mut self) -> &mut CompositorState {
		&mut self.compositor_state
	}
}

impl XdgShellHandler for WaylandCompositor {
	fn xdg_shell_state(&mut self) -> &mut XdgShellState {
		&mut self.xdg_shell_state
	}

	fn new_toplevel(&mut self, surface: ToplevelSurface) {
		surface.with_pending_state(|state| {
			state.states.set(xdg_toplevel::State::Activated);
		});
		surface.send_configure();
	}

	fn new_popup(
		&mut self,
		_surface: PopupSurface,
		_positioner: PositionerState,
	) {
	}

	fn grab(
		&mut self,
		_surface: PopupSurface,
		_seat: wl_seat::WlSeat,
		_serial: Serial,
	) {
		// Handle popup grab here
	}
}

delegate_xdg_shell!(WaylandCompositor);
delegate_compositor!(WaylandCompositor);
delegate_shm!(WaylandCompositor);
