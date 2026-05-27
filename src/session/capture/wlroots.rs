use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use async_shutdown::ShutdownManager;
use smithay::backend::allocator::dmabuf::AsDmabuf;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Buffer, Fourcc, Modifier};
use wayland_client::protocol::wl_buffer::WlBuffer;
use wayland_client::protocol::wl_keyboard;
use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::wl_pointer;
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::WlSeat;
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1};
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1;
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1;
use wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1};
use wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1;
use wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1;
use wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1;
use xkbcommon::xkb;

use crate::config::{KeyboardConfig, WlrootsCaptureConfig};
use crate::session::capture::CaptureInputSender;
use crate::session::compositor::find_render_node;
use crate::session::compositor::frame::{ExportedFrame, ExportedPlane, FrameColorSpace};
use crate::session::compositor::input::CompositorInputEvent;
use crate::session::manager::SessionShutdownReason;

const BUFFER_POOL_SIZE: usize = 3;
const WAYLAND_POLL_TIMEOUT_MS: i32 = 5;

pub struct WlrootsReady {
	pub hdr: bool,
	pub width: u32,
	pub height: u32,
	pub refresh_rate: u32,
}

type WlrootsHandles = (
	mpsc::Receiver<ExportedFrame>,
	CaptureInputSender,
	mpsc::Receiver<Result<WlrootsReady, String>>,
);

pub fn start_capture(
	config: WlrootsCaptureConfig,
	keyboard_config: KeyboardConfig,
	width: u32,
	height: u32,
	refresh_rate: u32,
	gpu: Option<String>,
	stop: ShutdownManager<SessionShutdownReason>,
) -> Result<WlrootsHandles, String> {
	let (frame_tx, frame_rx) = mpsc::sync_channel::<ExportedFrame>(2);
	let (input_tx, input_rx) = mpsc::channel::<CompositorInputEvent>();
	let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<WlrootsReady, String>>(1);

	std::thread::Builder::new()
		.name("wlroots-capture".to_string())
		.spawn(move || {
			let result = run_capture(
				config,
				keyboard_config,
				width,
				height,
				refresh_rate,
				gpu,
				frame_tx,
				input_rx,
				ready_tx.clone(),
				stop,
			);
			if let Err(e) = result {
				tracing::error!("wlroots capture failed: {e}");
				let _ = ready_tx.send(Err(e));
			}
		})
		.map_err(|e| format!("Failed to spawn wlroots capture thread: {e}"))?;

	Ok((frame_rx, CaptureInputSender::Wlroots(input_tx), ready_rx))
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
	config: WlrootsCaptureConfig,
	keyboard_config: KeyboardConfig,
	width: u32,
	height: u32,
	refresh_rate: u32,
	gpu: Option<String>,
	frame_tx: mpsc::SyncSender<ExportedFrame>,
	input_rx: mpsc::Receiver<CompositorInputEvent>,
	ready_tx: mpsc::SyncSender<Result<WlrootsReady, String>>,
	stop: ShutdownManager<SessionShutdownReason>,
) -> Result<(), String> {
	let _session_stop_token = stop.trigger_shutdown_token(SessionShutdownReason::CompositorStopped);
	let _delay_stop = stop.delay_shutdown_token();

	let render_node = find_render_node(&gpu)?;
	tracing::debug!("wlroots capture using render node: {}", render_node.display());

	let render_fd = std::fs::OpenOptions::new()
		.read(true)
		.write(true)
		.open(&render_node)
		.map_err(|e| format!("Failed to open render node {}: {e}", render_node.display()))?;
	let gbm_device = GbmDevice::new(render_fd).map_err(|e| format!("Failed to create GBM device: {e}"))?;
	let gbm_allocator = GbmAllocator::new(gbm_device, GbmBufferFlags::RENDERING);

	let connection = connect_to_display(config.display.as_deref())?;
	let mut event_queue = connection.new_event_queue::<WlrootsState>();
	let qh = event_queue.handle();

	let mut state = WlrootsState::new(
		config,
		keyboard_config,
		width,
		height,
		refresh_rate,
		gbm_allocator,
		frame_tx,
	);
	connection.display().get_registry(&qh, ());
	event_queue
		.roundtrip(&mut state)
		.map_err(|e| format!("Wayland registry roundtrip failed: {e}"))?;
	event_queue
		.roundtrip(&mut state)
		.map_err(|e| format!("Wayland initial event roundtrip failed: {e}"))?;

	state.finish_initialization(&connection, &qh)?;
	state.start_capture(&qh, true)?;

	while state.startup_result.is_none() && !stop.is_shutdown_triggered() {
		dispatch_once(&connection, &mut event_queue, &mut state, WAYLAND_POLL_TIMEOUT_MS)?;
		state.process_pending_input(&connection, &input_rx);
	}

	match state.startup_result.take() {
		Some(Ok(())) => {
			let ready_width = state.buffer_width.max(1);
			let ready_height = state.buffer_height.max(1);
			let _ = ready_tx.send(Ok(WlrootsReady {
				hdr: false,
				width: ready_width,
				height: ready_height,
				refresh_rate,
			}));
			tracing::info!(
				"wlroots capture started: {}x{} @ {}Hz (client requested {}x{})",
				ready_width,
				ready_height,
				refresh_rate,
				width,
				height
			);
		},
		Some(Err(e)) => return Err(e),
		None => return Ok(()),
	}

	while !stop.is_shutdown_triggered() {
		state.process_pending_input(&connection, &input_rx);
		dispatch_once(&connection, &mut event_queue, &mut state, WAYLAND_POLL_TIMEOUT_MS)?;
		state.start_capture(&qh, false)?;
	}

	tracing::info!("wlroots capture stopped.");
	Ok(())
}

fn connect_to_display(display: Option<&str>) -> Result<Connection, String> {
	if let Some(display) = display {
		let socket_path = wayland_socket_path(display)?;
		let stream = UnixStream::connect(&socket_path)
			.map_err(|e| format!("Failed to connect to Wayland display {}: {e}", socket_path.display()))?;
		Connection::from_socket(stream).map_err(|e| format!("Failed to initialize Wayland connection: {e}"))
	} else {
		Connection::connect_to_env().map_err(|e| format!("Failed to connect to Wayland compositor: {e}"))
	}
}

fn wayland_socket_path(display: &str) -> Result<PathBuf, String> {
	let path = PathBuf::from(display);
	if path.is_absolute() {
		return Ok(path);
	}

	let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
		.map(PathBuf::from)
		.ok_or_else(|| "XDG_RUNTIME_DIR is not set".to_string())?;
	if !runtime_dir.is_absolute() {
		return Err("XDG_RUNTIME_DIR is not absolute".to_string());
	}
	Ok(runtime_dir.join(path))
}

fn dispatch_once(
	connection: &Connection,
	event_queue: &mut EventQueue<WlrootsState>,
	state: &mut WlrootsState,
	timeout_ms: i32,
) -> Result<(), String> {
	event_queue
		.dispatch_pending(state)
		.map_err(|e| format!("Wayland dispatch failed: {e}"))?;
	connection.flush().map_err(|e| format!("Wayland flush failed: {e}"))?;

	let Some(guard) = event_queue.prepare_read() else {
		event_queue
			.dispatch_pending(state)
			.map_err(|e| format!("Wayland dispatch failed: {e}"))?;
		return Ok(());
	};

	let fd = guard.connection_fd().as_raw_fd();
	let mut pfd = libc::pollfd {
		fd,
		events: libc::POLLIN,
		revents: 0,
	};
	let poll_result = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
	if poll_result < 0 {
		drop(guard);
		return Err(std::io::Error::last_os_error().to_string());
	}

	if poll_result > 0 && (pfd.revents & libc::POLLIN) != 0 {
		guard.read().map_err(|e| format!("Wayland read failed: {e}"))?;
	} else {
		drop(guard);
	}

	event_queue
		.dispatch_pending(state)
		.map_err(|e| format!("Wayland dispatch failed: {e}"))?;
	Ok(())
}

#[derive(Clone)]
struct OutputInfo {
	global_name: u32,
	output: WlOutput,
	name: Option<String>,
	description: Option<String>,
	width: u32,
	height: u32,
}

struct CaptureSlot {
	dmabuf: smithay::backend::allocator::dmabuf::Dmabuf,
	wl_buffer: WlBuffer,
	consumed: Arc<AtomicBool>,
}

struct PendingFrame {
	frame: ZwlrScreencopyFrameV1,
	slot_index: Option<usize>,
	offered_format: Option<u32>,
	offered_width: u32,
	offered_height: u32,
	y_inverted: bool,
}

struct WlrootsState {
	config: WlrootsCaptureConfig,
	keyboard_config: KeyboardConfig,
	expected_width: u32,
	expected_height: u32,
	allocator: GbmAllocator<File>,
	frame_tx: mpsc::SyncSender<ExportedFrame>,

	screencopy_manager: Option<ZwlrScreencopyManagerV1>,
	linux_dmabuf: Option<ZwpLinuxDmabufV1>,
	dmabuf_modifiers: HashMap<u32, Vec<Modifier>>,
	virtual_pointer_manager: Option<ZwlrVirtualPointerManagerV1>,
	virtual_keyboard_manager: Option<ZwpVirtualKeyboardManagerV1>,
	outputs: Vec<OutputInfo>,
	seat: Option<WlSeat>,

	selected_output: Option<WlOutput>,
	virtual_pointer: Option<ZwlrVirtualPointerV1>,
	virtual_keyboard: Option<ZwpVirtualKeyboardV1>,
	keymap_file: Option<File>,
	xkb_state: Option<xkb::State>,

	buffer_format: Option<u32>,
	buffer_width: u32,
	buffer_height: u32,
	buffer_pool: Vec<CaptureSlot>,
	next_buffer_index: usize,
	frame_interval: Duration,
	last_capture_at: Option<Instant>,

	pending_frame: Option<PendingFrame>,
	capture_in_progress: bool,
	startup_result: Option<Result<(), String>>,
	startup_complete: bool,
	warned_y_inverted: bool,
	warned_size_mismatch: bool,
}

impl WlrootsState {
	#[allow(clippy::too_many_arguments)]
	fn new(
		config: WlrootsCaptureConfig,
		keyboard_config: KeyboardConfig,
		expected_width: u32,
		expected_height: u32,
		refresh_rate: u32,
		allocator: GbmAllocator<File>,
		frame_tx: mpsc::SyncSender<ExportedFrame>,
	) -> Self {
		Self {
			config,
			keyboard_config,
			expected_width,
			expected_height,
			allocator,
			frame_tx,
			screencopy_manager: None,
			linux_dmabuf: None,
			dmabuf_modifiers: HashMap::new(),
			virtual_pointer_manager: None,
			virtual_keyboard_manager: None,
			outputs: Vec::new(),
			seat: None,
			selected_output: None,
			virtual_pointer: None,
			virtual_keyboard: None,
			keymap_file: None,
			xkb_state: None,
			buffer_format: None,
			buffer_width: 0,
			buffer_height: 0,
			buffer_pool: Vec::new(),
			next_buffer_index: 0,
			frame_interval: Duration::from_secs_f64(1.0 / refresh_rate.max(1) as f64),
			last_capture_at: None,
			pending_frame: None,
			capture_in_progress: false,
			startup_result: None,
			startup_complete: false,
			warned_y_inverted: false,
			warned_size_mismatch: false,
		}
	}

	fn finish_initialization(&mut self, connection: &Connection, qh: &QueueHandle<Self>) -> Result<(), String> {
		let missing = [
			(self.screencopy_manager.is_none(), "zwlr_screencopy_manager_v1 v3"),
			(self.linux_dmabuf.is_none(), "zwp_linux_dmabuf_v1"),
			(
				self.virtual_pointer_manager.is_none(),
				"zwlr_virtual_pointer_manager_v1",
			),
			(
				self.virtual_keyboard_manager.is_none(),
				"zwp_virtual_keyboard_manager_v1",
			),
			(self.seat.is_none(), "wl_seat"),
		]
		.into_iter()
		.filter_map(|(missing, name)| missing.then_some(name))
		.collect::<Vec<_>>();
		if !missing.is_empty() {
			return Err(format!(
				"wlroots capture missing required Wayland globals: {}",
				missing.join(", ")
			));
		}

		let output = self.select_output()?;
		let output_name = output
			.name
			.as_deref()
			.or(output.description.as_deref())
			.map(ToOwned::to_owned)
			.unwrap_or_else(|| format!("#{}", output.global_name));
		tracing::info!("wlroots capture selected output: {output_name}");
		self.selected_output = Some(output.output.clone());

		self.init_virtual_input(qh)?;
		connection
			.flush()
			.map_err(|e| format!("Failed to flush virtual input setup: {e}"))?;
		Ok(())
	}

	fn select_output(&self) -> Result<OutputInfo, String> {
		if self.outputs.is_empty() {
			return Err("No wl_output objects advertised by compositor".to_string());
		}

		if let Some(name) = &self.config.output {
			self.outputs
				.iter()
				.find(|output| output.name.as_deref() == Some(name.as_str()))
				.or_else(|| {
					self.outputs
						.iter()
						.find(|output| output.description.as_deref() == Some(name.as_str()))
				})
				.cloned()
				.ok_or_else(|| {
					let available = self
						.outputs
						.iter()
						.map(|output| {
							output
								.name
								.as_deref()
								.or(output.description.as_deref())
								.map(ToOwned::to_owned)
								.unwrap_or_else(|| format!("#{}", output.global_name))
						})
						.collect::<Vec<_>>()
						.join(", ");
					format!("Requested wlroots output '{name}' was not found. Available outputs: {available}")
				})
		} else {
			Ok(self.outputs[0].clone())
		}
	}

	fn init_virtual_input(&mut self, qh: &QueueHandle<Self>) -> Result<(), String> {
		let seat = self.seat.as_ref().ok_or_else(|| "No wl_seat available".to_string())?;
		let output = self
			.selected_output
			.as_ref()
			.ok_or_else(|| "No selected output available".to_string())?;

		let pointer_manager = self
			.virtual_pointer_manager
			.as_ref()
			.ok_or_else(|| "Virtual pointer manager missing".to_string())?;
		let pointer = if pointer_manager.version() >= 2 {
			pointer_manager.create_virtual_pointer_with_output(Some(seat), Some(output), qh, ())
		} else {
			pointer_manager.create_virtual_pointer(Some(seat), qh, ())
		};
		pointer.axis_source(wl_pointer::AxisSource::Wheel);
		self.virtual_pointer = Some(pointer);

		let keyboard_manager = self
			.virtual_keyboard_manager
			.as_ref()
			.ok_or_else(|| "Virtual keyboard manager missing".to_string())?;
		let keyboard = keyboard_manager.create_virtual_keyboard(seat, qh, ());
		let (keymap_file, keymap_size, xkb_state) = create_keymap(&self.keyboard_config)?;
		keyboard.keymap(
			wl_keyboard::KeymapFormat::XkbV1.into(),
			keymap_file.as_fd(),
			keymap_size,
		);
		self.keymap_file = Some(keymap_file);
		self.xkb_state = Some(xkb_state);
		self.virtual_keyboard = Some(keyboard);

		Ok(())
	}

	fn start_capture(&mut self, qh: &QueueHandle<Self>, immediate: bool) -> Result<(), String> {
		if self.capture_in_progress {
			return Ok(());
		}
		if !immediate {
			if let Some(last_capture_at) = self.last_capture_at {
				if last_capture_at.elapsed() < self.frame_interval {
					return Ok(());
				}
			}
		}
		let manager = self
			.screencopy_manager
			.as_ref()
			.ok_or_else(|| "Screencopy manager missing".to_string())?;
		let output = self
			.selected_output
			.as_ref()
			.ok_or_else(|| "Selected output missing".to_string())?;
		let frame = manager.capture_output(self.config.render_cursor as i32, output, qh, ());
		self.pending_frame = Some(PendingFrame {
			frame: frame.clone(),
			slot_index: None,
			offered_format: None,
			offered_width: 0,
			offered_height: 0,
			y_inverted: false,
		});
		self.capture_in_progress = true;
		self.last_capture_at = Some(Instant::now());
		Ok(())
	}

	fn handle_linux_dmabuf_offer(&mut self, format: u32, width: u32, height: u32) {
		if let Some(pending) = self.pending_frame.as_mut() {
			pending.offered_format = Some(format);
			pending.offered_width = width;
			pending.offered_height = height;
		}
	}

	fn handle_buffer_done(&mut self, frame: &ZwlrScreencopyFrameV1, qh: &QueueHandle<Self>) {
		let Some((format, width, height)) = self.pending_frame.as_ref().and_then(|pending| {
			pending
				.offered_format
				.map(|format| (format, pending.offered_width, pending.offered_height))
		}) else {
			self.finish_capture_with_error("Compositor did not offer linux-dmabuf screencopy buffers".to_string());
			return;
		};

		if (width != self.expected_width || height != self.expected_height) && !self.warned_size_mismatch {
			self.warned_size_mismatch = true;
			tracing::warn!(
				"wlroots output size {width}x{height} does not match requested stream size {}x{}; streaming captured output size",
				self.expected_width,
				self.expected_height
			);
		}

		if let Err(e) = self.ensure_buffer_pool(format, width, height, qh) {
			self.finish_capture_with_error(e);
			return;
		}

		let Some(slot_index) = self.acquire_slot() else {
			tracing::trace!("No free wlroots capture buffer; skipping screencopy frame");
			self.finish_capture(false);
			return;
		};

		if let Some(pending) = self.pending_frame.as_mut() {
			pending.slot_index = Some(slot_index);
			// Moonlight expects a steady video cadence even when the desktop is
			// static; damage-only copies can stall indefinitely after startup.
			frame.copy(&self.buffer_pool[slot_index].wl_buffer);
		}
	}

	fn handle_flags(&mut self, flags: WEnum<zwlr_screencopy_frame_v1::Flags>) {
		if let Some(pending) = self.pending_frame.as_mut() {
			pending.y_inverted = match flags {
				WEnum::Value(flags) => flags.contains(zwlr_screencopy_frame_v1::Flags::YInvert),
				WEnum::Unknown(raw) => raw & zwlr_screencopy_frame_v1::Flags::YInvert.bits() != 0,
			};
		}
	}

	fn handle_ready(&mut self, qh: &QueueHandle<Self>) {
		let Some(pending) = self.pending_frame.as_ref() else {
			return;
		};
		let Some(slot_index) = pending.slot_index else {
			self.finish_capture(false);
			return;
		};
		if pending.y_inverted && !self.warned_y_inverted {
			self.warned_y_inverted = true;
			tracing::warn!("wlroots screencopy reported y-inverted frames; v1 streams them without correction");
		}

		let slot = &self.buffer_pool[slot_index];
		let planes = slot
			.dmabuf
			.handles()
			.zip(slot.dmabuf.offsets())
			.zip(slot.dmabuf.strides())
			.map(
				|((handle, offset), stride): ((std::os::fd::BorrowedFd<'_>, u32), u32)| ExportedPlane {
					fd: handle.as_raw_fd(),
					offset,
					stride,
				},
			)
			.collect();

		let frame = ExportedFrame {
			planes,
			format: slot.dmabuf.format().code as u32,
			modifier: Into::<u64>::into(slot.dmabuf.format().modifier),
			width: slot.dmabuf.width(),
			height: slot.dmabuf.height(),
			created_at: Instant::now(),
			buffer_index: slot_index,
			consumed: slot.consumed.clone(),
			color_space: FrameColorSpace::Srgb,
			hdr_metadata: None,
		};

		match self.frame_tx.try_send(frame) {
			Ok(()) => {
				if !self.startup_complete {
					self.startup_complete = true;
					self.startup_result = Some(Ok(()));
				}
			},
			Err(mpsc::TrySendError::Full(frame)) => {
				frame.consumed.store(true, Ordering::Release);
			},
			Err(mpsc::TrySendError::Disconnected(_)) => {
				tracing::debug!("wlroots frame channel disconnected");
			},
		}

		self.finish_capture(false);
		let _ = self.start_capture(qh, false);
	}

	fn handle_failed(&mut self, qh: &QueueHandle<Self>) {
		tracing::debug!("wlroots screencopy frame failed");
		if !self.startup_complete {
			self.startup_result = Some(Err("Initial wlroots screencopy frame failed".to_string()));
		}
		self.finish_capture(false);
		let _ = self.start_capture(qh, false);
	}

	fn finish_capture_with_error(&mut self, error: String) {
		tracing::warn!("{error}");
		if !self.startup_complete {
			self.startup_result = Some(Err(error));
		}
		self.finish_capture(false);
	}

	fn finish_capture(&mut self, release_slot: bool) {
		if let Some(pending) = self.pending_frame.take() {
			if release_slot {
				if let Some(slot_index) = pending.slot_index {
					self.buffer_pool[slot_index].consumed.store(true, Ordering::Release);
				}
			}
			pending.frame.destroy();
		}
		self.capture_in_progress = false;
	}

	fn ensure_buffer_pool(
		&mut self,
		format: u32,
		width: u32,
		height: u32,
		qh: &QueueHandle<Self>,
	) -> Result<(), String> {
		if self.buffer_format == Some(format) && self.buffer_width == width && self.buffer_height == height {
			return Ok(());
		}

		if self
			.buffer_pool
			.iter()
			.any(|slot| !slot.consumed.load(Ordering::Acquire))
		{
			return Err("Cannot resize wlroots capture pool while encoder is reading old buffers".to_string());
		}

		for slot in self.buffer_pool.drain(..) {
			slot.wl_buffer.destroy();
		}

		let fourcc = Fourcc::try_from(format).map_err(|e| format!("Unsupported screencopy DRM format: {e:?}"))?;
		let modifiers = self
			.dmabuf_modifiers
			.get(&format)
			.filter(|modifiers| !modifiers.is_empty())
			.cloned()
			.unwrap_or_else(|| vec![Modifier::Invalid]);
		let linux_dmabuf = self
			.linux_dmabuf
			.as_ref()
			.ok_or_else(|| "linux-dmabuf manager missing".to_string())?;

		let mut pool = Vec::with_capacity(BUFFER_POOL_SIZE);
		for i in 0..BUFFER_POOL_SIZE {
			let buffer = self
				.allocator
				.create_buffer_with_flags(width, height, fourcc, &modifiers, GbmBufferFlags::RENDERING)
				.map_err(|e| format!("Failed to allocate wlroots capture GBM buffer {i}: {e}"))?;
			let dmabuf = buffer
				.export()
				.map_err(|e| format!("Failed to export wlroots capture GBM buffer {i}: {e}"))?;
			let wl_buffer = create_wl_buffer(linux_dmabuf, &dmabuf, format, width, height, qh)?;
			pool.push(CaptureSlot {
				dmabuf,
				wl_buffer,
				consumed: Arc::new(AtomicBool::new(true)),
			});
		}

		self.buffer_format = Some(format);
		self.buffer_width = width;
		self.buffer_height = height;
		self.buffer_pool = pool;
		self.next_buffer_index = 0;
		Ok(())
	}

	fn acquire_slot(&mut self) -> Option<usize> {
		for offset in 0..self.buffer_pool.len() {
			let index = (self.next_buffer_index + offset) % self.buffer_pool.len();
			if self.buffer_pool[index].consumed.load(Ordering::Acquire) {
				self.buffer_pool[index].consumed.store(false, Ordering::Release);
				self.next_buffer_index = (index + 1) % self.buffer_pool.len();
				return Some(index);
			}
		}
		None
	}

	fn process_pending_input(&mut self, connection: &Connection, input_rx: &mpsc::Receiver<CompositorInputEvent>) {
		while let Ok(event) = input_rx.try_recv() {
			self.process_input(event);
		}
		let _ = connection.flush();
	}

	fn process_input(&mut self, event: CompositorInputEvent) {
		let time = monotonic_ms();
		match event {
			CompositorInputEvent::KeyDown { keycode } => self.send_key(time, keycode, true),
			CompositorInputEvent::KeyUp { keycode } => self.send_key(time, keycode, false),
			CompositorInputEvent::MouseMoveAbsolute {
				x,
				y,
				screen_width,
				screen_height,
			} => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					let x_extent = u32::try_from(screen_width.max(1)).unwrap_or(1);
					let y_extent = u32::try_from(screen_height.max(1)).unwrap_or(1);
					let x = u32::try_from(x.max(0)).unwrap_or(0).min(x_extent);
					let y = u32::try_from(y.max(0)).unwrap_or(0).min(y_extent);
					pointer.motion_absolute(time, x, y, x_extent, y_extent);
					pointer.frame();
				}
			},
			CompositorInputEvent::MouseMoveRelative { dx, dy } => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					pointer.motion(time, dx as f64, dy as f64);
					pointer.frame();
				}
			},
			CompositorInputEvent::MouseButtonDown { button } => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					pointer.button(time, button, wl_pointer::ButtonState::Pressed);
					pointer.frame();
				}
			},
			CompositorInputEvent::MouseButtonUp { button } => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					pointer.button(time, button, wl_pointer::ButtonState::Released);
					pointer.frame();
				}
			},
			CompositorInputEvent::ScrollVertical { amount } => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					pointer.axis(time, wl_pointer::Axis::VerticalScroll, -(amount as f64));
					pointer.frame();
				}
			},
			CompositorInputEvent::ScrollHorizontal { amount } => {
				if let Some(pointer) = self.virtual_pointer.as_ref() {
					pointer.axis(time, wl_pointer::Axis::HorizontalScroll, amount as f64);
					pointer.frame();
				}
			},
		}
	}

	fn send_key(&mut self, time: u32, keycode: u32, pressed: bool) {
		let Some(keyboard) = self.virtual_keyboard.as_ref() else {
			return;
		};

		keyboard.key(
			time,
			keycode,
			if pressed {
				wl_keyboard::KeyState::Pressed as u32
			} else {
				wl_keyboard::KeyState::Released as u32
			},
		);

		if let Some(xkb_state) = self.xkb_state.as_mut() {
			let direction = if pressed {
				xkb::KeyDirection::Down
			} else {
				xkb::KeyDirection::Up
			};
			let changed = xkb_state.update_key(xkb::Keycode::new(keycode + 8), direction);
			if changed != 0 {
				let depressed = xkb_state.serialize_mods(xkb::STATE_MODS_DEPRESSED);
				let latched = xkb_state.serialize_mods(xkb::STATE_MODS_LATCHED);
				let locked = xkb_state.serialize_mods(xkb::STATE_MODS_LOCKED);
				let group = xkb_state.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE);
				keyboard.modifiers(depressed, latched, locked, group);
			}
		}
	}
}

fn create_wl_buffer(
	linux_dmabuf: &ZwpLinuxDmabufV1,
	dmabuf: &smithay::backend::allocator::dmabuf::Dmabuf,
	format: u32,
	width: u32,
	height: u32,
	qh: &QueueHandle<WlrootsState>,
) -> Result<WlBuffer, String> {
	let params: ZwpLinuxBufferParamsV1 = linux_dmabuf.create_params(qh, ());
	let modifier = Into::<u64>::into(dmabuf.format().modifier);
	let modifier_hi = (modifier >> 32) as u32;
	let modifier_lo = modifier as u32;
	for (plane_idx, ((handle, offset), stride)) in
		dmabuf.handles().zip(dmabuf.offsets()).zip(dmabuf.strides()).enumerate()
	{
		params.add(
			handle.as_fd(),
			plane_idx as u32,
			offset,
			stride,
			modifier_hi,
			modifier_lo,
		);
	}
	let wl_buffer = params.create_immed(
		width as i32,
		height as i32,
		format,
		zwp_linux_buffer_params_v1::Flags::empty(),
		qh,
		(),
	);
	params.destroy();
	Ok(wl_buffer)
}

fn create_keymap(config: &KeyboardConfig) -> Result<(File, u32, xkb::State), String> {
	let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
	let model = if config.model.is_empty() {
		"pc105"
	} else {
		config.model.as_str()
	};
	let layout = if config.layout.is_empty() {
		"us"
	} else {
		config.layout.as_str()
	};
	let keymap = xkb::Keymap::new_from_names(
		&context,
		"",
		model,
		layout,
		config.variant.as_str(),
		config.options.clone(),
		xkb::KEYMAP_COMPILE_NO_FLAGS,
	)
	.ok_or_else(|| "Failed to compile XKB keymap for wlroots virtual keyboard".to_string())?;
	let mut keymap_bytes = keymap.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1).into_bytes();
	keymap_bytes.push(0);
	let keymap_size = u32::try_from(keymap_bytes.len()).map_err(|_| "XKB keymap is too large".to_string())?;
	let name = std::ffi::CString::new("moonshine-keymap").unwrap();
	let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
	if fd < 0 {
		return Err(format!(
			"Failed to create keymap memfd: {}",
			std::io::Error::last_os_error()
		));
	}
	let mut file = unsafe { File::from_raw_fd(fd) };
	file.write_all(&keymap_bytes)
		.map_err(|e| format!("Failed to write keymap: {e}"))?;
	file.rewind().map_err(|e| format!("Failed to rewind keymap: {e}"))?;
	let state = xkb::State::new(&keymap);
	Ok((file, keymap_size, state))
}

fn monotonic_ms() -> u32 {
	static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
	let start = START.get_or_init(Instant::now);
	start.elapsed().as_millis() as u32
}

impl Dispatch<WlRegistry, ()> for WlrootsState {
	fn event(
		state: &mut Self,
		registry: &WlRegistry,
		event: wayland_client::protocol::wl_registry::Event,
		_: &(),
		_: &Connection,
		qh: &QueueHandle<Self>,
	) {
		if let wayland_client::protocol::wl_registry::Event::Global {
			name,
			interface,
			version,
		} = event
		{
			match interface.as_str() {
				"zwlr_screencopy_manager_v1" if version >= 3 => {
					state.screencopy_manager =
						Some(registry.bind::<ZwlrScreencopyManagerV1, _, _>(name, version.min(3), qh, ()));
				},
				"zwp_linux_dmabuf_v1" => {
					state.linux_dmabuf = Some(registry.bind::<ZwpLinuxDmabufV1, _, _>(name, version.min(3), qh, ()));
				},
				"zwlr_virtual_pointer_manager_v1" => {
					state.virtual_pointer_manager =
						Some(registry.bind::<ZwlrVirtualPointerManagerV1, _, _>(name, version.min(2), qh, ()));
				},
				"zwp_virtual_keyboard_manager_v1" => {
					state.virtual_keyboard_manager =
						Some(registry.bind::<ZwpVirtualKeyboardManagerV1, _, _>(name, 1, qh, ()));
				},
				"wl_seat" if state.seat.is_none() => {
					state.seat = Some(registry.bind::<WlSeat, _, _>(name, version.min(7), qh, ()));
				},
				"wl_output" => {
					let output =
						registry.bind::<WlOutput, _, _>(name, version.min(4), qh, OutputData { global_name: name });
					state.outputs.push(OutputInfo {
						global_name: name,
						output,
						name: None,
						description: None,
						width: 0,
						height: 0,
					});
				},
				_ => {},
			}
		}
	}
}

struct OutputData {
	global_name: u32,
}

impl Dispatch<WlOutput, OutputData> for WlrootsState {
	fn event(
		state: &mut Self,
		_: &WlOutput,
		event: wl_output::Event,
		data: &OutputData,
		_: &Connection,
		_: &QueueHandle<Self>,
	) {
		let Some(output) = state
			.outputs
			.iter_mut()
			.find(|output| output.global_name == data.global_name)
		else {
			return;
		};
		match event {
			wl_output::Event::Mode { width, height, .. } => {
				output.width = width.max(0) as u32;
				output.height = height.max(0) as u32;
			},
			wl_output::Event::Name { name } => {
				output.name = Some(name);
			},
			wl_output::Event::Description { description } => {
				output.description = Some(description);
			},
			_ => {},
		}
	}
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for WlrootsState {
	fn event(
		state: &mut Self,
		_: &ZwpLinuxDmabufV1,
		event: wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::Event,
		_: &(),
		_: &Connection,
		_: &QueueHandle<Self>,
	) {
		match event {
			wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::Event::Format { format } => {
				state
					.dmabuf_modifiers
					.entry(format)
					.or_default()
					.push(Modifier::Invalid);
			},
			wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::Event::Modifier {
				format,
				modifier_hi,
				modifier_lo,
			} => {
				let modifier = (u64::from(modifier_hi) << 32) | u64::from(modifier_lo);
				let modifier = Modifier::from(modifier);
				let modifiers = state.dmabuf_modifiers.entry(format).or_default();
				if !modifiers.contains(&modifier) {
					modifiers.push(modifier);
				}
			},
			_ => {},
		}
	}
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for WlrootsState {
	fn event(
		state: &mut Self,
		frame: &ZwlrScreencopyFrameV1,
		event: zwlr_screencopy_frame_v1::Event,
		_: &(),
		_: &Connection,
		qh: &QueueHandle<Self>,
	) {
		match event {
			zwlr_screencopy_frame_v1::Event::LinuxDmabuf { format, width, height } => {
				state.handle_linux_dmabuf_offer(format, width, height);
			},
			zwlr_screencopy_frame_v1::Event::BufferDone => state.handle_buffer_done(frame, qh),
			zwlr_screencopy_frame_v1::Event::Flags { flags } => state.handle_flags(flags),
			zwlr_screencopy_frame_v1::Event::Ready { .. } => state.handle_ready(qh),
			zwlr_screencopy_frame_v1::Event::Failed => state.handle_failed(qh),
			_ => {},
		}
	}
}

delegate_noop!(WlrootsState: ignore WlSeat);
delegate_noop!(WlrootsState: ignore WlBuffer);
delegate_noop!(WlrootsState: ignore ZwpLinuxBufferParamsV1);
delegate_noop!(WlrootsState: ignore ZwlrScreencopyManagerV1);
delegate_noop!(WlrootsState: ignore ZwlrVirtualPointerManagerV1);
delegate_noop!(WlrootsState: ignore ZwlrVirtualPointerV1);
delegate_noop!(WlrootsState: ignore ZwpVirtualKeyboardManagerV1);
delegate_noop!(WlrootsState: ignore ZwpVirtualKeyboardV1);

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	#[ignore = "requires a running wlroots compositor exposing screencopy and virtual input protocols"]
	fn smoke_capture_local_wlroots_frame() {
		let display = std::env::var("MOONSHINE_WLROOTS_SMOKE_DISPLAY")
			.expect("set MOONSHINE_WLROOTS_SMOKE_DISPLAY to a Wayland display name or socket path");
		let output = std::env::var("MOONSHINE_WLROOTS_SMOKE_OUTPUT").ok();
		let width = std::env::var("MOONSHINE_WLROOTS_SMOKE_WIDTH")
			.ok()
			.and_then(|value| value.parse().ok())
			.unwrap_or(1920);
		let height = std::env::var("MOONSHINE_WLROOTS_SMOKE_HEIGHT")
			.ok()
			.and_then(|value| value.parse().ok())
			.unwrap_or(1080);
		let refresh_rate = std::env::var("MOONSHINE_WLROOTS_SMOKE_REFRESH")
			.ok()
			.and_then(|value| value.parse().ok())
			.unwrap_or(60);

		let shutdown = ShutdownManager::new();
		let (frame_rx, input_tx, ready_rx) = start_capture(
			WlrootsCaptureConfig {
				display: Some(display),
				output,
				render_cursor: false,
			},
			KeyboardConfig::default(),
			width,
			height,
			refresh_rate,
			None,
			shutdown.clone(),
		)
		.expect("wlroots capture thread should start");

		let ready = ready_rx
			.recv_timeout(Duration::from_secs(5))
			.expect("wlroots capture should report readiness")
			.expect("wlroots capture should initialize");
		assert!(!ready.hdr);
		assert!(ready.width > 0);
		assert!(ready.height > 0);

		input_tx
			.send(CompositorInputEvent::MouseMoveRelative { dx: 0, dy: 0 })
			.expect("virtual pointer input should be accepted");

		let frame = frame_rx
			.recv_timeout(Duration::from_secs(2))
			.expect("wlroots capture should produce a frame");
		assert_eq!(frame.width, ready.width);
		assert_eq!(frame.height, ready.height);
		assert!(!frame.planes.is_empty());
		assert_eq!(frame.color_space, FrameColorSpace::Srgb);
		assert!(frame.hdr_metadata.is_none());
		frame.consumed.store(true, Ordering::Release);

		let _ = shutdown.trigger_shutdown(SessionShutdownReason::UserStopped);
	}
}
