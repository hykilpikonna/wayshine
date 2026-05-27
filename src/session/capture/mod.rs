pub mod wlroots;

use super::compositor::input::CompositorInputEvent;

#[derive(Clone)]
pub enum CaptureInputSender {
	Compositor(calloop::channel::Sender<CompositorInputEvent>),
	Wlroots(std::sync::mpsc::Sender<CompositorInputEvent>),
}

impl CaptureInputSender {
	pub fn send(&self, event: CompositorInputEvent) -> Result<(), ()> {
		match self {
			Self::Compositor(tx) => tx
				.send(event)
				.map_err(|e| tracing::warn!("Failed to send compositor input: {e}")),
			Self::Wlroots(tx) => tx
				.send(event)
				.map_err(|e| tracing::warn!("Failed to send wlroots input: {e}")),
		}
	}
}
