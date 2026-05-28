//! RGB image blitting for wlroots captures whose source size differs from the
//! negotiated stream size.

use ash::vk;
use pixelforge::VideoContext;

use crate::session::compositor::frame::FrameTransform;

pub struct RgbBlitter {
	context: VideoContext,
	format: vk::Format,
	width: u32,
	height: u32,
	image: vk::Image,
	memory: vk::DeviceMemory,
	command_pool: vk::CommandPool,
	command_buffer: vk::CommandBuffer,
	fence: vk::Fence,
	initialized: bool,
}

impl RgbBlitter {
	pub fn new(context: VideoContext, format: vk::Format, width: u32, height: u32) -> Result<Self, String> {
		let device = context.device();
		let width = width.max(1);
		let height = height.max(1);

		let image_info = vk::ImageCreateInfo::default()
			.image_type(vk::ImageType::TYPE_2D)
			.format(format)
			.extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.tiling(vk::ImageTiling::OPTIMAL)
			.usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
			.sharing_mode(vk::SharingMode::EXCLUSIVE)
			.initial_layout(vk::ImageLayout::UNDEFINED);

		let image =
			unsafe { device.create_image(&image_info, None) }.map_err(|e| format!("RGB blit image creation: {e}"))?;

		let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
		let memory_type_index = context
			.find_memory_type(mem_requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
			.ok_or_else(|| "No suitable memory type for RGB blit image".to_string())?;

		let alloc_info = vk::MemoryAllocateInfo::default()
			.allocation_size(mem_requirements.size)
			.memory_type_index(memory_type_index);

		let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
			unsafe { device.destroy_image(image, None) };
			format!("RGB blit memory allocation: {e}")
		})?;

		if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
			unsafe {
				device.free_memory(memory, None);
				device.destroy_image(image, None);
			}
			return Err(format!("RGB blit memory bind: {e}"));
		}

		let pool_info = vk::CommandPoolCreateInfo::default()
			.queue_family_index(context.compute_queue_family())
			.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

		let command_pool = unsafe { device.create_command_pool(&pool_info, None) }.map_err(|e| {
			unsafe {
				device.free_memory(memory, None);
				device.destroy_image(image, None);
			}
			format!("RGB blit command pool: {e}")
		})?;

		let alloc_info = vk::CommandBufferAllocateInfo::default()
			.command_pool(command_pool)
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(1);

		let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }.map_err(|e| {
			unsafe {
				device.destroy_command_pool(command_pool, None);
				device.free_memory(memory, None);
				device.destroy_image(image, None);
			}
			format!("RGB blit command buffer: {e}")
		})?[0];

		let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| {
			unsafe {
				device.destroy_command_pool(command_pool, None);
				device.free_memory(memory, None);
				device.destroy_image(image, None);
			}
			format!("RGB blit fence: {e}")
		})?;

		Ok(Self {
			context,
			format,
			width,
			height,
			image,
			memory,
			command_pool,
			command_buffer,
			fence,
			initialized: false,
		})
	}

	pub fn format(&self) -> vk::Format {
		self.format
	}

	pub fn width(&self) -> u32 {
		self.width
	}

	pub fn height(&self) -> u32 {
		self.height
	}

	pub fn blit_aspect_fit(
		&mut self,
		src_image: vk::Image,
		src_layout: vk::ImageLayout,
		src_width: u32,
		src_height: u32,
		transform: FrameTransform,
	) -> Result<vk::Image, String> {
		let device = self.context.device();
		let src_width = src_width.max(1);
		let src_height = src_height.max(1);
		let dst_rect = aspect_fit_rect(src_width, src_height, self.width, self.height);
		let image_range = vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlags::COLOR,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		};
		let src_access = if src_layout == vk::ImageLayout::UNDEFINED {
			vk::AccessFlags::empty()
		} else {
			vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
		};
		let src_stage = if src_layout == vk::ImageLayout::UNDEFINED {
			vk::PipelineStageFlags::TOP_OF_PIPE
		} else {
			vk::PipelineStageFlags::ALL_COMMANDS
		};
		let dst_old_layout = if self.initialized {
			vk::ImageLayout::GENERAL
		} else {
			vk::ImageLayout::UNDEFINED
		};
		let dst_src_access = if self.initialized {
			vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
		} else {
			vk::AccessFlags::empty()
		};
		let dst_src_stage = if self.initialized {
			vk::PipelineStageFlags::ALL_COMMANDS
		} else {
			vk::PipelineStageFlags::TOP_OF_PIPE
		};

		unsafe {
			device
				.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
				.map_err(|e| format!("RGB blit reset command buffer: {e}"))?;

			let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			device
				.begin_command_buffer(self.command_buffer, &begin_info)
				.map_err(|e| format!("RGB blit begin command buffer: {e}"))?;

			let src_to_transfer = vk::ImageMemoryBarrier::default()
				.old_layout(src_layout)
				.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
				.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.image(src_image)
				.subresource_range(image_range)
				.src_access_mask(src_access)
				.dst_access_mask(vk::AccessFlags::TRANSFER_READ);

			let dst_to_transfer = vk::ImageMemoryBarrier::default()
				.old_layout(dst_old_layout)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.image(self.image)
				.subresource_range(image_range)
				.src_access_mask(dst_src_access)
				.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

			device.cmd_pipeline_barrier(
				self.command_buffer,
				src_stage | dst_src_stage,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[src_to_transfer, dst_to_transfer],
			);

			let clear_color = vk::ClearColorValue {
				float32: [0.0, 0.0, 0.0, 1.0],
			};
			device.cmd_clear_color_image(
				self.command_buffer,
				self.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&clear_color,
				&[image_range],
			);

			let clear_to_blit = vk::ImageMemoryBarrier::default()
				.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.image(self.image)
				.subresource_range(image_range)
				.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
				.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

			device.cmd_pipeline_barrier(
				self.command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[clear_to_blit],
			);

			let blit = vk::ImageBlit {
				src_subresource: vk::ImageSubresourceLayers {
					aspect_mask: vk::ImageAspectFlags::COLOR,
					mip_level: 0,
					base_array_layer: 0,
					layer_count: 1,
				},
				src_offsets: source_offsets(src_width, src_height, transform),
				dst_subresource: vk::ImageSubresourceLayers {
					aspect_mask: vk::ImageAspectFlags::COLOR,
					mip_level: 0,
					base_array_layer: 0,
					layer_count: 1,
				},
				dst_offsets: [
					vk::Offset3D {
						x: dst_rect.x as i32,
						y: dst_rect.y as i32,
						z: 0,
					},
					vk::Offset3D {
						x: (dst_rect.x + dst_rect.width) as i32,
						y: (dst_rect.y + dst_rect.height) as i32,
						z: 1,
					},
				],
			};

			device.cmd_blit_image(
				self.command_buffer,
				src_image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				self.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[blit],
				vk::Filter::LINEAR,
			);

			let src_to_general = vk::ImageMemoryBarrier::default()
				.old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
				.new_layout(vk::ImageLayout::GENERAL)
				.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.image(src_image)
				.subresource_range(image_range)
				.src_access_mask(vk::AccessFlags::TRANSFER_READ)
				.dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

			let dst_to_general = vk::ImageMemoryBarrier::default()
				.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(vk::ImageLayout::GENERAL)
				.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.image(self.image)
				.subresource_range(image_range)
				.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
				.dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

			device.cmd_pipeline_barrier(
				self.command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::ALL_COMMANDS,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[src_to_general, dst_to_general],
			);

			device
				.end_command_buffer(self.command_buffer)
				.map_err(|e| format!("RGB blit end command buffer: {e}"))?;

			device
				.reset_fences(&[self.fence])
				.map_err(|e| format!("RGB blit reset fence: {e}"))?;

			let command_buffers = [self.command_buffer];
			let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
			device
				.queue_submit(self.context.compute_queue(), &[submit_info], self.fence)
				.map_err(|e| format!("RGB blit queue submit: {e}"))?;

			device
				.wait_for_fences(&[self.fence], true, u64::MAX)
				.map_err(|e| format!("RGB blit wait fence: {e}"))?;
		}

		self.initialized = true;
		Ok(self.image)
	}
}

impl Drop for RgbBlitter {
	fn drop(&mut self) {
		let device = self.context.device();
		unsafe {
			device.destroy_fence(self.fence, None);
			device.destroy_command_pool(self.command_pool, None);
			device.destroy_image(self.image, None);
			device.free_memory(self.memory, None);
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlitRect {
	x: u32,
	y: u32,
	width: u32,
	height: u32,
}

fn aspect_fit_rect(source_width: u32, source_height: u32, output_width: u32, output_height: u32) -> BlitRect {
	let source_width = source_width.max(1) as f64;
	let source_height = source_height.max(1) as f64;
	let output_width = output_width.max(1);
	let output_height = output_height.max(1);
	let scale = (output_width as f64 / source_width).min(output_height as f64 / source_height);
	let width = (source_width * scale).round().max(1.0).min(output_width as f64) as u32;
	let height = (source_height * scale).round().max(1.0).min(output_height as f64) as u32;
	let x = (output_width - width) / 2;
	let y = (output_height - height) / 2;

	BlitRect { x, y, width, height }
}

fn source_offsets(src_width: u32, src_height: u32, transform: FrameTransform) -> [vk::Offset3D; 2] {
	let left = if transform.flip_x { src_width as i32 } else { 0 };
	let right = if transform.flip_x { 0 } else { src_width as i32 };
	let top = if transform.flip_y { src_height as i32 } else { 0 };
	let bottom = if transform.flip_y { 0 } else { src_height as i32 };

	[
		vk::Offset3D { x: left, y: top, z: 0 },
		vk::Offset3D {
			x: right,
			y: bottom,
			z: 1,
		},
	]
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn aspect_fit_rect_preserves_portrait_capture_in_landscape_stream() {
		assert_eq!(
			aspect_fit_rect(400, 640, 1920, 1080),
			BlitRect {
				x: 622,
				y: 0,
				width: 675,
				height: 1080,
			}
		);
	}

	#[test]
	fn aspect_fit_rect_uses_full_stream_for_matching_aspect_ratio() {
		assert_eq!(
			aspect_fit_rect(1920, 1080, 1280, 720),
			BlitRect {
				x: 0,
				y: 0,
				width: 1280,
				height: 720,
			}
		);
	}

	#[test]
	fn source_offsets_apply_180_degree_flip() {
		let offsets = source_offsets(
			1920,
			1080,
			FrameTransform {
				flip_x: true,
				flip_y: true,
			},
		);

		assert_eq!((offsets[0].x, offsets[0].y), (1920, 1080));
		assert_eq!((offsets[1].x, offsets[1].y), (0, 0));
	}
}
