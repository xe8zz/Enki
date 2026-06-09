use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUsage {
    DeviceOnly,
    Upload,
    Download,
    ZeroCopy,
}

impl MemoryUsage {
    pub fn to_vma(&self, is_unified_memory: bool) -> (vk_mem::MemoryUsage, vk_mem::AllocationCreateFlags) {
        match self {
            MemoryUsage::DeviceOnly => (
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::empty(),
            ),
            MemoryUsage::Upload => (
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ),
            MemoryUsage::Download => (
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
            ),
            MemoryUsage::ZeroCopy => {
                if is_unified_memory {
                    (
                        vk_mem::MemoryUsage::AutoPreferDevice,
                        vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    )
                } else {
                    (
                        vk_mem::MemoryUsage::AutoPreferHost,
                        vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    )
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    Storage,
    Uniform,
    TransferSrc,
    TransferDst,
    Indirect,
    Vertex,
    Index,
}

impl BufferUsage {
    pub fn to_vk(&self) -> vk::BufferUsageFlags {
        match self {
            BufferUsage::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferUsage::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUsage::TransferSrc => vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsage::TransferDst => vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsage::Indirect => vk::BufferUsageFlags::INDIRECT_BUFFER,
            BufferUsage::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferUsage::Index => vk::BufferUsageFlags::INDEX_BUFFER,
        }
    }
}