"""Test what vkMapMemory returns."""
from v4_runtime.vulkan_runtime import VulkanRuntime
import vulkan as vk
import ctypes
import numpy as np

rt = VulkanRuntime()
rt.initialize()
rt.pick_physical_device()
rt.create_device()

# Create buffer without data
buffer_ci = vk.VkBufferCreateInfo(
    size=16,
    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
)
buffer = vk.vkCreateBuffer(rt.device, buffer_ci, None)
mem_reqs = vk.vkGetBufferMemoryRequirements(rt.device, buffer)

# Find host-visible memory
mem_props = vk.vkGetPhysicalDeviceMemoryProperties(rt.physical_device)
memory_type_index = None
for i in range(mem_props.memoryTypeCount):
    if (mem_reqs.memoryTypeBits & (1 << i)):
        if mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            memory_type_index = i
            break

alloc_info = vk.VkMemoryAllocateInfo(
    allocationSize=mem_reqs.size,
    memoryTypeIndex=memory_type_index,
)
memory = vk.vkAllocateMemory(rt.device, alloc_info, None)
vk.vkBindBufferMemory(rt.device, buffer, memory, 0)

# Map and check
mapped_ptr = vk.vkMapMemory(rt.device, memory, 0, mem_reqs.size, 0)
print(f"vkMapMemory result type: {type(mapped_ptr)}")
print(f"vkMapMemory result len: {len(mapped_ptr)}")

# Test writing data
test_floats = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
print(f"Test floats: {test_floats}")
print(f"Test bytes len: {len(test_floats.tobytes())}")

# Write via slice
mapped_ptr[:len(test_floats.tobytes())] = test_floats.tobytes()
print("Write succeeded")

# Verify by reading back
read_back = np.frombuffer(mapped_ptr[:16], dtype=np.float32)
print(f"Read back: {read_back}")

vk.vkUnmapMemory(rt.device, memory)
print("Buffer map/unmap/write/readback worked!")

rt.cleanup()
