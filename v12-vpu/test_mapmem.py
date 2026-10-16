"""Test what vkMapMemory returns."""
from v4_runtime.vulkan_runtime import VulkanRuntime
import vulkan as vk

rt = VulkanRuntime()
rt.initialize()
rt.pick_physical_device()
rt.create_device()

# Create a small buffer
rt.create_buffer('test_buf', 16, bytearray(b'\x00' * 16), host_visible=True)

# Try to map it manually
buffer, memory, size = rt.buffers['test_buf']
print(f"Buffer: {buffer}, Memory: {memory}, Size: {size}")

result = vk.vkMapMemory(rt.device, memory, 0, size, 0)
print(f"vkMapMemory result type: {type(result)}")
print(f"vkMapMemory result value: {result}")

# Try different approaches to identify the mapping
if hasattr(result, '__class__'):
    print(f"Class: {result.__class__.__name__}")
    print(f"Dict: {vars(result) if not isinstance(result, (int, type(None))) else 'N/A'}")
    print(f"Dir: {[x for x in dir(result) if not x.startswith('_')][:20]}")

rt.cleanup()
