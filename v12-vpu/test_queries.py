"""Test different ways to call vkGetQueryPoolResults."""
import sys
sys.path.insert(0, '.')
from v4_runtime.vulkan_runtime import VulkanRuntime
import vulkan as vk
import numpy as np
import struct
import ctypes

rt = VulkanRuntime()
rt.initialize()
rt.pick_physical_device()
rt.create_device()
rt.create_command_pool()

# Test query pool directly
query_pool_ci = vk.VkQueryPoolCreateInfo(
    queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
    queryCount=2,
)
query_pool = vk.vkCreateQueryPool(rt.device, query_pool_ci, None)
print(f"Query pool created: {query_pool}")

# Create command buffer and write timestamps
cmd_ci = vk.VkCommandBufferAllocateInfo(
    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandPool=rt.command_pool,
    commandBufferCount=1,
)
cmd_buf = vk.vkAllocateCommandBuffers(rt.device, cmd_ci)[0]

begin_info = vk.VkCommandBufferBeginInfo(
    flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
)
vk.vkBeginCommandBuffer(cmd_buf, begin_info)

# Write two timestamps
vk.vkCmdWriteTimestamp(cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0)
vk.vkCmdWriteTimestamp(cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1)

vk.vkEndCommandBuffer(cmd_buf)

# Submit
fence_ci = vk.VkFenceCreateInfo()
fence = vk.vkCreateFence(rt.device, fence_ci, None)

submit_info = vk.VkSubmitInfo(
    commandBufferCount=1,
    pCommandBuffers=[cmd_buf],
)
vk.vkQueueSubmit(rt.compute_queue, 1, [submit_info], fence)
vk.vkWaitForFences(rt.device, 1, [fence], vk.VK_TRUE, 5000000000)

print("Fence signaled")

# Try different ways to read results
print("\nMethod 1: ctypes array directly")
arr = (ctypes.c_uint64 * 2)()
try:
    result = vk.vkGetQueryPoolResults(rt.device, query_pool, 0, 2,
                                     16, arr, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
    print(f"Result: {result}")
    print(f"Values: {arr[0]}, {arr[1]}")
except Exception as e:
    print(f"Failed: {e}")

print("\nMethod 2: numpy array")
np_arr = np.zeros(2, dtype=np.uint64)
try:
    result = vk.vkGetQueryPoolResults(rt.device, query_pool, 0, 2,
                                     np_arr.nbytes, np_arr, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
    print(f"Result: {result}")
    print(f"Values: {np_arr[0]}, {np_arr[1]}")
except Exception as e:
    print(f"Failed: {e}")

print("\nMethod 3: ffi buffer")
from vulkan._vulkan import ffi
buf = ffi.new('uint64_t[2]')
try:
    result = vk.vkGetQueryPoolResults(rt.device, query_pool, 0, 2,
                                     16, buf, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
    print(f"Result: {result}")
    print(f"Values: {buf[0]}, {buf[1]}")
except Exception as e:
    print(f"Failed: {e}")

vk.vkDestroyFence(rt.device, fence, None)
vk.vkDestroyQueryPool(rt.device, query_pool, None)
rt.cleanup()
