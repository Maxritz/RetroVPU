"""
Debug test for Vulkan compute dispatch with debug validation layers.
Uses vulkaninfo-confirmed 1.4.363 capabilities.
"""
import sys
sys.path.insert(0, '.')
from v4_runtime.vulkan_runtime import VulkanRuntime
import numpy as np
import struct

def main():
    print("=== Vulkan Compute Debug Test ===")
    
    rt = VulkanRuntime()
    rt.initialize()
    rt.pick_physical_device()
    rt.create_device()
    rt.create_command_pool()
    
    # Create buffers
    test_size = 256
    input_data = np.random.randn(test_size).astype(np.float32) * 0.3
    
    # Input buffer (host-visible for upload + readback)
    rt.create_buffer('in', test_size * 4, input_data, host_visible=True)
    rt.create_buffer('weight', test_size * 4, np.ones(test_size, dtype=np.float32), host_visible=True)
    rt.create_buffer('out', test_size * 4, host_visible=True)

    # Load and setup kernel
    shader = rt.load_shader('rmsnorm', 'shaders/generated/rmsnorm.spv')
    rt.create_compute_pipeline('rmsnorm', shader, push_const_size=128, bindings=[0, 1, 2])
    rt.allocate_descriptor_set('rmsnorm', ['in', 'weight', 'out'])
    
    # Push constants: count=test_size, others=0
    pc_data = struct.pack('<IIII', test_size, 0, 0, 0) + b'\x00' * 112
    print(f"Push constants: {len(pc_data)} bytes, count={test_size}")
    
    # Dispatch with 1 workgroup (128 elements = 2 batches of 64)
    # rmsnorm processes one row per workgroup
    workgroup_count = 1  # Use single workgroup for batch=0
    print(f"Dispatching with {workgroup_count} workgroups...")
    
    gpu_us = rt.dispatch_compute('rmsnorm', pc_data, (workgroup_count, 1, 1))
    print(f"GPU time: {gpu_us:.2f} us")
    
    # Read back output
    print("Reading output buffer...")
    output_view = rt.read_buffer('out')
    print(f"Output shape: {output_view.shape}")
    print(f"Output sample: {output_view[:8]}")
    
    # CPU reference
    mean_sq = np.mean(input_data * input_data)
    expected_norm = 1.0 / np.sqrt(mean_sq + 1e-6)
    expected = input_data * expected_norm
    print(f"CPU expected: {expected[:8]}")
    print(f"Match: {np.allclose(output_view, expected, atol=1e-5)}")
    
    rt.cleanup()
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
