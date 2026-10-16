from vulkan._vulkan import ffi

# Test pushing push constants with different byte formats
data = b'\x00' * 128

# Method 1: ffi.new with char array
buf1 = ffi.new('char[]', data)
print(f'char[] type: {type(buf1)}, len: {len(buf1)}')

# Method 2: Cast bytes to void*
ptr = ffi.cast('void *', ffi.from_buffer('char[]', data))
print(f'void* cast type: {type(ptr)}')

# Test with a simple command buffer call (won't actually run without device setup)
print("Both methods work for type handling")
