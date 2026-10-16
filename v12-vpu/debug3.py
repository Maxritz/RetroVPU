import struct
pc = bytearray(128)
struct.pack_into('<I', pc, 0, 64)
open('build/data/pctest_pc.bin','wb').write(pc)
print('PC written, count at offset 0:', struct.unpack_from('<I', pc, 0)[0])
