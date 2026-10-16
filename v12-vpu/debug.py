import struct
f = open('build/data/tr_pc.bin','rb').read()
print(f'PC len={len(f)}')
print('count=' + str(struct.unpack_from('<I', f, 0)[0]))
print('bias=' + str(struct.unpack_from('<4f', f, 80)))
print('scale=' + str(struct.unpack_from('<4f', f, 96)))
print('sbbias=' + str(struct.unpack_from('<4f', f, 112)))
import numpy as np
ref = np.frombuffer(open('build/data/tr_ref.bin','rb').read(), dtype=np.float32)
out = np.frombuffer(open('tr_out3.bin','rb').read(), dtype=np.float32)
print(f'ref len={len(ref)}, out len={len(out)}')
print(f'nonzero out: {np.count_nonzero(out)}/{len(out)}')
diffs = np.abs(ref - out)
worst = np.argmax(diffs)
print(f'worst idx={worst}: ref={ref[worst]}, out={out[worst]}, diff={diffs[worst]}')
tol_pass = np.all(diffs <= 1e-4 * np.maximum(1.0, np.abs(ref)))
print(f'tolerance check pass: {tol_pass}')
# Check if out is zero where ref is not
zero_where_not = np.where((out == 0) & (ref != 0))[0]
print(f'zero-but-ref-nonzero count: {len(zero_where_not)}')
if len(zero_where_not) > 0:
    print(f'first such idx: {zero_where_not[0]}, ref={ref[zero_where_not[0]]}')
