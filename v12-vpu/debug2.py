import numpy as np
out = np.frombuffer(open('tr_out4.bin','rb').read(), dtype=np.float32)
print(f'Nonzero: {np.count_nonzero(out)}/{len(out)}')
print(f'Values: {out[:20]}')
