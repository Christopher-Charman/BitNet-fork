"""Multirow I2_S GEMV parity across the 16-row boundary on CPU backends."""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_gemv_i2_i8_s.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

for n in (128, 256, 4096):
    for nc in (1, 4, 16, 17, 32):
        packed = ctypes.create_string_buffer(n // 4 * nc + 64)
        weights = []
        for row in range(nc):
            w = [float((-1, 0, 1, 0)[(i * 17 + row + i // 32) % 4]) for i in range(n)]
            weights.append(w)
            src = (ctypes.c_float * n)(*w)
            lib.quantize_i2_s(src, ctypes.byref(packed, row * n // 4), 1, n, None)
        acts = [int(i % 7 - 3) for i in range(n)]
        y = (ctypes.c_int8 * n)(*acts)
        out = (ctypes.c_float * (nc + 16))()
        lib.ggml_gemv_i2_i8_s(n, out, nc, packed, y, 1, nc)
        for row in range(nc):
            expected = sum((int(weights[row][i]) + 1) * acts[i] for i in range(n))
            assert out[row] == expected, (n, nc, row, out[row], expected)
        print(f"I2_S GEMV n={n} rows={nc} parity=PASS")
