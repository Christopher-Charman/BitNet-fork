"""Deterministic signed int8 extremes and 4096-element GEMM parity."""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_gemm_i2_i8_s.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
n, nc, nr = 4096, 4, 4
packed = ctypes.create_string_buffer(n * nc // 4 + 64)
weights = []
for row in range(nc):
    w = [float((-1, 0, 1)[(i + row) % 3]) for i in range(n)]
    weights.append(w)
    lib.quantize_i2_s((ctypes.c_float * n)(*w),
        ctypes.byref(packed, row * n // 4), 1, n, None)
for magnitude in (3, 31, 63, 100, 127):
    acts = [[-magnitude if i % 2 else magnitude for i in range(n)]
            for _ in range(nr)]
    y = (ctypes.c_int8 * (n * nr))(*(v for col in acts for v in col))
    out = (ctypes.c_float * (nc * nr))()
    lib.ggml_gemm_i2_i8_s(n, out, nc, packed, y, nr, nc)
    for col in range(nr):
        for row in range(nc):
            expected = sum((int(weights[row][i]) + 1) * acts[col][i]
                           for i in range(n))
            assert out[col * nc + row] == expected, (
                magnitude, col, row, out[col * nc + row], expected)
    print(f"I2_S GEMM signed activation magnitude={magnitude} parity=PASS")
