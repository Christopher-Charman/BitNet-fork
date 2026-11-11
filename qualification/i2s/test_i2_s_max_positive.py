"""Verify worst-case positive int8 accumulation without signed saturation."""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_gemm_i2_i8_s.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

for n in (128, 4096):
    rows = columns = 4
    packed = ctypes.create_string_buffer(n * rows // 4 + 64)
    for row in range(rows):
        src = (ctypes.c_float * n)(*([1.0] * n))
        lib.quantize_i2_s(src, ctypes.byref(packed, row * n // 4), 1, n, None)
    y = (ctypes.c_int8 * (n * columns))(*([127] * (n * columns)))
    out = (ctypes.c_float * (rows * columns))()
    lib.ggml_gemm_i2_i8_s(n, out, rows, packed, y, columns, rows)
    expected = float(2 * 127 * n)
    assert all(v == expected for v in out), (n, list(out), expected)
    print(f"I2_S GEMM max-positive n={n} expected={expected:.0f} parity=PASS")
