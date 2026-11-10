"""CPU-only I2_S packing, scalar/SIMD GEMV numerical regression.

Run after apply_pinned_llama_patch.py and rebuilding ggml-cpu:
BITNET_GGML_CPU_LIB=/absolute/path/to/libggml-cpu.so python qualification/i2s/test_i2_s_roundtrip.py
"""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
]
lib.quantize_i2_s.restype = ctypes.c_size_t
lib.dequantize_row_i2_s.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_float,
]
lib.ggml_vec_dot_i2_i8_s.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
for n in (128, 256, 512):
    expected = [float((-1, 0, 1, 0, 1, -1, 0, 1)[(i * 17 + i // 32) % 8]) for i in range(n)]
    src = (ctypes.c_float * n)(*expected)
    packed = ctypes.create_string_buffer(n // 4 + 32)
    decoded = (ctypes.c_float * n)()
    written = lib.quantize_i2_s(src, packed, 1, n, None)
    assert written <= len(packed), (written, len(packed))
    scale = ctypes.c_float.from_buffer(packed, n // 4).value
    lib.dequantize_row_i2_s(packed, decoded, n, scale)
    mismatches = sum(abs(decoded[i] - expected[i]) > 1e-6 for i in range(n))
    assert mismatches == 0, (n, mismatches)
    print(f"I2_S n={n} mismatches={mismatches} scale={scale} bytes={written}")

    acts = [int(i % 7 - 3) for i in range(n)]
    y = (ctypes.c_int8 * n)(*acts)
    result = (ctypes.c_float * 1)()
    lib.ggml_vec_dot_i2_i8_s(n, result, 1, packed, n, y, n, 1)
    expected_encoded = sum((int(expected[i]) + 1) * acts[i] for i in range(n))
    assert result[0] == expected_encoded, (n, result[0], expected_encoded)
    print(f"I2_S GEMV n={n} encoded_dot={result[0]} parity=PASS")
