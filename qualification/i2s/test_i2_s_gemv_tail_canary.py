#!/usr/bin/env python3
"""Verify I2_S GEMV tail blocks never overwrite the output canary."""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.ggml_gemv_i2_i8_s.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
]
for rows in (1, 2, 3, 5, 15, 16, 17, 18, 31, 32, 33):
    n = 128
    # All-zero encoded weights; activation input is zero.
    packed = (ctypes.c_uint8 * (rows * n // 4))()
    acts = (ctypes.c_int8 * n)()
    canary = 123456.0
    out = (ctypes.c_float * (rows + 16))(*([canary] * (rows + 16)))
    lib.ggml_gemv_i2_i8_s(n, out, rows, packed, acts, 1, rows)
    assert all(out[i] == 0 for i in range(rows)), (rows, list(out[:rows]))
    assert all(out[i] == canary for i in range(rows, rows + 16)), (rows, list(out[rows:]))
    print(f"I2_S GEMV rows={rows} output-canary=PASS")
