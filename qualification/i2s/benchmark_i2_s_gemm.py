"""Low-impact I2_S GEMM microbenchmark, CPU-only.

Measures only the kernel on the current host, not end-to-end inference.
Use BITNET_GGML_CPU_LIB to select an independently built backend.
"""
import ctypes
import os
import statistics
import time

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_gemm_i2_i8_s.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
for n, nc, nr in ((128, 8, 8), (512, 8, 8), (4096, 8, 8)):
    packed = ctypes.create_string_buffer(n * nc // 4 + 64)
    # Quantize each row into a separate temporary buffer: the quantizer
    # appends a scale after the packed row, overlapping the next row otherwise.
    for row in range(nc):
        values = (ctypes.c_float * n)(
            *(float((-1, 0, 1)[(i * 17 + row + i // 32) % 3]) for i in range(n)))
        row_packed = ctypes.create_string_buffer(n // 4 + 32)
        lib.quantize_i2_s(values, row_packed, 1, n, None)
        ctypes.memmove(ctypes.byref(packed, row * n // 4), row_packed, n // 4)
    activations = (ctypes.c_int8 * (n * nr))(
        *(int((i + col) % 7 - 3) for col in range(nr) for i in range(n)))
    output = (ctypes.c_float * (nc * nr))()
    def run():
        lib.ggml_gemm_i2_i8_s(n, output, nc, packed, activations, nr, nc)
    for _ in range(3):
        run()
    samples = []
    for _ in range(15):
        start = time.perf_counter_ns()
        run()
        samples.append(time.perf_counter_ns() - start)
    median_us = statistics.median(samples) / 1000
    print(f"I2_S GEMM n={n} rows={nc} columns={nr} median_us={median_us:.2f} "
          f"min_us={min(samples)/1000:.2f} samples={len(samples)}")
