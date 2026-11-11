"""I2_S multirow GEMM parity against encoded 0/1/2 reference.

Run against both AVX2 and scalar CPU-only ggml-cpu libraries.
"""
import ctypes
import os

lib = ctypes.CDLL(os.environ["BITNET_GGML_CPU_LIB"])
lib.quantize_i2_s.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                               ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.quantize_i2_s.restype = ctypes.c_size_t
lib.ggml_gemm_i2_i8_s.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

for n in (128, 256, 4096):
    for nc, nr in ((4, 4), (5, 3), (3, 5)):
        packed = ctypes.create_string_buffer(n // 4 * nc + 128)
        weights = []
        for row in range(nc):
            w = [float((-1, 0, 1, 0)[(i * 17 + row + i // 32) % 4]) for i in range(n)]
            weights.append(w)
            src = (ctypes.c_float * n)(*w)
            lib.quantize_i2_s(src, ctypes.byref(packed, row * n // 4), 1, n, None)
        acts = [[int((i + col) % 7 - 3) for i in range(n)] for col in range(nr)]
        flat = (ctypes.c_int8 * (n * nr))(*(v for col in acts for v in col))
        output = (ctypes.c_float * (nc * nr))()
        lib.ggml_gemm_i2_i8_s(n, output, nc, packed, flat, nr, nc)
        for col in range(nr):
            for row in range(nc):
                expected = sum((int(weights[row][i]) + 1) * acts[col][i] for i in range(n))
                assert output[col * nc + row] == expected, (
                    n, nc, nr, col, row, output[col * nc + row], expected)
        print(f"I2_S GEMM n={n} rows={nc} columns={nr} parity=PASS")
