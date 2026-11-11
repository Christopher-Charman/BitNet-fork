#!/usr/bin/env python3
"""Apply the isolated I2_S source correction to the pinned llama.cpp submodule.

This script modifies three pinned ggml CPU source files (quants.c, ggml-cpu-i2s.c and ggml-cpu.c).
Run from the BitNet-fork repository root. Requires the pinned submodule checkout.
"""
from pathlib import Path

path = Path("3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c")
source = path.read_text()
replacements = [
    (
        """        int byte_idx = i / 4;
        int bit_pos = 6 - 2 * (i % 4);
        q[byte_idx] |= (val << bit_pos);""",
        """        // I2_S stores 128 values as four 32-value SIMD lanes. Each\n        // packed byte holds the same position from all four lanes.\n        // Match dequantize_row_i2_s and the AVX2 GEMM/GEMV consumers.
        const int64_t block = i / 128;
        const int64_t local = i % 128;
        const int64_t byte_idx = block * 32 + local % 32;
        const int bit_pos = 6 - 2 * (int)(local / 32);
        q[byte_idx] |= (val << bit_pos);""",
    ),
    (
        """            int byte_idx = i / 4;
            int bit_pos = 6 - 2 * (i % 4);
            int w = map2bit[(x[row * (n/4) + byte_idx] >> bit_pos) & 0x03];""",
        """            // Match the 128-element, four-lane I2_S packing used by\n            // the SIMD kernels and dequantize_row_i2_s.\n            const int block = i / 128;
            const int local = i % 128;
            const int byte_idx = block * 32 + local % 32;
            const int bit_pos = 6 - 2 * (local / 32);
            int w = map2bit[(x[row * (n/4) + byte_idx] >> bit_pos) & 0x03];""",
    ),
    (
        """    static const int map2bit[4] = { -1, 0, 1, 0 };""",
        """    // The caller subtracts the activation sum after this encoded dot.\n    // Match the SIMD path: codes 0/1/2, not signed -1/0/+1.
    static const int map2bit[4] = { 0, 1, 2, 1 };""",
    ),
]
for before, after in replacements:
    count = source.count(before)
    if count != 1:
        raise SystemExit(f"Expected exactly one matching source block, found {count}: {before[:60]}")
    source = source.replace(before, after)
path.write_text(source)
print(f"Updated {path}")

# The portable GEMM fallback previously forwarded multirow/multicolumn
# operations to a scalar dot routine that only supports one result.
gemm_path = Path("3rdparty/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-i2s.c")
gemm_source = gemm_path.read_text()
function_start = gemm_source.index("void ggml_gemm_i2_i8_s(")
start = gemm_source.index("#if defined(ACT_PARALLEL)", function_start)
end = gemm_source.index("#endif // __AVX2__", start)
old = gemm_source[start:end]
assert "ggml_vec_dot_i2_i8_s" in old and "for (int64_t r0" in old
replacement = """    // Portable scalar path: the scalar vector dot only supports nrc=1.
    // Each output is indexed as column * bs + weight row.
    for (int64_t c = 0; c < nr; ++c) {
        for (int64_t r = 0; r < nc; ++r) {
            ggml_vec_dot_i2_i8_s(n, s + c * bs + r, 1,
                (const uint8_t *)vx + r * n / 4, n,
                (const int8_t *)vy + c * n, 0, 1);
        }
    }
"""
gemm_path.write_text(gemm_source[:start] + replacement + gemm_source[end:])
print(f"Updated {gemm_path}")

# The non-repacked I2_S matmul path rounded its final row boundary
# upwards to a multiple of four without clamping it to the tensor's
# actual row count. For ne01=5 it accessed rows 5..7 out of bounds.
matmul_path = Path("3rdparty/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c")
matmul_source = matmul_path.read_text()
anchor = """        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols) : src0_end;
        if (src0_start >= src0_end) return;"""
corrected = """        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols) : src0_end;
        // The final tile can contain fewer than four physical weight rows.
        // Never read or write beyond the actual tensor row count.
        src0_end   = MIN(src0_end, ne01);
        if (src0_start >= src0_end) return;"""
if matmul_source.count(anchor) != 1:
    raise SystemExit("Expected exactly one non-repacked I2_S row-boundary block")
matmul_path.write_text(matmul_source.replace(anchor, corrected))
print(f"Updated {matmul_path}")

# The GEMV loop dispatches blocks of 16 rows, but its final block may
# contain fewer rows. Passing nrc=16 to the vector-dot kernel reads past
# the weight tensor and writes beyond the output tensor for tails 1..15.
# Clamp the final dispatch to the physical row count.
gemv_source = gemm_path.read_text()
gemv_anchor = """        ggml_vec_dot_i2_i8_s(n, s + iir0, 1, (const uint8_t *)vx + iir0 * n / 4, n, vy, 0, blck_0);"""
gemv_fix = """        const int64_t rows_remaining = nc - iir0;
        const int64_t rows_this_block = rows_remaining < blck_0 ? rows_remaining : blck_0;
        ggml_vec_dot_i2_i8_s(n, s + iir0, 1, (const uint8_t *)vx + iir0 * n / 4, n, vy, 0, (int)rows_this_block);"""
if gemv_source.count(gemv_anchor) != 1:
    raise SystemExit("Expected exactly one GEMV 16-row dispatch")
gemm_path.write_text(gemv_source.replace(gemv_anchor, gemv_fix))
print(f"Clamped final GEMV row block in {gemm_path}")
