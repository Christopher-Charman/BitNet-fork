"""Check I2_S matmul thread tile boundaries against physical tensor rows.

The original final-tile ceiling produced out-of-bounds row accesses for
non-multiples of four (for example five rows rounded to eight).
"""
from pathlib import Path
import re

path = Path("3rdparty/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c")
source = path.read_text()
section = source.split("// I2_S GEMV/GEMM fast path:", 1)[1].split(
    "const float * scale", 1)[0]
assert "src0_end   = MIN(src0_end, ne01);" in section
assert "src0_start = (src0_start % matmul_num_cols)" in section

for rows in range(1, 34):
    for threads in range(1, 17):
        covered = []
        for thread in range(threads):
            start = (thread * rows) // threads
            end = ((thread + 1) * rows) // threads
            start = (start + 3) // 4 * 4
            end = min((end + 3) // 4 * 4, rows)
            if start < end:
                assert 0 <= start < end <= rows, (rows, threads, start, end)
                covered.extend(range(start, end))
        assert covered == list(range(rows)), (rows, threads, covered)
print("I2_S non-repacked matmul row bounds: 528 partition configurations PASS")
