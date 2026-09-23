"""Fused tensor-core attention on packed rows: layout, masking, padding tokens and graph replay."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from laya_cuda.ops import Ops
from laya_cuda.runtime import span

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("lengths,window", [([32], -1), ([7, 96], 64), ([128, 1], -1), ([250, 131], 64),
                                            ([256, 219, 145], -1), ([500], 64), ([1024, 333], -1)])
def test_packed_attention_matches_oracle(lengths, window):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        op = Ops(stream)
        try:
            h = 2
            starts = np.cumsum([0]+[span(n) for n in lengths])[:-1]
            m = span(sum(span(n) for n in lengths)+16)  # include tokens past the last row
            rows = np.full(m, -1, np.int32)
            for row, (start, n) in enumerate(zip(starts, lengths)):
                rows[start:start+span(n)] = row
            # Token-major Wqkv output: [m, (q, k, v), h, 64].
            x = np.random.default_rng(m).normal(0, 1, (m, 3, h, 64)).astype(np.float16)
            source, layout = cp.asarray(x), cp.empty(x.size, cp.float16)
            inputs = [cp.asarray(np.asarray(v, np.int32)) for v in (rows, starts, lengths)]
            out = cp.full((m, h*64), np.nan, cp.float16)

            def run():
                op.elements("layout", x.size, source, layout, None, None, inputs[0], m, h)
                op.call("attention", (h*m//16+3)//4, layout, out, *inputs, m, h, window, threads=128)
            run()
            stream.synchronize()
            eager = out.get()
            stream.begin_capture()
            run()
            graph = stream.end_capture()
            out.fill(np.nan)
            graph.launch(stream)
            stream.synchronize()
            np.testing.assert_array_equal(out.get(), eager)
            real = np.zeros(m, bool)
            for start, n in zip(starts, lengths):
                real[start:start+n] = True
                q, k, v = x[start:start+n].astype(np.float64).transpose(1, 2, 0, 3)
                s = q @ k.transpose(0, 2, 1)/8
                if window >= 0:
                    s[:, abs(np.arange(n)[:, None]-np.arange(n)) > window] = -np.inf
                p = np.exp(s-s.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                expected = (p @ v).transpose(1, 0, 2).reshape(n, h*64)
                # FP16 output rounding dominates; probabilities keep FP32-level precision.
                np.testing.assert_allclose(eager[start:start+n], expected, atol=6e-4, rtol=1e-3)
            np.testing.assert_array_equal(eager[~real], 0)
        finally:
            stream.synchronize()
            op.close()
