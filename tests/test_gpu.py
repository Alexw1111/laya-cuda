import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from laya_cuda.ops import Ops

pytestmark = pytest.mark.gpu


@pytest.fixture
def ops():
    if cp.cuda.runtime.getDeviceCount()<1:
        pytest.skip("CUDA required")
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        op = Ops(stream)
        yield op
        stream.synchronize()
        op.close()


def test_gemm_and_graph(ops):
    rng = np.random.default_rng(7)
    x,w = [rng.normal(size=s).astype(np.float16) for s in [(7,64),(32,64)]]
    a,b,out = cp.asarray(x),cp.asarray(w),cp.empty((7,32),cp.float16)
    ops.linear(a,b,out)
    ops.stream.synchronize()
    ops.stream.begin_capture()
    ops.linear(a,b,out)
    graph = ops.stream.end_capture()
    graph.launch(ops.stream)
    ops.stream.synchronize()
    np.testing.assert_allclose(out.get(),(x.astype(np.float32)@w.astype(np.float32).T).astype(np.float16),atol=.008,rtol=.002)


def test_rope_preserves_fp32_until_final_cast(ops):
    rng=np.random.default_rng(17)
    b,h,l=1,2,768
    x=rng.normal(size=(b,l,3,h,64)).astype(np.float16)
    angle=np.arange(l,dtype=np.float32)[:,None]/(np.float32(10000)**(np.arange(0,64,2,dtype=np.float32)/64))
    cosine,sine=np.cos(angle),np.sin(angle)
    out=cp.empty((3,b,h,l,64),cp.float16)
    ops.elements("layout",x.size,cp.asarray(x),out,cp.asarray(cosine),cp.asarray(sine),cp.arange(l,dtype=cp.int32),l,h)
    ops.stream.synchronize()
    expected=x.transpose(2,0,3,1,4).copy()
    for part in (0,1):
        original=expected[part].astype(np.float32)
        rotated=np.concatenate((-original[...,32:],original[...,:32]),axis=-1)
        expected[part]=(original*np.concatenate((cosine,cosine),axis=-1)+rotated*np.concatenate((sine,sine),axis=-1)).astype(np.float16)
    actual=out.get()
    np.testing.assert_array_equal(actual[:2],expected[:2])
    # Values interleave key pairs, [b][h][l/2][64][2], for the attention MMA.
    np.testing.assert_array_equal(actual[2].reshape(b,h,l//2,64,2),expected[2].reshape(b,h,l//2,2,64).transpose(0,1,2,4,3))


def test_linear_adds_bias_before_rounding(ops):
    x=np.zeros((1,64),np.float16);x[0,:2]=[1,.0006]
    weight=np.ones((32,64),np.float16)
    bias=np.full(32,-1,np.float16)
    out=cp.empty((1,32),cp.float16)
    scratch=cp.empty(out.shape,cp.float32)
    ops.linear(cp.asarray(x),cp.asarray(weight),out,cp.asarray(bias),scratch)
    ops.stream.synchronize()
    oracle=(x.astype(np.float32)@weight.astype(np.float32).T+bias.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(out.get(),oracle)


def test_layernorm(ops):
    rng=np.random.default_rng(11)
    x=rng.normal(size=(3,768)).astype(np.float32)
    gamma=rng.normal(size=768).astype(np.float32)
    residual,out=cp.asarray(x),cp.empty_like(cp.asarray(x),dtype=cp.float16)
    ops.call("norm",3,residual,None,cp.asarray(gamma),None,out,768,1e-5,0)
    ops.stream.synchronize()
    expected=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*gamma
    np.testing.assert_allclose(out.get(),expected,atol=.004,rtol=.002)


@pytest.mark.parametrize("width", [64, 768, 1024, 1088])
@pytest.mark.parametrize("normalize,out32", [(False, False), (False, True), (True, False), (True, True)])
def test_layernorm_residual_and_aliases(ops, width, normalize, out32):
    rng = np.random.default_rng(31)
    source = rng.normal(2, .3, (3, width)).astype(np.float32)
    branch = rng.normal(0, .1, source.shape).astype(np.float16)
    gamma = rng.normal(1, .1, width).astype(np.float32)
    beta = rng.normal(0, .1, width).astype(np.float32)
    residual = cp.asarray(source)
    output = residual if out32 else cp.empty(source.shape, cp.float16)
    ops.call("norm", 3, residual, cp.asarray(branch), cp.asarray(gamma) if normalize else None,
             cp.asarray(beta) if normalize else None, output, width, 1e-5, int(out32))
    ops.stream.synchronize()
    added = source + branch.astype(np.float32)
    expected = added
    if normalize:
        expected = (added-added.mean(-1, keepdims=True))/np.sqrt(added.var(-1, keepdims=True)+1e-5)
        expected = expected*gamma+beta
    expected = expected.astype(np.float32 if out32 else np.float16)
    if normalize:
        np.testing.assert_allclose(output.get(), expected, atol=.004, rtol=.002)
    else:
        np.testing.assert_array_equal(output.get(), expected)
    if not out32:
        np.testing.assert_array_equal(residual.get(), added)


def test_selected_head_rows_preserve_order_and_duplicates(ops):
    values=np.arange(8*64,dtype=np.float16).reshape(8,64)
    indices=np.array([0,7,3,3,1],np.int32)
    out=cp.empty((len(indices),64),cp.float16)
    ops.elements("gather_half",out.size,cp.asarray(values),cp.asarray(indices),out,len(indices),64)
    ops.stream.synchronize()
    np.testing.assert_array_equal(out.get(),values[indices])
