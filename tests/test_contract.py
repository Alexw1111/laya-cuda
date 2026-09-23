import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from laya_cuda.laya import Laya, question, softmax


def test_import_is_light():
    subprocess.run([sys.executable,"-c", """
import sys
import laya_cuda
assert not {'torch', 'transformers', 'cupy', 'psutil', 'pyarrow', 'benchmarks', 'tools'} & sys.modules.keys()
assert set(laya_cuda.__all__) == {'Engine', 'BatchEngine'}
"""],check=True)


def test_kernel_source_is_ascii():
    # CuPy writes kernel source in the process locale's encoding, which a native library can reset to ASCII.
    assert (Path(__file__).parents[1]/"laya_cuda"/"kernels.cu").read_bytes().isascii()


def test_gpu_architecture_check():
    pytest.importorskip("cupy")
    from laya_cuda.ops import check_arch
    check_arch("NVIDIA A100", 8, 0)
    check_arch("NVIDIA GeForce RTX 4090", 8, 9)
    with pytest.raises(RuntimeError, match=r"Tesla T4 is sm_75; .*compute capability 8\.0 or newer"):
        check_arch("Tesla T4", 7, 5)


def test_questions():
    q = question({"type":"choice","instructions":{"ask":"team"},"criteria":{"no":False,"zero":0,"empty":None}})
    assert q[3] == ["no: false","zero: 0","empty"]
    assert question({"type":"noul","instructions":"yes?"})[3] == ["false: no, the statement does not hold","true: yes, the statement holds"]
    with pytest.raises(ValueError):
        question({"type":"choice","instructions":"x","criteria":[]})


def test_softmax_stable():
    np.testing.assert_allclose(softmax([10000,10000]),[.5,.5])


def test_decoding_single_option_calibration_and_nonfinite():
    adapter=object.__new__(Laya)
    adapter.cfg={"temperature":[1,1,1],"temperature_by_options":{"choice:2":.1}}
    single={"q":{"type":"choice","instructions":"choose","criteria":["only"]}}
    items=[([1,2],[1],0)]
    answer=adapter.decode(np.array([[7.]]),np.array([[0.,0.]]),single,items)["answers"]["q"]
    assert answer["probabilities"]=={"only":1.0} and answer["confidence"]==1.0
    assert answer["action"]=={"act_probability":.5}
    double={"q":{"type":"choice","instructions":"choose","criteria":["a","b"]}}
    output=adapter.decode(np.array([[0.,1.]]),np.array([[0.,0.]]),double,items)
    assert output["answers"]["q"]["probabilities"]["b"]==round(float(softmax([0,2])[1]),4)
    with pytest.raises(RuntimeError,match="Non-finite"):
        adapter.decode(np.array([[np.nan]]),np.array([[0.,0.]]),single,items)


def test_missing_checkpoint_and_invalid_cache(tmp_path):
    from laya_cuda import Engine
    from laya_cuda.models import resolve
    with pytest.raises(FileNotFoundError,match="Incomplete checkpoint"):
        resolve(tmp_path)
    with pytest.raises(FileNotFoundError,match="Unknown alias"):
        resolve(tmp_path/"missing")
    for limit in (0,65,True):
        with pytest.raises(ValueError,match="max_cached_shapes"):
            Engine(tmp_path,max_cached_shapes=limit)


def test_unsupported_encoder_config(monkeypatch,tmp_path):
    import laya_cuda.laya as implementation
    monkeypatch.setattr(implementation,"read_json",lambda _: {"model_type":"bert"})
    with pytest.raises(ValueError,match="ModernBERT"):
        Laya(tmp_path)


def test_state_requires_json_values():
    adapter=object.__new__(Laya)
    adapter.encode=lambda value: []
    with pytest.raises(TypeError):
        adapter.prepare({"unsupported":object()},{"q":{"type":"noul","instructions":"yes?"}})


def test_temperature_division_uses_fp32():
    adapter=object.__new__(Laya)
    adapter.cfg={"temperature":[.7,1,1]}
    queries={"q":{"type":"choice","instructions":"choose","criteria":["a","b"]}}
    logits=np.array([[32,32.03125]],np.float16)
    result=adapter.decode(logits,np.array([[0,0]],np.float16),queries,[([1,2],[1,2],0)])
    expected=softmax(logits[0].astype(np.float32)/np.float32(.7))
    assert list(result["answers"]["q"]["probabilities"].values())==[round(float(p),4) for p in expected]


@pytest.mark.parametrize("model",["laya","laya-multilingual","laya-typed-decisions"])
def test_token_contract(model):
    path = Path("models")/model
    if not path.exists():
        pytest.skip("local checkpoint not available")
    a = Laya(path)
    queries = {"q":{"type":"choice","instructions":"Choose [MASK] a team","criteria":{"billing":"payment","other":None}}}
    seq,markers,kind = a.prepare("refund "*5000,queries)[0]
    assert len(seq)==a.max_len and seq[0]==a.cls and seq[-1]==a.sep
    assert len(markers)==2 and all(seq[i]==a.mask for i in markers) and kind==0
    with pytest.raises(ValueError):
        a.prepare("",{})
