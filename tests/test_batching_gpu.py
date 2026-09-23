from pathlib import Path

import numpy as np
import pytest

from laya_cuda import Engine, BatchEngine
from benchmarks.run import probabilities,label


@pytest.mark.gpu
@pytest.mark.parametrize('model',['laya','laya-multilingual','laya-typed-decisions'])
def test_batched_model_contract(model):
    path=Path('models')/model
    if not path.exists():pytest.skip('local checkpoint required')
    questions={'same':{'type':'choice','instructions':'Choose a topic.','criteria':['payment','software']},
               'score':{'type':'score','instructions':'Urgency?','criteria':['low','medium','high']},
               'noul':{'type':'noul','instructions':'Does it require a refund?'}}
    states=['Please refund my payment.','软件出现错误，请处理。',' refund'*110,' software'*500]
    requests=[(state,questions) for state in states]
    requests.append(('Please refund my payment.',{'same':{'type':'choice','instructions':'Choose a topic.',
                    'criteria':['payment','software','delivery','account','other']}}))
    requests.append(('Please refund my payment.',{'same':{'type':'choice','instructions':'Choose a topic.',
                    'criteria':['only']}}))
    with Engine(path) as engine:
        expected=[engine.predict(state,qs) for state,qs in requests]
    with BatchEngine(path,max_batch_tokens=8192,max_wait_ms=5) as engine:
        futures=[engine.submit(state,qs) for state,qs in requests*2]
        differences=[];agreement=[]
        for i,future in enumerate(futures):
            actual=future.result(120);reference=expected[i%len(requests)]
            assert actual['usage']==reference['usage']
            assert list(actual['answers'])==list(reference['answers'])
            for key,a in actual['answers'].items():
                b=reference['answers'][key]
                differences.extend(abs(np.array(probabilities(a))-probabilities(b)))
                agreement.append(label(a)==label(b))
                assert abs(a['action']['act_probability']-b['action']['act_probability'])<=.01
        assert np.isfinite(differences).all()
        assert np.mean(differences)<=.001 and max(differences)<=.01
        assert np.mean(agreement)>=.995
