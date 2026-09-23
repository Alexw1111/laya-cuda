"""Compare a serialized Engine with BatchEngine under bounded outstanding load."""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from laya_cuda import Engine, BatchEngine


def run(model,mode,work,concurrency,samples,warmup,batch_tokens):
    executor=ThreadPoolExecutor(1) if mode=='serial' else None
    started=time.perf_counter()
    engine=executor.submit(Engine,model).result() if executor else BatchEngine(model,max_batch_tokens=batch_tokens)
    load_ms=(time.perf_counter()-started)*1000
    def submit(state,questions):
        return executor.submit(engine.predict,state,questions) if executor else engine.submit(state,questions)
    try:
        # Warm all input cases at the same outstanding concurrency.
        for i in range(0,warmup,concurrency):
            futures=[submit(*work[j%len(work)]) for j in range(i,min(i+concurrency,warmup))]
            for f in futures:f.result()
        pending={};records=[];sent=0;started=time.perf_counter()
        while sent<samples or pending:
            while sent<samples and len(pending)<concurrency:
                state,questions=work[sent%len(work)]
                before=time.perf_counter()
                original=submit(state,questions)
                f=Future()
                # Capture completion rather than the later time the load driver polls it.
                def complete(done,finished=f):
                    try:
                        done.result()
                        finished.set_result((time.perf_counter(),getattr(done,'metrics',{})))
                    except BaseException as error:
                        finished.set_exception(error)
                original.add_done_callback(complete)
                pending[f]=(before,len(questions));sent+=1
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                before,n=pending.pop(f)
                completed,metrics=f.result()
                records.append({'ms':(completed-before)*1000,'questions':n,'metrics':metrics})
        seconds=time.perf_counter()-started
        values=[r['ms'] for r in records]
        return {'load_ms':load_ms,'p50':float(np.percentile(values,50)),
                'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)),
                'requests_s':samples/seconds,'questions_s':sum(r['questions'] for r in records)/seconds,
                'seconds':seconds,'records':records}
    finally:
        if executor:
            executor.submit(engine.close).result();executor.shutdown()
        else:engine.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/laya')
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--questions',type=int,default=1)
    parser.add_argument('--batch-tokens',type=int,default=4096)
    parser.add_argument('--concurrency',type=int,nargs='+',default=[1,8,32,64])
    parser.add_argument('--lengths',nargs='+',choices=['short','long','mixed'],default=['short','long','mixed'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from laya_cuda.models import adapter,resolve
    a=adapter(resolve(args.model,None))
    questions={str(i):{'type':'choice','instructions':'Choose a topic.',
                      'criteria':['payment','software']} for i in range(args.questions)}
    overhead=len(a.prepare('',questions)[0][0])
    short=(' refund'*max(0,32-overhead),questions)
    long=(' refund'*max(0,a.max_len-overhead),questions)
    result={'platform':platform.platform(),'protocol':vars(args),'cases':{},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('laya_cuda').glob('*.py')},
            'tokens':{'short':[len(x[0]) for x in a.prepare(*short)],'long':[len(x[0]) for x in a.prepare(*long)]}}
    for name,work in [('short',[short]),('long',[long]),('mixed',[short,long])]:
        if name not in args.lengths:continue
        for concurrency in args.concurrency:
            rounds=[]
            for r in range(args.rounds):
                pair={}
                for mode in (('serial','batch') if r%2==0 else ('batch','serial')):
                    pair[mode]=run(args.model,mode,work,concurrency,args.samples,args.warmup,args.batch_tokens)
                rounds.append(pair)
                print(name,concurrency,r,{k:round(v['requests_s'],1) for k,v in pair.items()},flush=True)
            result['cases'][f'{name}-c{concurrency}']=rounds
            Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
