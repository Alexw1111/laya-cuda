"""Laya's checkpoint and request contract, adapted from its Apache-2.0 SDK.

See NOTICE. GPU execution is supplied by the shared runtime, not the SDK.
"""
import hashlib
import json
import math

import numpy as np
from tokenizers import Tokenizer

from .models import read_json

TYPES = ("choice", "score", "noul")


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def question(value):
    kind, instruction, criteria = value["type"], value["instructions"], value.get("criteria")
    if kind not in TYPES:
        raise ValueError(f"Unsupported question type: {kind}")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or not all(isinstance(k, str) for k in criteria):
            raise ValueError("Choice criteria must be a mapping or list of string labels")
        options = [k if v is None or v == "" else f"{k}: {text(v)}" for k, v in criteria.items()]
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("Score criteria must be a list")
        options = [f"level {i}: {text(v)}" for i, v in enumerate(criteria)]
    else:
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a mapping")
        options = [f"{k}: {text(criteria[k]) if criteria.get(k) not in (None, '') else default}"
                   for k, default in (("false", "no, the statement does not hold"),
                                      ("true", "yes, the statement holds"))]
    if not options:
        raise ValueError("Each question must have at least one option")
    instruction = instruction if isinstance(instruction, str) else json.dumps(instruction)
    return kind, instruction, criteria, options


def softmax(values):
    values = np.asarray(values, np.float32)
    exp = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class Laya:
    def __init__(self, path):
        self.path, self.cfg = path, read_json(path / "rl_agent_config.json")
        self.config = read_json(path / "encoder/config.json")
        c = self.config
        if c.get("model_type") != "modernbert" or c.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("Only ModernBERT/mmBERT with exact GELU is supported")
        self.width, self.heads = c["hidden_size"], c["num_attention_heads"]
        if self.width // self.heads != 64 or self.width % self.heads:
            raise ValueError("v1 requires 64-wide attention heads")
        if any(c.get(k, False) for k in ("attention_bias", "mlp_bias", "norm_bias", "rope_scaling")):
            raise ValueError("Biased or scaled-RoPE encoder variants are unsupported")
        self.layers, self.intermediate = c["num_hidden_layers"], c["intermediate_size"]
        self.ff_width = max(2*self.intermediate,4*self.width)
        self.global_every, self.window = c.get("global_attn_every_n_layers", 3), c.get("local_attention", 128) // 2
        expected = ["full_attention" if i % self.global_every == 0 else "sliding_attention" for i in range(self.layers)]
        if c.get("layer_types", expected) != expected:
            raise ValueError("Unsupported attention topology")
        self.theta = []
        for kind, default in (("full_attention", 160000), ("sliding_attention", 10000)):
            rope = c.get("rope_parameters", {}).get(kind, {})
            if rope.get("rope_type", "default") != "default":
                raise ValueError("Only default RoPE is supported")
            field = "global_rope_theta" if kind == "full_attention" else "local_rope_theta"
            self.theta.append(float(rope.get("rope_theta", c.get(field, default)) or self.theta[0]))
        self.eps = float(c.get("norm_eps", 1e-5))
        self.max_len, self.head_max_len = self.cfg["max_len"], self.cfg["head_max_len"]
        self.head_layers, self.n_act = self.cfg["head_layers"], len(self.cfg.get("act_costs", {})) + 1
        if not 0 < self.max_len <= min(1024, c["max_position_embeddings"]) or self.head_layers < 1:
            raise ValueError("Invalid context budget or decision head")
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer/tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        specials = {}
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            if (path / "tokenizer" / name).is_file():
                specials.update(read_json(path / "tokenizer" / name))
        for key in ("cls", "sep", "mask", "pad"):
            token = specials.get(key + "_token")
            token = token.get("content") if isinstance(token, dict) else token
            value = self.tokenizer.token_to_id(token) if isinstance(token, str) else None
            if value is None:
                raise ValueError(f"Missing tokenizer special token: {key}")
            setattr(self, key, value)
            if key == "mask":
                self.mask_text = token

    def encode(self, value):
        return self.tokenizer.encode(value.replace(self.mask_text, " "), add_special_tokens=False).ids

    def prepare(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Provide a nonempty mapping of questions")
        parsed = [question(definition) for definition in questions.values()]
        texts = [state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)]
        for kind, instruction, _, options in parsed:
            texts += [f"{kind} question: {instruction}"] + [" " + opt for opt in options]
        texts = [v.replace(self.mask_text, " ") for v in texts]
        # Batched encoding pays a thread-pool dispatch; it wins from about a dozen texts on.
        # Either way each text encodes independently, with identical IDs.
        encoded = (self.tokenizer.encode_batch(texts, add_special_tokens=False) if len(texts) >= 12
                   else [self.tokenizer.encode(v, add_special_tokens=False) for v in texts])
        encoded = iter(e.ids for e in encoded)
        state_ids, items = next(encoded), []
        for qid, (kind, instruction, criteria, options) in zip(questions, parsed):
            head = next(encoded)
            opts = [[self.mask] + next(encoded)[:48] for _ in options]
            budget = self.head_max_len - sum(map(len, opts))
            if budget < 16:
                cap = max(4, (self.head_max_len - 16) // len(opts))
                opts = [v[:cap] for v in opts]
                budget = self.head_max_len - sum(map(len, opts))
            ids, markers = [self.cls] + head[:max(8, budget)] + [self.sep], []
            for opt in opts:
                markers.append(len(ids))
                ids.extend(opt)
            ids.append(self.sep)
            ids = (ids + state_ids[:max(0, self.max_len - len(ids) - 1)] + [self.sep])[:self.max_len]
            if markers[-1] >= len(ids):
                raise ValueError(f"Question {qid!r} exceeds the option token budget")
            items.append((ids, markers, TYPES.index(kind)))
        return items

    def decode(self, logits, acts, questions, items):
        if not np.isfinite(logits).all() or not np.isfinite(acts).all():
            raise RuntimeError("Non-finite model output; no decision returned")
        actions, answers = softmax(acts), {}
        for row, (qid, definition) in enumerate(questions.items()):
            kind, _, criteria, options = question(definition)
            count, qt = len(options), TYPES.index(kind)
            bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
            temp = self.cfg.get("temperature_by_options", {}).get(f"{kind}:{bucket}", self.cfg.get("temperature", [1]*3)[qt])
            try:
                temp = float(temp)
            except (TypeError, ValueError):
                temp = 1.0
            temp = float(np.clip(temp, .5, 5)) if math.isfinite(temp) else 1.0
            prob = softmax(logits[row, :count].astype(np.float32) / np.float32(temp))
            conf = float(np.clip(1 + (prob * np.log(np.maximum(prob, 1e-12))).sum() / math.log(count), 0, 1)) if count>1 else 1.0
            answer = {"type": kind, "confidence": round(conf, 4), "action": {"act_probability": round(float(actions[row, 0]), 4)}}
            if kind == "noul":
                answer.update(noul=round(float(prob[1]), 4), confidence=round(float(max(prob[1], 1-prob[1])), 4))
            else:
                keys = list(criteria) if kind == "choice" else list(map(str, range(count)))
                answer["probabilities"] = {k: round(float(p), 4) for k, p in zip(keys, prob)}
                if kind == "choice":
                    answer["choice"] = keys[int(prob.argmax())]
                else:
                    answer.update(score=round(float(np.arange(count) @ prob), 4), legend=dict(zip(keys, criteria)))
            answers[qid] = answer
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": sum(len(i[0]) for i in items), "output_tokens": 0}}

    def weight_shapes(self):
        w, f = self.width, self.intermediate
        shapes = {"encoder.embeddings.tok_embeddings.weight": (self.config["vocab_size"], w),
                  "encoder.embeddings.norm.weight": (w,), "encoder.final_norm.weight": (w,),
                  "type_emb.weight": (3, w), "temperature": (3,),
                  "scorer.0.weight": (w,), "scorer.0.bias": (w,), "scorer.1.weight": (w, w),
                  "scorer.1.bias": (w,), "scorer.3.weight": (1, w), "scorer.3.bias": (1,),
                  "act_head.0.weight": (256, w+4), "act_head.0.bias": (256,),
                  "act_head.2.weight": (self.n_act, 256), "act_head.2.bias": (self.n_act,)}
        for i in range(self.layers):
            p = f"encoder.layers.{i}."
            shapes.update({p+"attn.Wqkv.weight": (3*w, w), p+"attn.Wo.weight": (w, w),
                           p+"mlp.Wi.weight": (2*f, w), p+"mlp.Wo.weight": (w, f), p+"mlp_norm.weight": (w,)})
            if i:
                shapes[p+"attn_norm.weight"] = (w,)
        for i in range(self.head_layers):
            p = f"head.layers.{i}."
            shapes.update({p+k: shape for k, shape in {
                "self_attn.in_proj_weight": (3*w, w), "self_attn.in_proj_bias": (3*w,),
                "self_attn.out_proj.weight": (w, w), "self_attn.out_proj.bias": (w,),
                "linear1.weight": (4*w, w), "linear1.bias": (4*w,), "linear2.weight": (w, 4*w),
                "linear2.bias": (w,), "norm1.weight": (w,), "norm1.bias": (w,),
                "norm2.weight": (w,), "norm2.bias": (w,)}.items()})
        return shapes

    def fingerprint(self, items):
        return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()

    def allocate_head(self, s):
        import cupy as cp
        m,w,selected = s.m,self.width,s.b*(s.k+1)
        s.linear_scratch = cp.empty((m,4*w),cp.float32)
        s.selected = cp.empty((selected,w),cp.float32)
        s.selected16,s.scored = [cp.empty((selected,w),cp.float16) for _ in range(2)]
        # Decision logits stay FP32: they feed the returned probabilities directly.
        s.logits = cp.empty((s.b,s.k+1),cp.float32)
        s.features = cp.empty((s.b,w+4),cp.float16)
        s.act_hidden,s.acts = cp.empty((s.b,256),cp.float16),cp.empty((s.b,self.n_act),cp.float32)

    def read_outputs(self, s):
        import cupy as cp
        return cp.asnumpy(s.logits)[:,1:],cp.asnumpy(s.acts)

    def official(self, device):
        from .official import Official
        return Official(self.path,device)

    def forward_head(self, s):
        """Only the last head layer's readout rows need its pointwise tail."""
        op,w = s.op,s.runtime.weights
        m,d,count = s.m,self.width,s.b*(s.k+1)
        op.elements("type_add",m*d,s.residual,w["type_emb.weight"],s.inputs["kinds"],m,d)
        for i in range(self.head_layers):
            p = f"head.layers.{i}."
            s.norm(s.x,p+"norm1",eps=1e-5)
            s.linear(s.x,s.qkv,p+"self_attn.in_proj",packed=True)
            s.attend()
            residual,x,branch,attention,rows = s.residual,s.x,s.branch,s.attention,m
            if i==self.head_layers-1:
                op.elements("gather",count*d,residual,s.inputs["indices"],s.selected,count,d)
                op.elements("gather_half",count*d,attention,s.inputs["indices"],s.scored,count,d)
                residual,x,branch,attention,rows = s.selected,s.selected16,s.branch[:count],s.scored,count
            s.linear(attention,branch,p+"self_attn.out_proj")
            s.norm(x,p+"norm2",branch,source=residual,eps=1e-5)
            ff = s.ff.ravel()[:rows*4*d].reshape(rows,4*d)
            s.linear(x,ff,p+"linear1")
            op.elements("activate",ff.size,ff,ff,ff.size,4*d,1)
            s.linear(ff,branch,p+"linear2")
            op.elements("add",rows*d,residual,branch,rows*d)
        if not self.head_layers:
            op.elements("gather",count*d,s.residual,s.inputs["indices"],s.selected,count,d)
        s.norm(s.selected16,"scorer.0",source=s.selected,eps=1e-5)
        s.linear(s.selected16,s.scored,"scorer.1")
        op.elements("activate",s.scored.size,s.scored,s.scored,s.scored.size,d,0)
        s.linear(s.scored,s.logits,"scorer.3")
        op.call("features",s.b,s.selected,s.logits,s.inputs["counts"],s.features,s.b,s.k,d)
        s.linear(s.features,s.act_hidden,"act_head.0")
        op.elements("activate",s.act_hidden.size,s.act_hidden,s.act_hidden,s.act_hidden.size,256,0)
        s.linear(s.act_hidden,s.acts,"act_head.2")
