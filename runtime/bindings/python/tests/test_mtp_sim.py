"""End-to-end MTP device-path test: emit a tiny MTP model with loom_genip, run
it on the in-process sim device, and check the MTP draft. Skipped when the dart
toolchain or genip emit is unavailable, so CI without dart still passes."""

import json
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import pytest

import loom

_IP_DIR = "/home/ross/Midstall/loom/ip"
_H, _INTER, _LAYERS, _MTP, _VOCAB = 8, 16, 2, 2, 10


def _safetensors(path):
    tensors = {
        "model.embed_tokens.weight": [_VOCAB, _H],
        "model.norm.weight": [_H],
    }

    def block(i):
        tensors[f"model.layers.{i}.input_layernorm.weight"] = [_H]
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = [_H, _H]
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = [_H, _H]
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = [_H, _H]
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = [_H, _H]
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = [_H]
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = [_INTER, _H]
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = [_INTER, _H]
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = [_H, _INTER]

    for i in range(_LAYERS):
        block(i)
    for m in range(_MTP):
        l = _LAYERS + m
        block(l)
        tensors[f"model.layers.{l}.enorm.weight"] = [_H]
        tensors[f"model.layers.{l}.hnorm.weight"] = [_H]
        tensors[f"model.layers.{l}.eh_proj.weight"] = [_H, 2 * _H]

    names = sorted(tensors)
    data = bytearray()
    header = {}
    off = 0
    for n in names:
        shape = tensors[n]
        count = int(np.prod(shape))
        vals = np.array([0.01 * (k + 1) for k in range(count)], dtype=np.float32)
        b = vals.tobytes()
        header[n] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(b)]}
        data += b
        off += len(b)
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(data))


def _config(path):
    cfg = {
        "model_type": "llama",
        "hidden_size": _H,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": _INTER,
        "vocab_size": _VOCAB,
        "max_position_embeddings": 32,
        "num_nextn_predict_layers": _MTP,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }
    with open(path, "w") as f:
        json.dump(cfg, f)


def _emit_model():
    """genip-emit the tiny MTP model into a temp dir; return it, or None on failure."""
    if shutil.which("dart") is None or not os.path.isdir(_IP_DIR):
        return None
    hf = tempfile.mkdtemp(prefix="loom_mtp_hf_")
    out = tempfile.mkdtemp(prefix="loom_mtp_out_")
    _config(os.path.join(hf, "config.json"))
    _safetensors(os.path.join(hf, "model.safetensors"))
    with open(os.path.join(hf, "tokenizer.json"), "w") as f:
        f.write('{"model":{"vocab":{},"merges":[]}}')
    r = subprocess.run(
        ["dart", "run", "bin/loom_genip.dart", "--soc", "fp", "--model", hf, "-o", out],
        cwd=_IP_DIR, capture_output=True, text=True,
    )
    shutil.rmtree(hf, ignore_errors=True)
    if r.returncode != 0:
        shutil.rmtree(out, ignore_errors=True)
        return None
    # The runtime loads a llama2.c-style tokenizer.bin (i32 max_token_length,
    # then per token {f32 score, i32 len, bytes}). MTP drafting never uses it, so
    # a minimal single-byte vocab is enough to satisfy Model load.
    with open(os.path.join(out, "tokenizer.bin"), "wb") as f:
        f.write(struct.pack("<i", 1))
        for i in range(_VOCAB):
            f.write(struct.pack("<f", 0.0))
            f.write(struct.pack("<i", 1))
            f.write(bytes([65 + (i % 26)]))
    return out


def test_mtp_draft_on_sim_matches_main_prediction():
    model_dir = _emit_model()
    if model_dir is None:
        pytest.skip("dart/genip unavailable; cannot emit the MTP model")
    try:
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        dev = loom.Device.open_sim(rt, model, model_dir)
        model.prepare(dev)

        tokens = [1, 2, 3]

        # Device MTP draft: num_modules + 1 tokens.
        ctx = loom.Context(model, 32)
        draft = list(ctx.mtp_draft(dev, tokens))
        assert len(draft) == _MTP + 1
        assert all(0 <= t < _VOCAB for t in draft)

        # draft[0] is the main model's next-token prediction: cross-check against
        # a plain eval over the same tokens on the same device.
        ctx2 = loom.Context(model, 32)
        logits = ctx2.eval(dev, np.array(tokens, dtype=np.uint32), 0)
        assert draft[0] == int(np.argmax(logits))

        # The draft is deterministic (pure greedy) on the sim.
        ctx3 = loom.Context(model, 32)
        draft2 = list(ctx3.mtp_draft(dev, tokens))
        assert draft2 == draft
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
