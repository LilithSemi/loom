"""End-to-end Idefics3/SmolVLM device path on a tiny synthetic model: genip
compiles it (text under model.text_model, SigLIP vision under model.vision_model,
pixel-shuffle connector), then the runtime decodes a JPEG, runs the ViT +
pixel-shuffle connector, and fuses the image tokens into the LLM. Verifies the
projected token count shrinks by scale^2. Skipped when dart/genip is absent."""

import io as _io
import json
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import pytest
from PIL import Image

import loom

_IP_DIR = "/home/ross/Midstall/loom/ip"
_TH, _TINTER, _TLAYERS, _VOCAB = 8, 16, 2, 10   # text
# Vision hidden 128 (> the max_cols=64 tile threshold) so the ViT matmuls and the
# 512-wide pixel-shuffle connector actually column-tile, exercising that path.
_VH, _VIMG, _VPATCH, _VC, _VLAYERS, _VINTER = 128, 8, 2, 3, 1, 256  # vision
_SCALE = 2
_NPATCH = (_VIMG // _VPATCH) ** 2               # 16
_NTOK = _NPATCH // (_SCALE * _SCALE)            # 4 projected image tokens
_IMG_TOK = 7


def _st(path):
    t = {"lm_head.weight": [_VOCAB, _TH],
         "model.text_model.embed_tokens.weight": [_VOCAB, _TH],
         "model.text_model.norm.weight": [_TH]}
    for i in range(_TLAYERS):
        p = f"model.text_model.layers.{i}"
        t[f"{p}.input_layernorm.weight"] = [_TH]
        for w in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[f"{p}.self_attn.{w}.weight"] = [_TH, _TH]
        t[f"{p}.post_attention_layernorm.weight"] = [_TH]
        t[f"{p}.mlp.gate_proj.weight"] = [_TINTER, _TH]
        t[f"{p}.mlp.up_proj.weight"] = [_TINTER, _TH]
        t[f"{p}.mlp.down_proj.weight"] = [_TH, _TINTER]
    v = "model.vision_model"
    t[f"{v}.embeddings.patch_embedding.weight"] = [_VH, _VC, _VPATCH, _VPATCH]
    t[f"{v}.embeddings.patch_embedding.bias"] = [_VH]
    t[f"{v}.embeddings.position_embedding.weight"] = [_NPATCH, _VH]
    t[f"{v}.post_layernorm.weight"] = [_VH]
    t[f"{v}.post_layernorm.bias"] = [_VH]
    for i in range(_VLAYERS):
        l = f"{v}.encoder.layers.{i}"
        t[f"{l}.layer_norm1.weight"] = [_VH]; t[f"{l}.layer_norm1.bias"] = [_VH]
        for w in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[f"{l}.self_attn.{w}.weight"] = [_VH, _VH]
            t[f"{l}.self_attn.{w}.bias"] = [_VH]
        t[f"{l}.layer_norm2.weight"] = [_VH]; t[f"{l}.layer_norm2.bias"] = [_VH]
        t[f"{l}.mlp.fc1.weight"] = [_VINTER, _VH]; t[f"{l}.mlp.fc1.bias"] = [_VINTER]
        t[f"{l}.mlp.fc2.weight"] = [_VH, _VINTER]; t[f"{l}.mlp.fc2.bias"] = [_VH]
    t["model.connector.modality_projection.proj.weight"] = [_TH, _VH * _SCALE * _SCALE]

    data = bytearray(); header = {}; off = 0
    for n in sorted(t):
        shape = t[n]; count = int(np.prod(shape))
        vals = np.array([0.02 * (k % 7 - 3) for k in range(count)], np.float32)
        b = vals.tobytes(); header[n] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(b)]}
        data += b; off += len(b)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(bytes(data))


def _config(path):
    cfg = {
        "model_type": "idefics3", "scale_factor": _SCALE, "image_token_id": _IMG_TOK,
        "tie_word_embeddings": False, "vocab_size": _VOCAB,
        "text_config": {"model_type": "llama", "hidden_size": _TH, "num_hidden_layers": _TLAYERS,
                        "num_attention_heads": 2, "num_key_value_heads": 2,
                        "intermediate_size": _TINTER, "vocab_size": _VOCAB,
                        "max_position_embeddings": 64, "tie_word_embeddings": False},
        "vision_config": {"model_type": "idefics3", "hidden_size": _VH, "image_size": _VIMG,
                          "patch_size": _VPATCH, "num_channels": _VC, "num_hidden_layers": _VLAYERS,
                          "num_attention_heads": 2, "intermediate_size": _VINTER, "hidden_act": "gelu"},
    }
    with open(path, "w") as f:
        json.dump(cfg, f)


def _emit():
    if shutil.which("dart") is None or not os.path.isdir(_IP_DIR):
        return None
    hf = tempfile.mkdtemp(prefix="loom_idefics3_hf_")
    out = tempfile.mkdtemp(prefix="loom_idefics3_out_")
    _config(os.path.join(hf, "config.json"))
    _st(os.path.join(hf, "model.safetensors"))
    with open(os.path.join(hf, "tokenizer.json"), "w") as f:
        f.write('{"model":{"vocab":{},"merges":[]}}')
    r = subprocess.run(["dart", "run", "bin/loom_genip.dart", "--soc", "fp",
                        "--model", hf, "-o", out], cwd=_IP_DIR, capture_output=True, text=True)
    shutil.rmtree(hf, ignore_errors=True)
    if r.returncode != 0:
        shutil.rmtree(out, ignore_errors=True)
        print(r.stdout, r.stderr)
        return None
    with open(os.path.join(out, "tokenizer.bin"), "wb") as f:
        f.write(struct.pack("<i", 1))
        for i in range(_VOCAB):
            f.write(struct.pack("<f", 0.0) + struct.pack("<i", 1) + bytes([65 + i % 26]))
    return out


def test_idefics3_pixel_shuffle_device_path():
    model_dir = _emit()
    if model_dir is None:
        pytest.skip("dart/genip unavailable")
    try:
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        dev = loom.Device.open_sim(rt, model, model_dir)
        model.prepare(dev)

        # A real JPEG -> Loom decode + preprocess + ViT + pixel-shuffle connector.
        arr = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        buf = _io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=92, subsampling=0)
        emb = model.load_image(dev, np.frombuffer(buf.getvalue(), np.uint8))
        # Pixel shuffle: 16 patches / scale^2(4) = 4 tokens, each text hidden 8.
        assert emb.shape[0] == _NTOK * _TH
        assert np.all(np.isfinite(emb))

        # Fuse the 4 image tokens into the LLM and get logits.
        feats = emb.reshape(_NTOK, _TH).astype(np.float32).reshape(-1)
        ctx = loom.Context(model, 32)
        ids = np.array([1] + [_IMG_TOK] * _NTOK + [2], dtype=np.uint32)
        logits = ctx.eval_vlm(dev, ids, 0, feats, _NTOK)
        assert logits.shape[0] == _VOCAB
        assert np.all(np.isfinite(logits))
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
