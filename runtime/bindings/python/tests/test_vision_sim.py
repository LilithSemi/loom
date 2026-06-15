"""End-to-end vision device-path test: emit a tiny VLM with loom_genip, run its
ViT tower + projector on the in-process sim device via Model.encode_image, and
check the projected image embeddings. Skipped when dart/genip is unavailable."""

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
_H, _INTER, _LAYERS, _VOCAB = 8, 16, 2, 10
_VIMG, _VPATCH, _VC, _VLAYERS, _VINTER = 4, 2, 3, 1, 16
_SEQ = (_VIMG // _VPATCH) * (_VIMG // _VPATCH) + 1  # + class token = 5


def _write_safetensors(path):
    t = {
        "model.embed_tokens.weight": [_VOCAB, _H],
        "model.norm.weight": [_H],
    }

    def block(i):
        t[f"model.layers.{i}.input_layernorm.weight"] = [_H]
        for w in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[f"model.layers.{i}.self_attn.{w}.weight"] = [_H, _H]
        t[f"model.layers.{i}.post_attention_layernorm.weight"] = [_H]
        t[f"model.layers.{i}.mlp.gate_proj.weight"] = [_INTER, _H]
        t[f"model.layers.{i}.mlp.up_proj.weight"] = [_INTER, _H]
        t[f"model.layers.{i}.mlp.down_proj.weight"] = [_H, _INTER]

    for i in range(_LAYERS):
        block(i)

    v = "vision_model"
    t[f"{v}.embeddings.patch_embedding.weight"] = [_H, _VC, _VPATCH, _VPATCH]
    t[f"{v}.embeddings.patch_embedding.bias"] = [_H]
    t[f"{v}.embeddings.class_embedding"] = [_H]
    t[f"{v}.embeddings.position_embedding.weight"] = [_SEQ, _H]
    for nm in ("pre_layrnorm", "post_layernorm"):
        t[f"{v}.{nm}.weight"] = [_H]
        t[f"{v}.{nm}.bias"] = [_H]
    for i in range(_VLAYERS):
        l = f"{v}.encoder.layers.{i}"
        t[f"{l}.layer_norm1.weight"] = [_H]
        t[f"{l}.layer_norm1.bias"] = [_H]
        for w in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[f"{l}.self_attn.{w}.weight"] = [_H, _H]
            t[f"{l}.self_attn.{w}.bias"] = [_H]
        t[f"{l}.layer_norm2.weight"] = [_H]
        t[f"{l}.layer_norm2.bias"] = [_H]
        t[f"{l}.mlp.fc1.weight"] = [_VINTER, _H]
        t[f"{l}.mlp.fc1.bias"] = [_VINTER]
        t[f"{l}.mlp.fc2.weight"] = [_H, _VINTER]
        t[f"{l}.mlp.fc2.bias"] = [_H]
    t["multi_modal_projector.linear_1.weight"] = [_H, _H]
    t["multi_modal_projector.linear_1.bias"] = [_H]
    t["multi_modal_projector.linear_2.weight"] = [_H, _H]
    t["multi_modal_projector.linear_2.bias"] = [_H]

    data = bytearray()
    header = {}
    off = 0
    for n in sorted(t):
        shape = t[n]
        count = int(np.prod(shape))
        vals = np.array([0.02 * (k % 7 - 3) for k in range(count)], dtype=np.float32)
        b = vals.tobytes()
        header[n] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(b)]}
        data += b
        off += len(b)
    hb = json.dumps(header).encode()
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
        "tie_word_embeddings": True,
        "image_token_index": 7,
        "mm_projector_type": "mlp2x_gelu",
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": _H,
            "image_size": _VIMG,
            "patch_size": _VPATCH,
            "num_channels": _VC,
            "num_hidden_layers": _VLAYERS,
            "num_attention_heads": 2,
            "intermediate_size": _VINTER,
            "hidden_act": "gelu",
        },
    }
    with open(path, "w") as f:
        json.dump(cfg, f)


def _emit_vlm():
    if shutil.which("dart") is None or not os.path.isdir(_IP_DIR):
        return None
    hf = tempfile.mkdtemp(prefix="loom_vlm_hf_")
    out = tempfile.mkdtemp(prefix="loom_vlm_out_")
    _config(os.path.join(hf, "config.json"))
    _write_safetensors(os.path.join(hf, "model.safetensors"))
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
    # Minimal tokenizer.bin so Model load succeeds (vision path never uses it).
    with open(os.path.join(out, "tokenizer.bin"), "wb") as f:
        f.write(struct.pack("<i", 1))
        for i in range(_VOCAB):
            f.write(struct.pack("<f", 0.0) + struct.pack("<i", 1) + bytes([65 + i % 26]))
    return out


def _image(scale=0.1):
    n = _VC * _VIMG * _VIMG
    return np.array([(i % 9 - 4) * scale for i in range(n)], dtype=np.float32)


def test_encode_image_on_sim_shape_finite_deterministic():
    model_dir = _emit_vlm()
    if model_dir is None:
        pytest.skip("dart/genip unavailable; cannot emit the VLM")
    try:
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        dev = loom.Device.open_sim(rt, model, model_dir)
        model.prepare(dev)

        emb = model.encode_image(dev, _image())
        # seq_len (5) rows of text hidden (8) -> 40 values.
        assert emb.shape[0] == _SEQ * _H
        assert np.all(np.isfinite(emb))

        # Deterministic on the sim.
        emb2 = model.encode_image(dev, _image())
        np.testing.assert_array_equal(emb, emb2)

        # Sensitive to the input image.
        emb3 = model.encode_image(dev, _image(scale=0.5))
        assert np.abs(emb - emb3).max() > 1e-6
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def test_vlm_eval_fuses_image_embeds_on_sim():
    """Full VLM device path: encode an image, then eval a token sequence with the
    image placeholder (id 7) splicing the projected vision embeddings."""
    model_dir = _emit_vlm()
    if model_dir is None:
        pytest.skip("dart/genip unavailable; cannot emit the VLM")
    try:
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        dev = loom.Device.open_sim(rt, model, model_dir)
        model.prepare(dev)

        emb = model.encode_image(dev, _image())  # seq_len*hidden
        # Sequence: text token, then one image placeholder consuming one vision
        # row, then a text token. Provide one vision embedding (row 0).
        one_vision = emb[:_H].astype(np.float32)
        ctx = loom.Context(model, 32)
        logits = ctx.eval_vlm(dev, np.array([1, 7, 2], dtype=np.uint32), 0, one_vision, 1)
        assert logits.shape[0] == _VOCAB
        assert np.all(np.isfinite(logits))

        # Placeholder-count mismatch is an error.
        ctx2 = loom.Context(model, 32)
        with pytest.raises(loom.LoomError):
            ctx2.eval_vlm(dev, np.array([1, 7, 7], dtype=np.uint32), 0, one_vision, 1)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def test_load_image_decodes_a_real_jpeg_and_encodes_on_sim():
    """The full front door: a real JPEG (encoded by PIL) -> Loom's own decoder +
    preprocess + ViT + projector on the sim, in one loom_load_image call."""
    import io as _io
    from PIL import Image

    model_dir = _emit_vlm()
    if model_dir is None:
        pytest.skip("dart/genip unavailable; cannot emit the VLM")
    try:
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        dev = loom.Device.open_sim(rt, model, model_dir)
        model.prepare(dev)

        # A small RGB image, JPEG-encoded (4:4:4). image_size is 4, but the
        # decoder + preprocess resize whatever comes in.
        rng = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        buf = _io.BytesIO()
        Image.fromarray(rng, "RGB").save(buf, format="JPEG", quality=92, subsampling=0)
        jpeg = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        emb = model.load_image(dev, jpeg)
        assert emb.shape[0] == _SEQ * _H
        assert np.all(np.isfinite(emb))

        # Deterministic.
        emb2 = model.load_image(dev, jpeg)
        np.testing.assert_array_equal(emb, emb2)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
