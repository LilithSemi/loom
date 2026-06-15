"""End-to-end BitNet ternary check on the REAL trained model: genip compiled the
ternary TinyStories model (int4-packed ternary, per-tensor scale), and the Zig
sim must reproduce the trainer's first greedy token, the same one the Dart fp64
golden reproduced. This seals the compile-to-run path for ternary. Skipped when
the genip output / fixture are absent (they live in gitignored .cache)."""

import json
import os

import numpy as np
import pytest

import loom

_GENIP = "/home/ross/Midstall/loom/.cache/ternary-stories-genip"
_FIX = "/home/ross/Midstall/loom/.cache/ternary-stories/fixture.json"


def test_ternary_model_sim_matches_trainer_first_token():
    if not os.path.isfile(os.path.join(_GENIP, "loom.json")) or not os.path.isfile(_FIX):
        pytest.skip("genip ternary output / fixture absent")
    cfg = json.load(open(os.path.join(_GENIP, "loom.json")))
    assert cfg.get("quant") == "bitnet_ternary"
    vocab = cfg["vocab"]

    fix = json.load(open(_FIX))
    prompt = np.array(fix["prompt_ids"], dtype=np.uint32)
    expected_first = fix["greedy_ids"][0]

    rt = loom.Runtime()
    model = loom.Model(rt, _GENIP)
    dev = loom.Device.open_sim(rt, model, _GENIP)
    model.prepare(dev)
    ctx = loom.Context(model, 64)
    logits = ctx.eval(dev, prompt, 0)

    assert logits.shape[0] == vocab
    assert np.all(np.isfinite(logits))
    # The int4-packed ternary model in the sim reproduces the trainer/golden's
    # first greedy token: the compile-to-run datapath is faithful.
    assert int(np.argmax(logits)) == expected_first, (
        f"sim argmax {int(np.argmax(logits))} != trainer {expected_first}"
    )
