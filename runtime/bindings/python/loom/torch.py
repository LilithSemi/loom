"""HuggingFace-compatible wrapper: a real transformers PreTrainedModel driving
Loom via the pyloom bindings, so HF's own .generate() (greedy + sampling) runs on
the Loom accelerator or its sim. Importing this module requires torch +
transformers (nix develop .#rt provides them). Genip-baked models, single-sequence.
"""

try:
    import numpy as np
    import torch
    from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "loom.torch needs torch + transformers (nix develop .#rt provides them)"
    ) from e

import loom


class LoomConfig(PretrainedConfig):
    model_type = "loom"

    def __init__(self, **kw):
        # Defaults keep PretrainedConfig happy when reloaded without our fields.
        self.hidden_size = kw.pop("hidden_size", 0)
        self.num_hidden_layers = kw.pop("num_hidden_layers", 0)
        self.num_attention_heads = kw.pop("num_attention_heads", 0)
        self.num_key_value_heads = kw.pop("num_key_value_heads", 0)
        self.head_dim = kw.pop("head_dim", 0)
        self.intermediate_size = kw.pop("intermediate_size", 0)
        super().__init__(**kw)

    @classmethod
    def from_info(cls, info):
        return cls(
            vocab_size=info["vocab"],
            hidden_size=info["hidden"],
            num_hidden_layers=info["layers"],
            num_attention_heads=info["num_heads"],
            num_key_value_heads=info["num_kv_heads"],
            head_dim=info["head_dim"],
            intermediate_size=info["intermediate"],
            # The model's trained context; HF uses it to bound positions/RoPE.
            # Fall back to 2048 if the manifest predates max_seq (reported as 0).
            max_position_embeddings=info.get("max_seq") or 2048,
            bos_token_id=1,
            eos_token_id=2,
        )


class LoomCache:
    """NOT a transformers Cache: the real KV lives on-device in loom_context. This
    only reports the sequence length, which is all generate needs from us since the
    model declares `_supports_default_dynamic_cache() == False` (the mamba/zamba
    "the model manages its own cache" path). HF uses this length to slice the next
    input_ids and to tell forward the current position."""

    def __init__(self):
        self._len = 0

    def get_seq_length(self, layer_idx=0):
        return self._len


class LoomTokenizer:
    """Minimal tokenizer over loom's tok512, enough for ergonomic generate() calls.
    Not a full transformers PreTrainedTokenizer."""

    def __init__(self, model):
        self._model = model

    def __call__(self, text, return_tensors=None, add_bos=True):
        ids = self._model.tokenize(text, add_bos=add_bos)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids if not (skip_special_tokens and int(i) in (1, 2))]
        return self._model.detokenize(ids)


class LoomForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LoomConfig

    def __init__(self, config):
        super().__init__(config)
        self._rt = None
        self._model = None
        self._ctx = None
        self._dev = None

    @classmethod
    def from_flashed(cls, model_dir, *, transport="sim", n_ctx=2048, device=None):
        rt = loom.Runtime()
        model = loom.Model(rt, model_dir)
        if device is None:
            if transport == "sim":
                device = loom.Device.open_sim(rt, model, model_dir)
            elif isinstance(transport, tuple) and transport[0] == "uart":
                device = loom.Device.open_uart(rt, transport[1], transport[2])
            elif transport == "usb":
                device = loom.Device.open_usb(rt)
            else:
                raise ValueError(f"unknown transport {transport!r}")
        # Load the on-chip BRAM weight cache onto a real device (no-op for sim and
        # models built without --fp-bram-cache-kb).
        model.prepare(device)
        self = cls(LoomConfig.from_info(model.info()))
        self._rt = rt
        self._model = model
        self._ctx = loom.Context(model, n_ctx)
        self._dev = device
        return self

    # This model holds no nn parameters (compute is on-device), so the base
    # PreTrainedModel device/dtype (which read self.parameters()) would raise.
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    @property
    def tokenizer(self):
        return LoomTokenizer(self._model)

    # We manage our own (external, on-device) KV cache, so tell generate not to
    # build a DynamicCache and hand us None instead (transformers 5.x escape route).
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    # transformers 5.x pre-slices input_ids to the new tokens and drops
    # cache_position, so the position is the cache's current length.
    def forward(self, input_ids=None, past_key_values=None, use_cache=True, **kw):
        assert input_ids.shape[0] == 1, "loom is single-sequence (batch==1)"
        pos = past_key_values.get_seq_length() if past_key_values is not None else 0
        tokens = [int(t) for t in input_ids[0].tolist()]
        logits_np = self._ctx.eval(self._dev, tokens, pos)  # [vocab]
        logits = torch.from_numpy(np.asarray(logits_np)).view(1, 1, -1)
        cache = past_key_values if isinstance(past_key_values, LoomCache) else LoomCache()
        cache._len = pos + len(tokens)
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)


__all__ = ["LoomConfig", "LoomForCausalLM", "LoomTokenizer", "LoomCache"]
