"""The runtime loads the genip-compiled LTB1 BPE tokenizer via magic sniff and
round-trips the golden vector. Skips if the gitignored genip output is absent."""
import os
import pytest
import loom

_GENIP = "/home/ross/Midstall/loom/.cache/ternary-stories-genip"


def test_bpe_tokenizer_roundtrips_golden_vector():
    if not os.path.isfile(os.path.join(_GENIP, "tokenizer.bin")):
        pytest.skip("genip ternary output absent")
    rt = loom.Runtime()
    model = loom.Model(rt, _GENIP)
    ids = model.tokenize("Once upon a time", add_bos=False)
    assert list(ids) == [432, 447, 259, 396]
    assert model.detokenize([432, 447, 259, 396]) == "Once upon a time"
