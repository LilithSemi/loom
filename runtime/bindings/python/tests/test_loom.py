import numpy as np

import loom


def test_version():
    assert "loom" in loom.version()


def test_softmax_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    got = loom.softmax(x)
    ref = np.exp(x - x.max())
    ref /= ref.sum()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_silu_matches_numpy():
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32)
    got = loom.silu(x)
    ref = x / (1.0 + np.exp(-x))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_layernorm_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gamma = np.ones(4, dtype=np.float32)
    beta = np.zeros(4, dtype=np.float32)
    got = loom.layernorm(x, gamma, beta, 0.0)
    ref = (x - x.mean()) / np.sqrt(x.var() + 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_gelu_matches_tanh_approx():
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    got = loom.gelu(x)
    ref = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_moe_route_picks_top_k_and_renormalizes():
    # logits rank experts 0>1>2>3; top-2 = {0,1}; norm_topk rescales to sum 1.
    logits = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    idx, w = loom.moe_route(logits, 2, True)
    assert list(idx) == [0, 1]
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5, atol=1e-6)
    e = np.exp(1.0)
    np.testing.assert_allclose(w, [e / (e + 1.0), 1.0 / (e + 1.0)], rtol=1e-4, atol=1e-5)


def test_moe_route_without_norm_keeps_softmax_probs():
    logits = np.zeros(4, dtype=np.float32)  # uniform -> each prob 0.25
    idx, w = loom.moe_route(logits, 2, False)
    assert len(idx) == 2
    np.testing.assert_allclose(w, [0.25, 0.25], rtol=1e-5, atol=1e-6)
