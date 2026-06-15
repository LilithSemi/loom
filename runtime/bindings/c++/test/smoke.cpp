#include <loom-cpp.hpp>

#include <cstdio>
#include <cstring>

int main() {
  Loom::Runtime rt;

  // Host ops link and run without a device.
  float v[4] = {1.f, 2.f, 3.f, 4.f};
  Loom::softmax(v, 4);
  float sum = v[0] + v[1] + v[2] + v[3];
  if (sum < 0.99f || sum > 1.01f) {
    std::fprintf(stderr, "FAIL: softmax did not normalize (sum=%f)\n", sum);
    return 1;
  }

  // LayerNorm zero-centers: sum of the normalized (gamma=1,beta=0) output ~= 0.
  float ln[4] = {1.f, 2.f, 3.f, 4.f};
  float g[4] = {1.f, 1.f, 1.f, 1.f};
  float b[4] = {0.f, 0.f, 0.f, 0.f};
  float lno[4];
  Loom::layernorm(ln, g, b, 0.f, lno, 4);
  float lnsum = lno[0] + lno[1] + lno[2] + lno[3];
  if (lnsum < -0.01f || lnsum > 0.01f) {
    std::fprintf(stderr, "FAIL: layernorm not zero-centered (sum=%f)\n", lnsum);
    return 1;
  }

  // GELU(0)=0.
  float ge[1] = {0.f};
  Loom::gelu(ge, 1);
  if (ge[0] < -1e-6f || ge[0] > 1e-6f) {
    std::fprintf(stderr, "FAIL: gelu(0) != 0 (%f)\n", ge[0]);
    return 1;
  }

  // MoE router: logits rank experts 0>1>2>3, top-2 = {0,1}, weights sum to 1.
  float logits[4] = {4.f, 3.f, 2.f, 1.f};
  size_t idx[2];
  float w[2];
  size_t k = Loom::moe_route(logits, 4, 2, true, idx, w);
  if (k != 2 || idx[0] != 0 || idx[1] != 1 || w[0] + w[1] < 0.99f ||
      w[0] + w[1] > 1.01f) {
    std::fprintf(stderr, "FAIL: moe_route k=%zu idx=%zu,%zu w=%f,%f\n", k,
                 idx[0], idx[1], w[0], w[1]);
    return 1;
  }

  try {
    Loom::Model m(rt, "/definitely/not/a/model");
    (void)m;
    std::fprintf(stderr, "FAIL: expected model load to throw\n");
    return 1;
  } catch (const Loom::Error &e) {
    if (e.code != LOOM_ERR_MODEL_LOAD || std::strlen(e.what()) == 0) {
      std::fprintf(stderr, "FAIL: unexpected error code=%d msg=%s\n", e.code,
                   e.what());
      return 1;
    }
  }
  return 0;
}
