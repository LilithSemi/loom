#pragma once

// Header-only C++ RAII layer over the Loom C ABI (loom.h). There is no
// separate C++ library: every method is an inline call into the extern "C"
// symbols in libloom. Non-OK statuses throw Loom::Error.

#include "loom.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Loom {

class Error : public std::runtime_error {
public:
  int code;
  explicit Error(const loom_error_t &e)
      : std::runtime_error(e.message), code(e.code) {}
};

inline void check(loom_status_t rc, const loom_error_t &err) {
  if (rc != LOOM_OK)
    throw Error(err);
}

class Runtime {
public:
  Runtime() {
    loom_error_t e{};
    check(loom_runtime_create(&rt_, &e), e);
  }
  ~Runtime() {
    if (rt_)
      loom_runtime_destroy(rt_);
  }
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&o) noexcept : rt_(o.rt_) { o.rt_ = nullptr; }

  loom_runtime_t *raw() const { return rt_; }

private:
  loom_runtime_t *rt_ = nullptr;
};

class Model; // for Device::openSim, which needs a loaded model's handle

class Device {
public:
  static Device openUart(Runtime &rt, const std::string &port, uint32_t baud) {
    Device d;
    loom_error_t e{};
    check(loom_device_open_uart(rt.raw(), port.c_str(), baud, &d.dev_, &e), e);
    return d;
  }
  static Device openUsb(Runtime &rt) {
    Device d;
    loom_error_t e{};
    check(loom_device_open_usb(rt.raw(), &d.dev_, &e), e);
    return d;
  }
  // Reference/testing backend from a genip dir. Defined after Model (below).
  static Device openSim(Runtime &rt, Model &model, const std::string &dir);

  Device() = default;
  ~Device() {
    if (dev_)
      loom_device_close(dev_);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&o) noexcept : dev_(o.dev_) { o.dev_ = nullptr; }

  loom_device_t *raw() const { return dev_; }

private:
  loom_device_t *dev_ = nullptr;
};

class Model {
public:
  Model(Runtime &rt, const std::string &dir) {
    loom_error_t e{};
    check(loom_model_load(rt.raw(), dir.c_str(), &m_, &e), e);
  }
  ~Model() {
    if (m_)
      loom_model_free(m_);
  }
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&o) noexcept : m_(o.m_) { o.m_ = nullptr; }

  // Called once per generated token with its id and rendered UTF-8 piece.
  using Callback = std::function<void(uint32_t id, std::string_view piece)>;

  // Loads this model's BRAM weight cache onto `dev` once, before generating.
  // No-op for non-cache models and the sim device; call after opening a real
  // device and before generate()/eval().
  void prepare(Device &dev) {
    loom_error_t e{};
    check(loom_device_prepare(m_, dev.raw(), &e), e);
  }

  // Runs the vision tower + projector on pixels, returning seq_len rows of
  // output_dim projected text-space embeddings flattened row-major.
  std::vector<float> encode_image(Device &dev, const std::vector<float> &pixels) {
    // Generous cap; the call reports the real dims and we shrink to fit.
    std::vector<float> out(pixels.size() * 64 + 4096);
    size_t seq = 0, dim = 0;
    loom_error_t e{};
    check(loom_encode_image(m_, dev.raw(), pixels.data(), pixels.size(),
                            out.data(), out.size(), &seq, &dim, &e),
          e);
    out.resize(seq * dim);
    return out;
  }

  // One-shot: decode a JPEG and run the tower+projector, returning the projected
  // text-space image embeddings (seq_len rows of output_dim, flattened).
  std::vector<float> load_image(Device &dev, const std::vector<uint8_t> &jpeg) {
    std::vector<float> out(1 << 20); // generous; shrunk to the reported dims
    size_t seq = 0, dim = 0;
    loom_error_t e{};
    check(loom_load_image(m_, dev.raw(), jpeg.data(), jpeg.size(), out.data(),
                          out.size(), &seq, &dim, &e),
          e);
    out.resize(seq * dim);
    return out;
  }

  // Streams generation to `cb` and returns the number of tokens produced.
  // max_tokens/poll_timeout/repeat_window of 0 select the runtime defaults.
  size_t generate(Device &dev, std::string_view prompt, Callback cb,
                  size_t max_tokens = 0, size_t poll_timeout = 0,
                  size_t repeat_window = 0) {
    std::string p(prompt);
    loom_generate_opts_t opts{p.c_str(), max_tokens, poll_timeout, repeat_window};
    size_t produced = 0;
    loom_error_t e{};
    auto trampoline = [](uint32_t id, const char *u, size_t n, void *ud) {
      (*static_cast<Callback *>(ud))(id, std::string_view(u, n));
    };
    check(loom_generate(m_, dev.raw(), &opts, trampoline, &cb, &produced, &e),
          e);
    return produced;
  }

  loom_model_t *raw() const { return m_; }

  loom_model_info_t info() const {
    loom_model_info_t i{};
    loom_model_info(m_, &i);
    return i;
  }

  // Borrowed handle into the model's matrix table; valid while this Model
  // lives.
  const loom_matrix_t *matrix(const std::string &name) const {
    const loom_matrix_t *mt = nullptr;
    loom_error_t e{};
    check(loom_model_matrix_by_name(m_, name.c_str(), &mt, &e), e);
    return mt;
  }

  // One flash-resident matmul: out[rows] = mat @ x[cols].
  std::vector<float> linear(Device &dev, const loom_matrix_t *mat,
                            const std::vector<float> &x,
                            size_t poll_timeout = 0) {
    uint32_t rows = 0, cols = 0;
    loom_matrix_dims(mat, &rows, &cols);
    std::vector<float> out(rows);
    loom_error_t e{};
    check(
        loom_linear(m_, dev.raw(), mat, x.data(), out.data(), poll_timeout, &e),
        e);
    return out;
  }

  // Column-tiled matmul (block_cols = device maxCols) for wide matrices.
  std::vector<float> linearColTiled(Device &dev, const loom_matrix_t *mat,
                                    size_t block_cols, const std::vector<float> &x,
                                    size_t poll_timeout = 0) {
    uint32_t rows = 0, cols = 0;
    loom_matrix_dims(mat, &rows, &cols);
    std::vector<float> out(rows);
    loom_error_t e{};
    check(loom_linear_col_tiled(m_, dev.raw(), mat, block_cols, x.data(),
                                out.data(), poll_timeout, &e),
          e);
    return out;
  }

  std::vector<uint32_t> tokenize(const std::string &text, bool add_bos = true) {
    std::vector<uint32_t> ids(text.size() + 8);
    size_t n = 0;
    loom_error_t e{};
    check(loom_tokenize(m_, text.c_str(), add_bos, ids.data(), ids.size(), &n,
                        &e),
          e);
    ids.resize(n);
    return ids;
  }

  std::string detokenize(const std::vector<uint32_t> &ids) {
    std::string buf(ids.size() * 16 + 16, '\0');
    size_t len = 0;
    loom_error_t e{};
    check(loom_detokenize(m_, ids.data(), ids.size(), buf.data(), buf.size(),
                          &len, &e),
          e);
    buf.resize(len);
    return buf;
  }

private:
  loom_model_t *m_ = nullptr;
};

inline Device Device::openSim(Runtime &rt, Model &model,
                              const std::string &dir) {
  Device d;
  loom_error_t e{};
  check(loom_device_open_sim(rt.raw(), model.raw(), dir.c_str(), &d.dev_, &e),
        e);
  return d;
}

// Stateful decode: owns a persistent KV cache. Feed tokens, read logits,
// sample, feed the next token.
class Context {
public:
  Context(Model &model, size_t n_ctx) {
    loom_error_t e{};
    check(loom_context_create(model.raw(), n_ctx, &ctx_, &e), e);
    vocab_ = model.info().vocab;
  }
  ~Context() {
    if (ctx_)
      loom_context_free(ctx_);
  }
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&o) noexcept : ctx_(o.ctx_), vocab_(o.vocab_) {
    o.ctx_ = nullptr;
  }

  // Steps the tokens starting at `pos`, returning the vocab logits for the
  // last.
  std::vector<float> eval(Device &dev, const std::vector<uint32_t> &tokens,
                          size_t pos) {
    std::vector<float> logits(vocab_);
    loom_error_t e{};
    check(loom_eval(ctx_, dev.raw(), tokens.data(), tokens.size(), pos,
                    logits.data(), &e),
          e);
    return logits;
  }

  // Vision-language eval: steps the tokens from `pos`, splicing vision_embeds
  // (num_vision rows of hidden, from Model::encode_image) at each image
  // placeholder. Returns the last position's vocab logits.
  std::vector<float> eval_vlm(Device &dev, const std::vector<uint32_t> &tokens,
                              size_t pos, const std::vector<float> &vision_embeds,
                              size_t num_vision) {
    std::vector<float> logits(vocab_);
    loom_error_t e{};
    check(loom_eval_vlm(ctx_, dev.raw(), tokens.data(), tokens.size(), pos,
                        vision_embeds.data(), num_vision, logits.data(), &e),
          e);
    return logits;
  }

  // MTP draft: prefills tokens, chains the MTP heads, returns num_modules+1
  // drafted token ids (main next token + one per module). The host verifies
  // them with one eval() and accepts the longest correct prefix.
  std::vector<uint32_t> mtp_draft(Device &dev,
                                  const std::vector<uint32_t> &tokens) {
    std::vector<uint32_t> draft(512); // holds num_modules+1 (always small)
    size_t k = 0;
    loom_error_t e{};
    check(loom_mtp_draft(ctx_, dev.raw(), tokens.data(), tokens.size(),
                         draft.data(), &k, &e),
          e);
    draft.resize(k);
    return draft;
  }

private:
  loom_context_t *ctx_ = nullptr;
  uint32_t vocab_ = 0;
};

inline const char *version() { return loom_version(); }

// Host-side ops (no device), for callers assembling their own forward pass.
inline void rmsnorm(const float *x, const float *gamma, float eps, float *out,
                    size_t n) {
  loom_rmsnorm(x, gamma, eps, out, n);
}
inline void silu(float *x, size_t n) { loom_silu(x, n); }
inline void layernorm(const float *x, const float *gamma, const float *beta,
                      float eps, float *out, size_t n) {
  loom_layernorm(x, gamma, beta, eps, out, n);
}
inline void gelu(float *x, size_t n) { loom_gelu(x, n); }
inline void softmax(float *x, size_t n) { loom_softmax(x, n); }
inline void rope(float *head, size_t pos, float theta, size_t head_dim) {
  loom_rope(head, pos, theta, head_dim);
}
inline void add(float *x, const float *bias, size_t n) { loom_add(x, bias, n); }
// MoE router selection: softmax + top-k + (optional) renormalized weights.
// Returns the number of experts chosen; idx/w must hold top_k elements.
inline size_t moe_route(const float *logits, size_t num_experts, size_t top_k,
                        bool norm_topk, size_t *idx, float *w) {
  return loom_moe_route(logits, num_experts, top_k, norm_topk, idx, w);
}

} // namespace Loom
