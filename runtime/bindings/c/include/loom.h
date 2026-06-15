#pragma once

/* C ABI for the Loom runtime: open a transport, load a genip model, and stream
 * generated tokens. Every fallible call returns a `loom_status_t` and fills a
 * caller-owned `loom_error_t` (no global/thread-local error state). Handles are
 * opaque; create them via the `*_create`/`*_open`/`*_load` calls and release
 * them with the matching `*_destroy`/`*_close`/`*_free`. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_status {
    LOOM_OK = 0,
    LOOM_ERR_ALLOC = -1,
    LOOM_ERR_TRANSPORT = -2,
    LOOM_ERR_MODEL_LOAD = -3,
    LOOM_ERR_GENERATE = -4,
    LOOM_ERR_INVALID = -5,
    LOOM_ERR_CONTEXT = -6,
    LOOM_ERR_EVAL = -7
} loom_status_t;

/* Filled on failure with a status code and a NUL-terminated message (the Zig
 * error name). Pass `NULL` to any call to ignore error detail. */
typedef struct loom_error {
    int code;
    char message[256];
} loom_error_t;

typedef struct loom_runtime loom_runtime_t;
typedef struct loom_device loom_device_t;
typedef struct loom_model loom_model_t;

/* Runtime owns the async I/O backing every device and model. Create one first. */
loom_status_t loom_runtime_create(loom_runtime_t **out, loom_error_t *err);
void loom_runtime_destroy(loom_runtime_t *rt);

/* Devices: real silicon over a transport. */
loom_status_t loom_device_open_uart(loom_runtime_t *rt, const char *port, uint32_t baud, loom_device_t **out, loom_error_t *err);
loom_status_t loom_device_open_usb(loom_runtime_t *rt, loom_device_t **out, loom_error_t *err);
/* Reference/testing backend: in-process emulator built from a genip dir's flash
 * images (weights.bin + scales_flash.bin). Runs eval/linear/generate with no
 * hardware. */
loom_status_t loom_device_open_sim(loom_runtime_t *rt, loom_model_t *model, const char *dir, loom_device_t **out, loom_error_t *err);
void loom_device_close(loom_device_t *dev);

/* Models: a genip output directory (loom.json + scales + glue + tokenizer). */
loom_status_t loom_model_load(loom_runtime_t *rt, const char *dir, loom_model_t **out, loom_error_t *err);
void loom_model_free(loom_model_t *model);

/* Loads a BRAM-weight-cache model's hot weights onto `dev` once, before
 * generating. No-op for non-cache models and the sim device, so always safe to
 * call. Call after opening the device and before loom_eval/loom_generate. */
loom_status_t loom_device_prepare(loom_model_t *model, loom_device_t *dev, loom_error_t *err);

/* Called once per generated token with its id and rendered UTF-8 piece (not
 * NUL-terminated; use `len`). Fires on generated tokens only, not the prompt. */
typedef void (*loom_token_cb)(uint32_t id, const char *utf8, size_t len, void *user_data);

typedef struct loom_generate_opts {
    const char *prompt;   /* NUL-terminated; NULL or "" starts bare from BOS. */
    size_t max_tokens;    /* 0 = run until a stop token, capped at the model's
                             trained context (max_seq). */
    size_t poll_timeout;  /* 0 selects the runtime default. */
    size_t repeat_window; /* loop-stop: end when the last N output tokens recur.
                             0 selects the default (8). */
} loom_generate_opts_t;

/* Greedily decodes from the prompt, running every matmul on `dev`. Stops at EOS
 * or the model's BOS delimiter. `out_produced` (nullable) gets the token count. */
loom_status_t loom_generate(loom_model_t *model, loom_device_t *dev, const loom_generate_opts_t *opts,
                            loom_token_cb cb, void *user_data, size_t *out_produced, loom_error_t *err);

const char *loom_version(void);

/* ---- Lower-level surface: logits, matmul primitive, ops, tokenizer ---- */

/* Static model geometry. */
typedef struct loom_model_info {
    uint32_t hidden, vocab, layers, num_heads, num_kv_heads, head_dim, intermediate;
    uint32_t max_seq;   /* trained context length (0 if the manifest omits it) */
    uint32_t n_matrices;
    float rope_theta, norm_eps;
} loom_model_info_t;
loom_status_t loom_model_info(loom_model_t *model, loom_model_info_t *out);

/* A weight matrix, borrowed from the model (valid while the model lives). */
typedef struct loom_matrix loom_matrix_t;
size_t loom_model_num_matrices(loom_model_t *model);
loom_status_t loom_model_matrix(loom_model_t *model, size_t index, const loom_matrix_t **out, loom_error_t *err);
loom_status_t loom_model_matrix_by_name(loom_model_t *model, const char *name, const loom_matrix_t **out, loom_error_t *err);
void loom_matrix_dims(const loom_matrix_t *mat, uint32_t *rows, uint32_t *cols);

/* One flash-resident matmul: out[rows] = mat @ x[cols]. poll_timeout 0 = default. */
loom_status_t loom_linear(loom_model_t *model, loom_device_t *dev, const loom_matrix_t *mat,
                          const float *x, float *out, size_t poll_timeout, loom_error_t *err);

/* Column-tiled matmul for a matrix wider than the accelerator's column capacity
 * (block_cols = device maxCols): runs the wide matmul as consecutive col-blocks
 * and sums the partials, so a small accelerator handles any width. */
loom_status_t loom_linear_col_tiled(loom_model_t *model, loom_device_t *dev, const loom_matrix_t *mat,
                                    size_t block_cols, const float *x, float *out, size_t poll_timeout,
                                    loom_error_t *err);

/* Stateful decode: a context owns a persistent KV cache (size n_ctx). */
typedef struct loom_context loom_context_t;
loom_status_t loom_context_create(loom_model_t *model, size_t n_ctx, loom_context_t **out, loom_error_t *err);
void loom_context_free(loom_context_t *ctx);
/* Steps `n` tokens starting at `pos`, updating the KV cache, and writes `vocab`
 * logits for the LAST token into `logits_out`. Caller samples + feeds next. */
loom_status_t loom_eval(loom_context_t *ctx, loom_device_t *dev, const uint32_t *tokens, size_t n, size_t pos, float *logits_out, loom_error_t *err);

/* Multi-Token Prediction draft: prefills tokens[0..n], then chains the MTP heads
 * to draft num_modules+1 tokens (main next token + one per module) into
 * draft_out, writing the count to k_out. draft_out must hold num_modules+1 ids.
 * The device-side draft primitive; the host verifies with one loom_eval over the
 * drafted positions and accepts the longest correct prefix (output stays greedy).
 * Fails if the model has no MTP heads. */
loom_status_t loom_mtp_draft(loom_context_t *ctx, loom_device_t *dev, const uint32_t *tokens, size_t n, uint32_t *draft_out, size_t *k_out, loom_error_t *err);

/* Vision encode: runs the ViT tower + projector on pixels (num_channels*
 * image_size^2, channels-first, resized+normalized), writing seq_len*output_dim
 * projected text-space embeddings into out (out_cap guards it). seq_out/dim_out
 * report the produced dims. The host splices these at the image placeholder and
 * calls loom_eval. Fails if the model has no vision tower / projector. */
loom_status_t loom_encode_image(loom_model_t *model, loom_device_t *dev, const float *pixels, size_t npix, float *out, size_t out_cap, size_t *seq_out, size_t *dim_out, loom_error_t *err);

/* One-shot: decode a baseline JPEG (jpeg[0..n]), resize+normalize per the model's
 * vision config, run the tower+projector on dev, write seq_len*output_dim
 * text-space embeddings into out. The image front door for the VLM. */
loom_status_t loom_load_image(loom_model_t *model, loom_device_t *dev, const uint8_t *jpeg, size_t n, float *out, size_t out_cap, size_t *seq_out, size_t *dim_out, loom_error_t *err);

/* Vision-language eval: steps n tokens from pos; each token == image_token_index
 * consumes the next row of vision_embeds (num_vision*hidden, from
 * loom_encode_image) as its input embedding. Writes the last position's vocab
 * logits into logits_out. Placeholder count must equal num_vision. */
loom_status_t loom_eval_vlm(loom_context_t *ctx, loom_device_t *dev, const uint32_t *tokens, size_t n, size_t pos, const float *vision_embeds, size_t num_vision, float *logits_out, loom_error_t *err);

/* Host-side ops (no device). In-place where the pointer is non-const. */
void loom_rmsnorm(const float *x, const float *gamma, float eps, float *out, size_t n);
void loom_silu(float *x, size_t n);
/* LayerNorm (ViT towers): out = (x-mean)/sqrt(var+eps)*gamma+beta. */
void loom_layernorm(const float *x, const float *gamma, const float *beta, float eps, float *out, size_t n);
/* GELU in place (tanh approximation), the ViT MLP activation. */
void loom_gelu(float *x, size_t n);
void loom_softmax(float *x, size_t n);
void loom_rope(float *head, size_t pos, float theta, size_t head_dim);

/* Elementwise in-place add x[i] += bias[i] (q/k/v bias in a composed forward,
 * e.g. Qwen2). The whole-model loom_generate/loom_eval path applies bias itself. */
void loom_add(float *x, const float *bias, size_t n);

/* Mixture-of-Experts router selection: softmax logits[0..num_experts], pick the
 * top_k highest, write their indices to idx[0..top_k] and combine weights to
 * w[0..top_k] (renormalized to sum 1 when norm_topk). Returns the number of
 * experts selected (min(top_k, num_experts)). For callers composing a MoE
 * forward from the host ops + loom_linear; loom_eval routes internally. */
size_t loom_moe_route(const float *logits, size_t num_experts, size_t top_k, bool norm_topk, size_t *idx, float *w);

/* Tokenizer over the model's vocab. */
loom_status_t loom_tokenize(loom_model_t *model, const char *text, bool add_bos, uint32_t *out_ids, size_t cap, size_t *n, loom_error_t *err);
loom_status_t loom_detokenize(loom_model_t *model, const uint32_t *ids, size_t n, char *buf, size_t cap, size_t *len, loom_error_t *err);

#ifdef __cplusplus
}
#endif
