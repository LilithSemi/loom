// nanobind extension module `pyloom`, wrapping the header-only C++ layer
// (loom-cpp.hpp) over the Loom C ABI. Tensors cross as nb::ndarray (numpy or
// torch, zero-copy in; a fresh numpy array out). keep_alive ties dependent
// handles' lifetimes so Python GC ordering can't dangle a C pointer.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <loom-cpp.hpp>

#include <algorithm>
#include <vector>

namespace nb = nanobind;

// 1-D contiguous float input; accepts numpy and torch (dlpack) CPU arrays.
using F32In = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static std::vector<float> to_vec(const F32In &a) {
    return std::vector<float>(a.data(), a.data() + a.shape(0));
}

// A fresh owning numpy array holding a copy of `v`.
static nb::ndarray<nb::numpy, float> to_np(const std::vector<float> &v) {
    float *data = new float[v.size()];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    return nb::ndarray<nb::numpy, float>(data, {v.size()}, owner);
}

// A borrowed matrix handle; valid while its owning Model is alive (enforced by
// keep_alive on Model.matrix). loom_matrix_t is opaque, so we hold the pointer.
struct PyMatrix {
    const loom_matrix_t *mat;
};

NB_MODULE(pyloom, m) {
    nb::exception<Loom::Error>(m, "LoomError");

    nb::class_<Loom::Runtime>(m, "Runtime").def(nb::init<>());

    nb::class_<Loom::Device>(m, "Device")
        .def_static("open_uart", &Loom::Device::openUart, nb::keep_alive<0, 1>(),
                    nb::arg("rt"), nb::arg("port"), nb::arg("baud"))
        .def_static("open_usb", &Loom::Device::openUsb, nb::keep_alive<0, 1>(), nb::arg("rt"))
        .def_static("open_sim", &Loom::Device::openSim, nb::keep_alive<0, 1>(),
                    nb::arg("rt"), nb::arg("model"), nb::arg("dir"));

    nb::class_<PyMatrix>(m, "Matrix").def_prop_ro("dims", [](const PyMatrix &pm) {
        uint32_t r = 0, c = 0;
        loom_matrix_dims(pm.mat, &r, &c);
        return nb::make_tuple(r, c);
    });

    nb::class_<Loom::Model>(m, "Model")
        .def(nb::init<Loom::Runtime &, const std::string &>(), nb::keep_alive<1, 2>(),
             nb::arg("rt"), nb::arg("dir"))
        .def("info",
             [](Loom::Model &self) {
                 auto i = self.info();
                 nb::dict d;
                 d["hidden"] = i.hidden;
                 d["vocab"] = i.vocab;
                 d["layers"] = i.layers;
                 d["num_heads"] = i.num_heads;
                 d["num_kv_heads"] = i.num_kv_heads;
                 d["head_dim"] = i.head_dim;
                 d["intermediate"] = i.intermediate;
                 d["max_seq"] = i.max_seq;
                 d["n_matrices"] = i.n_matrices;
                 d["rope_theta"] = i.rope_theta;
                 d["norm_eps"] = i.norm_eps;
                 return d;
             })
        .def("matrix",
             [](Loom::Model &self, const std::string &name) { return PyMatrix{self.matrix(name)}; },
             nb::keep_alive<0, 1>(), nb::arg("name"))
        .def("tokenize", &Loom::Model::tokenize, nb::arg("text"), nb::arg("add_bos") = true)
        .def("detokenize", &Loom::Model::detokenize, nb::arg("ids"))
        .def("linear",
             [](Loom::Model &self, Loom::Device &dev, const PyMatrix &mat, F32In x) {
                 return to_np(self.linear(dev, mat.mat, to_vec(x)));
             },
             nb::arg("dev"), nb::arg("matrix"), nb::arg("x"))
        .def("linear_col_tiled",
             [](Loom::Model &self, Loom::Device &dev, const PyMatrix &mat, size_t block_cols, F32In x) {
                 return to_np(self.linearColTiled(dev, mat.mat, block_cols, to_vec(x)));
             },
             nb::arg("dev"), nb::arg("matrix"), nb::arg("block_cols"), nb::arg("x"))
        .def("prepare",
             [](Loom::Model &self, Loom::Device &dev) { self.prepare(dev); },
             nb::arg("dev"))
        .def("encode_image",
             [](Loom::Model &self, Loom::Device &dev, F32In pixels) {
                 return to_np(self.encode_image(dev, to_vec(pixels)));
             },
             nb::arg("dev"), nb::arg("pixels"))
        .def("load_image",
             [](Loom::Model &self, Loom::Device &dev,
                nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig> jpeg) {
                 std::vector<uint8_t> bytes(jpeg.data(), jpeg.data() + jpeg.shape(0));
                 return to_np(self.load_image(dev, bytes));
             },
             nb::arg("dev"), nb::arg("jpeg"))
        .def("generate",
             [](Loom::Model &self, Loom::Device &dev, const std::string &prompt, nb::callable cb,
                size_t max_tokens, size_t poll_timeout, size_t repeat_window) {
                 return self.generate(
                     dev, prompt,
                     [&](uint32_t id, std::string_view piece) {
                         cb(id, nb::str(piece.data(), piece.size()));
                     },
                     max_tokens, poll_timeout, repeat_window);
             },
             nb::arg("dev"), nb::arg("prompt"), nb::arg("callback"), nb::arg("max_tokens") = 0,
             nb::arg("poll_timeout") = 0, nb::arg("repeat_window") = 0);

    nb::class_<Loom::Context>(m, "Context")
        .def(nb::init<Loom::Model &, size_t>(), nb::keep_alive<1, 2>(), nb::arg("model"),
             nb::arg("n_ctx"))
        .def("eval",
             [](Loom::Context &self, Loom::Device &dev, const std::vector<uint32_t> &tokens,
                size_t pos) { return to_np(self.eval(dev, tokens, pos)); },
             nb::arg("dev"), nb::arg("tokens"), nb::arg("pos"))
        .def("mtp_draft",
             [](Loom::Context &self, Loom::Device &dev,
                const std::vector<uint32_t> &tokens) {
                 return self.mtp_draft(dev, tokens); // -> list[int]
             },
             nb::arg("dev"), nb::arg("tokens"))
        .def("eval_vlm",
             [](Loom::Context &self, Loom::Device &dev, const std::vector<uint32_t> &tokens,
                size_t pos, F32In vision_embeds, size_t num_vision) {
                 return to_np(self.eval_vlm(dev, tokens, pos, to_vec(vision_embeds), num_vision));
             },
             nb::arg("dev"), nb::arg("tokens"), nb::arg("pos"), nb::arg("vision_embeds"),
             nb::arg("num_vision"));

    m.def("version", []() { return Loom::version(); });
    m.def("rmsnorm",
          [](F32In x, F32In gamma, float eps) {
              std::vector<float> out(x.shape(0));
              Loom::rmsnorm(x.data(), gamma.data(), eps, out.data(), x.shape(0));
              return to_np(out);
          },
          nb::arg("x"), nb::arg("gamma"), nb::arg("eps"));
    m.def("silu", [](F32In x) { auto v = to_vec(x); Loom::silu(v.data(), v.size()); return to_np(v); }, nb::arg("x"));
    m.def("layernorm",
          [](F32In x, F32In gamma, F32In beta, float eps) {
              std::vector<float> out(x.shape(0));
              Loom::layernorm(x.data(), gamma.data(), beta.data(), eps, out.data(), x.shape(0));
              return to_np(out);
          },
          nb::arg("x"), nb::arg("gamma"), nb::arg("beta"), nb::arg("eps"));
    m.def("gelu", [](F32In x) { auto v = to_vec(x); Loom::gelu(v.data(), v.size()); return to_np(v); }, nb::arg("x"));
    m.def("softmax", [](F32In x) { auto v = to_vec(x); Loom::softmax(v.data(), v.size()); return to_np(v); }, nb::arg("x"));
    m.def("add", [](F32In x, F32In bias) { auto v = to_vec(x); Loom::add(v.data(), bias.data(), v.size()); return to_np(v); }, nb::arg("x"), nb::arg("bias"));
    // MoE router selection -> (expert indices int64, combine weights f32), each
    // length min(top_k, num_experts). Composes a MoE FFN with .linear(...).
    m.def("moe_route",
          [](F32In logits, size_t top_k, bool norm_topk) {
              const size_t ne = logits.shape(0);
              const size_t k = std::min(top_k, ne);
              std::vector<size_t> idx(k);
              std::vector<float> w(k);
              const size_t chosen = Loom::moe_route(logits.data(), ne, top_k, norm_topk, idx.data(), w.data());
              int64_t *ip = new int64_t[chosen];
              for (size_t i = 0; i < chosen; i++) ip[i] = static_cast<int64_t>(idx[i]);
              nb::capsule iown(ip, [](void *p) noexcept { delete[] static_cast<int64_t *>(p); });
              nb::ndarray<nb::numpy, int64_t> iarr(ip, {chosen}, iown);
              float *wp = new float[chosen];
              std::copy(w.begin(), w.begin() + chosen, wp);
              nb::capsule wown(wp, [](void *p) noexcept { delete[] static_cast<float *>(p); });
              nb::ndarray<nb::numpy, float> warr(wp, {chosen}, wown);
              return nb::make_tuple(iarr, warr);
          },
          nb::arg("logits"), nb::arg("top_k"), nb::arg("norm_topk") = true);
}
