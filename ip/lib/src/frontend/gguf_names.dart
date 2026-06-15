import 'dart:typed_data';

import '../ir/tensor.dart';
import '../weights/gguf.dart';
import '../weights/weight_store.dart';

/// Translate a HuggingFace llama weight-tensor name to its GGUF (llama.cpp)
/// equivalent. Throws [ArgumentError] on a name outside the known llama set.
String hfToGgufName(String hf) {
  const flat = {
    'model.embed_tokens.weight': 'token_embd.weight',
    'model.norm.weight': 'output_norm.weight',
    'lm_head.weight': 'output.weight',
  };
  final direct = flat[hf];
  if (direct != null) return direct;

  final m = RegExp(r'^model\.layers\.(\d+)\.(.+)$').firstMatch(hf);
  if (m != null) {
    final i = m.group(1)!;
    const perLayer = {
      'input_layernorm.weight': 'attn_norm.weight',
      'self_attn.q_proj.weight': 'attn_q.weight',
      'self_attn.k_proj.weight': 'attn_k.weight',
      'self_attn.v_proj.weight': 'attn_v.weight',
      'self_attn.o_proj.weight': 'attn_output.weight',
      'post_attention_layernorm.weight': 'ffn_norm.weight',
      'mlp.gate_proj.weight': 'ffn_gate.weight',
      'mlp.up_proj.weight': 'ffn_up.weight',
      'mlp.down_proj.weight': 'ffn_down.weight',
    };
    final tail = perLayer[m.group(2)];
    if (tail != null) return 'blk.$i.$tail';
  }
  throw ArgumentError.value(hf, 'hfName', 'no GGUF name mapping');
}

/// A [WeightStore] view over a [GgufStore] that presents tensors under their
/// HuggingFace names AND in HuggingFace layout. Two layout fixes are applied:
///
/// 1. Transpose. GGUF reports a 2D tensor's dims reversed from HF (`ne` order),
///    while the raw data is already row-major in HF `[out, in]` order. So the
///    fix is to reverse the reported shape and keep the bytes.
/// 2. RoPE un-permutation. llama.cpp's converter permutes the q and k projection
///    ROWS so its rope convention works, HF uses the other convention. We invert
///    that row permutation for `attn_q`/`attn_k` so the weights match HF.
///
/// F16/F32 tensors pass through, Q8_0 is dequantized to f32. The K-quants are
/// intentionally NOT supported (dequantize throws): ingest a high-precision GGUF
/// and let Loom do its own low-bit quant, rather than double-quantizing.
class GgufHfView implements WeightStore {
  final GgufStore inner;
  final int numHeads;
  final int numKvHeads;
  final int headDim;

  GgufHfView(
    this.inner, {
    required this.numHeads,
    required this.numKvHeads,
    required this.headDim,
  });

  @override
  bool contains(String name) => inner.contains(hfToGgufName(name));

  // The underlying GGUF store's own tensor names, not their HF equivalents
  // (hfToGgufName has no general inverse for arbitrary GGUF names).
  @override
  Iterable<String> get names => inner.names;

  @override
  TensorView get(String name) {
    final gname = hfToGgufName(name);
    final t = _rawGgufTensor(gname);
    if (gname.endsWith('attn_q.weight')) {
      return _unpermuteRows(t, name, numHeads);
    }
    if (gname.endsWith('attn_k.weight')) {
      return _unpermuteRows(t, name, numKvHeads);
    }
    // Everything else: reverse the shape (transpose), keep the bytes.
    return TensorView(
      name: name,
      shape: t.shape.reversed.toList(),
      dtype: t.dtype,
      bytes: t.bytes,
    );
  }

  /// The GGUF tensor in its native GGUF shape (ne order), as a float TensorView.
  /// F16/F32 pass through, Q8_0 is dequantized to f32. Unsupported quant types
  /// throw (via [GgufStore.dequantize]), which is the intended precision guard.
  TensorView _rawGgufTensor(String gname) {
    final type = inner.typeOf(gname);
    if (type == GgmlType.f16 || type == GgmlType.f32) {
      return inner.get(gname);
    }
    final values = inner.dequantize(gname);
    final shape = inner.shapeOf(gname);
    final bytes = ByteData(values.length * 4);
    for (var i = 0; i < values.length; i++) {
      bytes.setFloat32(i * 4, values[i], Endian.little);
    }
    return TensorView(
      name: gname,
      shape: shape,
      dtype: TensorDType.f32,
      bytes: bytes,
    );
  }

  /// Invert llama.cpp's q/k row permutation. GGUF stores head rows grouped as
  /// [head_dim/2, 2], HF wants [2, head_dim/2]. For an HF output row d within a
  /// head (d = s*(head_dim/2) + p, s in {0,1}) the source GGUF row is p*2 + s.
  TensorView _unpermuteRows(TensorView t, String hfName, int nHead) {
    final hfShape = t.shape.reversed.toList(); // [out, in]
    final out = hfShape[0];
    final inF = hfShape[1];
    final bpe = t.dtype.bytesPerElement;
    final rowBytes = inF * bpe;
    final src = t.bytes.buffer.asUint8List(
      t.bytes.offsetInBytes,
      t.bytes.lengthInBytes,
    );
    final dst = Uint8List(out * rowBytes);
    final half = headDim ~/ 2;
    for (var head = 0; head < nHead; head++) {
      for (var d = 0; d < headDim; d++) {
        final s = d ~/ half;
        final p = d % half;
        final srcRow = head * headDim + p * 2 + s;
        final dstRow = head * headDim + d;
        dst.setRange(
          dstRow * rowBytes,
          dstRow * rowBytes + rowBytes,
          src,
          srcRow * rowBytes,
        );
      }
    }
    return TensorView(
      name: hfName,
      shape: hfShape,
      dtype: t.dtype,
      bytes: ByteData.sublistView(dst),
    );
  }
}
