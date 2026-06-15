import 'dart:typed_data';

import '../ir/model_graph.dart';
import '../weights/binding.dart';
import 'attention.dart';
import 'moe.dart';
import 'mtp.dart';
import 'ops.dart';

/// Golden (fp64 reference) forward pass and greedy generation for a
/// decoder-only LLM (Llama/Qwen2 style).
///
/// All weight tensors are dequantized to [Float64List] once in the constructor
/// and cached. Subsequent calls to [forward] and [generate] perform pure-Dart
/// fp64 arithmetic.
/// Signature of a linear (matrix-vector) implementation: `y = W @ x` where W is
/// row-major [outDim, inDim]. The default is the fp64 [linear]. Callers can pass
/// a W4A8 quantized version, or one that routes the matmul to the Loom device,
/// to run the SAME forward with a different linear backend.
typedef LinearImpl =
    Float64List Function(Float64List w, int outDim, int inDim, Float64List x);

class GoldenRunner {
  final ModelGraph _graph;
  final Float64List _embedTokens;
  final Float64List _lmHead;
  final Float64List _finalNormGamma;
  final List<_LayerWeights> _layerWeights;

  /// Decoded Multi-Token Prediction module weights, null if the model has none.
  final List<MtpModuleWeights>? _mtpModules;

  /// The linear backend used for every projection / MLP / lm_head matmul.
  final LinearImpl _lin;

  GoldenRunner(ModelGraph graph, BoundModel model, {LinearImpl? linearImpl})
    : _graph = graph,
      _lin = linearImpl ?? linear,
      _mtpModules = _decodeMtp(model, graph),
      _embedTokens = model.embedTokens.toFloat64List(),
      _lmHead = model.lmHead.toFloat64List(),
      _finalNormGamma = model.finalNorm.toFloat64List(),
      _layerWeights = List<_LayerWeights>.unmodifiable([
        for (final bl in model.layers)
          _LayerWeights(
            inputNormGamma: bl.inputNorm.toFloat64List(),
            qProj: bl.qProj.toFloat64List(),
            kProj: bl.kProj.toFloat64List(),
            vProj: bl.vProj.toFloat64List(),
            oProj: bl.oProj.toFloat64List(),
            postAttnNormGamma: bl.postAttnNorm.toFloat64List(),
            gate: bl.gate?.toFloat64List(),
            up: bl.up?.toFloat64List(),
            down: bl.down?.toFloat64List(),
            moe: bl.moe == null
                ? null
                : _MoeW(
                    router: bl.moe!.router.toFloat64List(),
                    experts: [
                      for (final e in bl.moe!.experts)
                        MoeExpert(
                          gate: e.gate.toFloat64List(),
                          up: e.up.toFloat64List(),
                          down: e.down.toFloat64List(),
                          moeInter:
                              e.gate.shape[0], // gate is [moeInter x hidden]
                        ),
                    ],
                  ),
            qBias: bl.qBias?.toFloat64List(),
            kBias: bl.kBias?.toFloat64List(),
            vBias: bl.vBias?.toFloat64List(),
          ),
      ]);

  // Public API

  /// Runs a full forward pass over [tokenIds] and returns logits (length ==
  /// [ModelGraph.vocabSize]) for the LAST token position.
  ///
  /// Throws [ArgumentError] if [tokenIds] is empty.
  List<double> forward(List<int> tokenIds) {
    final hiddens = _hiddenStates(tokenIds);
    return _logitsAt(hiddens[hiddens.length - 1]);
  }

  /// Logits for EVERY position (teacher-forced): final norm + lm_head per hidden.
  List<List<double>> forwardAll(List<int> tokenIds) {
    final hiddens = _hiddenStates(tokenIds);
    return [for (final h in hiddens) _logitsAt(h)];
  }

  /// Multi-Token Prediction draft logits: the main model's next-token logits,
  /// then one logit set per MTP module (each predicting one token further).
  /// Returns `numModules + 1` logit sets. Throws if the model has no MTP heads.
  List<List<double>> mtpDraftLogits(List<int> tokenIds) {
    final mods = _mtpModules;
    if (mods == null) {
      throw StateError('mtpDraftLogits: model has no MTP heads');
    }
    final h = _graph.hiddenSize;
    final eps = _graph.layers.last.normEps;
    var hidden = _hiddenStates(tokenIds).last; // main last hidden (pre-norm)
    final logits = <List<double>>[_logitsAt(hidden)];
    var tok = _argmax(logits.first);
    for (final mw in mods) {
      final row = tok * h;
      final embedRow = Float64List.fromList(_embedTokens.sublist(row, row + h));
      hidden = mtpModuleForward(hidden, embedRow, mw, h, eps, lin: _lin);
      final lg = _logitsAt(hidden);
      logits.add(lg);
      tok = _argmax(lg);
    }
    return logits;
  }

  /// The greedy MTP draft: `numModules + 1` token ids (the main model's next
  /// token, then one per MTP module).
  List<int> mtpDraft(List<int> tokenIds) => [
    for (final lg in mtpDraftLogits(tokenIds)) _argmax(lg),
  ];

  /// Vision-language forward: runs the LLM over [tokenIds] with each
  /// [imageTokenIndex] placeholder replaced by the next already-projected vision
  /// embedding (text-space, `hiddenSize`-length). Returns the last position's
  /// logits. The number of placeholders must equal [visionEmbeds].length.
  List<double> forwardWithVision(
    List<int> tokenIds,
    List<Float64List> visionEmbeds,
    int imageTokenIndex,
  ) {
    final placeholders = tokenIds.where((t) => t == imageTokenIndex).length;
    if (placeholders != visionEmbeds.length) {
      throw ArgumentError(
        'image placeholder count ($placeholders) != visionEmbeds '
        '(${visionEmbeds.length})',
      );
    }
    final H = _graph.hiddenSize;
    var next = 0;
    final embeds = <Float64List>[
      for (final t in tokenIds)
        if (t == imageTokenIndex)
          () {
            final v = visionEmbeds[next++];
            if (v.length != H) {
              throw ArgumentError(
                'vision embed length ${v.length} != hidden $H',
              );
            }
            return Float64List.fromList(v);
          }()
        else
          _embedRow(t),
    ];
    final hiddens = _hiddenStatesFromEmbeds(embeds);
    return _logitsAt(hiddens[hiddens.length - 1]);
  }

  List<double> _logitsAt(Float64List hidden) {
    final finalEps = _graph.layers.last.normEps;
    final f = rmsNorm(hidden, _finalNormGamma, finalEps);
    return _lin(_lmHead, _graph.vocabSize, _graph.hiddenSize, f).toList();
  }

  /// A copy of the embedding table row for [tokenId] (copied so residuals do
  /// not corrupt the cached table).
  Float64List _embedRow(int tokenId) {
    final H = _graph.hiddenSize;
    final row = tokenId * H;
    return Float64List.fromList(_embedTokens.sublist(row, row + H));
  }

  /// Builds hidden states from token embeddings and runs all transformer layers.
  ///
  /// Throws [ArgumentError] if [tokenIds] is empty.
  List<Float64List> _hiddenStates(List<int> tokenIds) {
    if (tokenIds.isEmpty) {
      throw ArgumentError.value(tokenIds, 'tokenIds', 'must be non-empty');
    }
    return _hiddenStatesFromEmbeds([for (final t in tokenIds) _embedRow(t)]);
  }

  /// Runs all transformer layers over pre-built input embeddings [embeds] (one
  /// `hiddenSize`-length row per position). This is the shared core of the
  /// text and vision-language forward paths; [forwardWithVision] uses it after
  /// splicing projected image embeddings into the token embedding sequence.
  List<Float64List> _hiddenStatesFromEmbeds(List<Float64List> embeds) {
    if (embeds.isEmpty) {
      throw ArgumentError.value(embeds, 'embeds', 'must be non-empty');
    }

    final H = _graph.hiddenSize;
    final T = embeds.length;

    // Copy so residuals do not mutate a caller's embedding rows.
    final hiddens = [for (final e in embeds) Float64List.fromList(e)];

    // Transformer layers.
    for (var l = 0; l < _graph.layers.length; l++) {
      final spec = _graph.layers[l];
      final w = _layerWeights[l];
      final attn = spec.attention;
      final mlp = spec.mlp;
      final nH = attn.numHeads;
      final nKV = attn.numKvHeads;
      final hd = attn.headDim;
      final iSize = mlp.intermediateSize;
      final theta = attn.ropeTheta;
      final normEps = spec.normEps;

      final qRows = <Float64List>[];
      final kRows = <Float64List>[];
      final vRows = <Float64List>[];

      for (var t = 0; t < T; t++) {
        final n = rmsNorm(hiddens[t], w.inputNormGamma, normEps);
        final qFull = _lin(w.qProj, nH * hd, H, n);
        final kFull = _lin(w.kProj, nKV * hd, H, n);
        final vFull = _lin(w.vProj, nKV * hd, H, n);

        // Qwen2 q/k/v bias, added to the projection before RoPE.
        _addBias(qFull, w.qBias);
        _addBias(kFull, w.kBias);
        _addBias(vFull, w.vBias);

        // Apply RoPE per head - mutate in place using a copy-modify-writeback.
        for (var h = 0; h < nH; h++) {
          final offset = h * hd;
          final headSlice = Float64List.fromList(
            qFull.sublist(offset, offset + hd),
          );
          applyRopeHead(headSlice, t, theta);
          qFull.setRange(offset, offset + hd, headSlice);
        }
        for (var h = 0; h < nKV; h++) {
          final offset = h * hd;
          final headSlice = Float64List.fromList(
            kFull.sublist(offset, offset + hd),
          );
          applyRopeHead(headSlice, t, theta);
          kFull.setRange(offset, offset + hd, headSlice);
        }

        qRows.add(qFull);
        kRows.add(kFull);
        vRows.add(vFull);
      }

      final attnOut = causalGqaAttention(qRows, kRows, vRows, nH, nKV, hd);

      for (var t = 0; t < T; t++) {
        final o = _lin(w.oProj, H, nH * hd, attnOut[t]);
        addInPlace(hiddens[t], o);
      }

      final moeSpec = mlp.moe;
      if (moeSpec != null) {
        final mw = w.moe!;
        for (var t = 0; t < T; t++) {
          final n2 = rmsNorm(hiddens[t], w.postAttnNormGamma, normEps);
          final d = moeMlp(
            n2,
            H,
            mw.router,
            moeSpec.numExperts,
            mw.experts,
            moeSpec.topK,
            normTopK: moeSpec.normTopK,
          );
          addInPlace(hiddens[t], d);
        }
      } else {
        if (!mlp.gated) {
          throw ArgumentError(
            'GoldenRunner only supports gated MLPs (gated=true); '
            'layer $l has gated=false',
          );
        }
        for (var t = 0; t < T; t++) {
          final n2 = rmsNorm(hiddens[t], w.postAttnNormGamma, normEps);
          final g = _lin(w.gate!, iSize, H, n2);
          final u = _lin(w.up!, iSize, H, n2);
          final act = silu(g);
          final inter = mul(act, u);
          final d = _lin(w.down!, H, iSize, inter);
          addInPlace(hiddens[t], d);
        }
      }
    }

    return hiddens;
  }

  /// Greedy autoregressive generation. Returns exactly [maxNewTokens] newly
  /// generated token ids (not including the prompt).
  List<int> generate(List<int> promptTokens, {required int maxNewTokens}) {
    final current = List<int>.from(promptTokens);
    final generated = <int>[];
    for (var i = 0; i < maxNewTokens; i++) {
      final logits = forward(current);
      final next = _argmax(logits);
      current.add(next);
      generated.add(next);
    }
    return generated;
  }

  /// MTP speculative decoding. Each round the MTP heads draft `numModules + 1`
  /// tokens, a SINGLE verify forward over the extended sequence teacher-forces
  /// them, and the longest correct prefix is accepted plus one bonus token from
  /// the verifier. The output is IDENTICAL to [generate] (every kept token is
  /// either a draft that matched greedy verification or the greedy token itself)
  /// but uses fewer sequential forwards. Requires MTP heads.
  ///
  /// [onRound] reports `(accepted, drafted)` per verify forward: `accepted` is
  /// how many tokens were committed (>= 2: at least one draft + the bonus),
  /// `drafted` is the draft length. Averaging accepted/round gives the speedup.
  List<int> generateSpeculative(
    List<int> promptTokens, {
    required int maxNewTokens,
    void Function(int accepted, int drafted)? onRound,
  }) {
    if (_mtpModules == null) {
      throw StateError('generateSpeculative: model has no MTP heads');
    }
    final current = List<int>.from(promptTokens);
    final generated = <int>[];
    while (generated.length < maxNewTokens) {
      final draft = mtpDraft(current); // numModules+1 tokens, draft[0] = greedy
      final k = draft.length;
      final extended = [...current, ...draft];
      final logits = forwardAll(extended);
      final l = current.length;

      // Accept the longest leading run of drafts that match the verifier, then
      // one bonus token (the verifier's correct token at the first mismatch).
      var p = 0;
      while (p < k && _argmax(logits[l - 1 + p]) == draft[p]) {
        p++;
      }
      final bonus = _argmax(logits[l - 1 + p]);
      final accept = [...draft.sublist(0, p), bonus];

      var committed = 0;
      for (final t in accept) {
        if (generated.length >= maxNewTokens) break;
        current.add(t);
        generated.add(t);
        committed++;
      }
      onRound?.call(committed, k);
    }
    return generated;
  }

  // Private helpers

  static int _argmax(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }
}

// Decodes the model's MTP module weights (or null). Each module's transformer
// block reuses the main model's attention/MLP shape and requires a dense
// (non-MoE) block.
List<MtpModuleWeights>? _decodeMtp(BoundModel model, ModelGraph graph) {
  final mtp = model.mtp;
  if (mtp == null) return null;
  final base = graph.layers.first;
  final attn = base.attention;
  return [
    for (final mod in mtp.modules)
      MtpModuleWeights(
        enorm: mod.enorm.toFloat64List(),
        hnorm: mod.hnorm.toFloat64List(),
        ehProj: mod.ehProj.toFloat64List(),
        inputNorm: mod.block.inputNorm.toFloat64List(),
        qProj: mod.block.qProj.toFloat64List(),
        kProj: mod.block.kProj.toFloat64List(),
        vProj: mod.block.vProj.toFloat64List(),
        oProj: mod.block.oProj.toFloat64List(),
        postNorm: mod.block.postAttnNorm.toFloat64List(),
        gate: mod.block.gate!.toFloat64List(),
        up: mod.block.up!.toFloat64List(),
        down: mod.block.down!.toFloat64List(),
        numHeads: attn.numHeads,
        numKvHeads: attn.numKvHeads,
        headDim: attn.headDim,
        intermediate: base.mlp.intermediateSize,
      ),
  ];
}

// Adds [bias] into [v] elementwise, a no-op when [bias] is null.
void _addBias(Float64List v, Float64List? bias) {
  if (bias == null) return;
  for (var i = 0; i < v.length; i++) {
    v[i] += bias[i];
  }
}

// Internal weight cache for one layer.
class _LayerWeights {
  final Float64List inputNormGamma;
  final Float64List qProj;
  final Float64List kProj;
  final Float64List vProj;
  final Float64List oProj;
  final Float64List postAttnNormGamma;

  /// Dense gated-FFN weights (null for MoE layers, which use [moe]).
  final Float64List? gate;
  final Float64List? up;
  final Float64List? down;

  /// Decoded MoE weights (router + experts) for a MoE layer, else null.
  final _MoeW? moe;

  final Float64List? qBias;
  final Float64List? kBias;
  final Float64List? vBias;

  _LayerWeights({
    required this.inputNormGamma,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.postAttnNormGamma,
    this.gate,
    this.up,
    this.down,
    this.moe,
    this.qBias,
    this.kBias,
    this.vBias,
  });
}

/// Decoded Mixture-of-Experts weights for one layer.
class _MoeW {
  final Float64List router; // [numExperts x hidden]
  final List<MoeExpert> experts;
  const _MoeW({required this.router, required this.experts});
}
