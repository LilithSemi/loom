import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

Float64List _f(List<double> v) => Float64List.fromList(v);
Float64List _iota(int n, {double scale = 1.0}) =>
    Float64List.fromList([for (var i = 0; i < n; i++) i * scale]);

// A tiny ViT: hidden 4, patch 2, image 4 (so 4 patches), 1 channel, 2 heads
// headDim 2, intermediate 8, 1 block, no class token, no pre-LN. [identityEmbed]
// makes patch_embed the identity with zero bias/pos so visionEmbed == patches.
VisionWeights _tinyVision({bool identityEmbed = false}) {
  const hidden = 4, patchLen = 4, inter = 8;
  Float64List ln1() => _f([1, 1, 1, 1]);
  Float64List zeros(int n) => Float64List(n);
  final patchEmbed = identityEmbed
      ? _f([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) // 4x4 identity
      : _f([for (var i = 0; i < hidden * patchLen; i++) 0.02 * (i % 7 - 3)]);
  final block = VisionBlockWeights(
    ln1Gamma: ln1(),
    ln1Beta: zeros(hidden),
    qProj: _f([for (var i = 0; i < hidden * hidden; i++) 0.03 * (i % 5 - 2)]),
    qBias: zeros(hidden),
    kProj: _f([for (var i = 0; i < hidden * hidden; i++) 0.02 * (i % 4 - 1)]),
    kBias: zeros(hidden),
    vProj: _f([for (var i = 0; i < hidden * hidden; i++) 0.04 * (i % 6 - 2)]),
    vBias: zeros(hidden),
    oProj: _f([for (var i = 0; i < hidden * hidden; i++) 0.01 * (i % 3 - 1)]),
    oBias: zeros(hidden),
    ln2Gamma: ln1(),
    ln2Beta: zeros(hidden),
    fc1: _f([for (var i = 0; i < inter * hidden; i++) 0.02 * (i % 5 - 2)]),
    fc1Bias: zeros(inter),
    fc2: _f([for (var i = 0; i < hidden * inter; i++) 0.02 * (i % 4 - 1)]),
    fc2Bias: zeros(hidden),
  );
  return VisionWeights(
    patchEmbed: patchEmbed,
    patchEmbedBias: zeros(hidden),
    classToken: null,
    posEmbed: zeros(4 * hidden), // seqLen(4) x hidden, all zero
    preLnGamma: null,
    preLnBeta: null,
    blocks: [block],
    postLnGamma: ln1(),
    postLnBeta: zeros(hidden),
    hidden: hidden,
    numHeads: 2,
    headDim: 2,
    intermediate: inter,
    patchSize: 2,
    numChannels: 1,
    imageSize: 4,
    lnEps: 1e-5,
    hasClassToken: false,
  );
}

void main() {
  test('layerNorm normalizes to zero-mean/unit-var then scales+shifts', () {
    final y = layerNorm(
      _f([1, 2, 3, 4]),
      _f([1, 1, 1, 1]),
      _f([0, 0, 0, 0]),
      0.0,
    );
    // mean 2.5, var 1.25, inv = 1/sqrt(1.25).
    final inv = 1.0 / 1.1180339887498949;
    expect(y[0], closeTo(-1.5 * inv, 1e-12));
    expect(y[3], closeTo(1.5 * inv, 1e-12));
    var sum = 0.0;
    for (final v in y) sum += v;
    expect(sum, closeTo(0.0, 1e-12));
  });

  test('gelu(0)=0, matches the tanh approximation, saturates to identity', () {
    final y = gelu(_f([0, 1, -1, 30]));
    expect(y[0], closeTo(0.0, 1e-12));
    expect(y[1], closeTo(0.8411919906082768, 1e-9)); // gelu_tanh(1)
    expect(y[2], closeTo(-0.15880800939172324, 1e-9));
    expect(y[3], closeTo(30.0, 1e-6));
  });

  test(
    'patchify splits a channels-first image into (c,row,col) patch vectors',
    () {
      final w = _tinyVision();
      // 1x4x4 image with values 0..15 row-major.
      final patches = patchify(_iota(16), w);
      expect(patches.length, 4);
      expect(patches[0], _f([0, 1, 4, 5])); // top-left 2x2
      expect(patches[1], _f([2, 3, 6, 7])); // top-right
      expect(patches[2], _f([8, 9, 12, 13])); // bottom-left
      expect(patches[3], _f([10, 11, 14, 15])); // bottom-right
    },
  );

  test('visionEmbed with an identity patch-embed reproduces the patches', () {
    final w = _tinyVision(identityEmbed: true);
    final rows = visionEmbed(_iota(16), w);
    expect(rows.length, 4); // no class token
    expect(rows[0], _f([0, 1, 4, 5]));
    expect(rows[3], _f([10, 11, 14, 15]));
  });

  test('bidirectional attention: identical values collapse to that value', () {
    // v all equal -> any convex attention weighting returns that vector.
    const t = 3, heads = 2, hd = 2;
    final v = [
      for (var i = 0; i < t; i++) _f([5, -3, 5, -3]),
    ];
    final q = [
      for (var i = 0; i < t; i++) _f([0.1 * i, 0.2, -0.1, 0.3 * i]),
    ];
    final k = [
      for (var i = 0; i < t; i++) _f([0.2, -0.1 * i, 0.4, 0.1]),
    ];
    final out = bidirectionalAttention(q, k, v, heads, hd);
    for (final row in out) {
      expect(row[0], closeTo(5.0, 1e-12));
      expect(row[1], closeTo(-3.0, 1e-12));
      expect(row[3], closeTo(-3.0, 1e-12));
    }
  });

  test('encodeImage returns seqLen x hidden, finite and deterministic', () {
    final w = _tinyVision();
    final a = encodeImage(_iota(16, scale: 0.1), w);
    final b = encodeImage(_iota(16, scale: 0.1), w);
    expect(a.length, w.seqLen); // 4 patches, no class token
    for (final row in a) {
      expect(row.length, w.hidden);
      for (final x in row) {
        expect(x.isFinite, isTrue);
      }
    }
    for (var i = 0; i < a.length; i++) {
      expect(a[i], orderedEquals(b[i]));
    }
  });
}
