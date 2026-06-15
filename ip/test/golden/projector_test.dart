import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

Float64List _f(List<double> v) => Float64List.fromList(v);
Float64List _id(int n) => Float64List.fromList([
  for (var r = 0; r < n; r++)
    for (var c = 0; c < n; c++) r == c ? 1.0 : 0.0,
]);

void main() {
  test('single-linear projector applies weight + bias', () {
    // out = linear1 @ v + bias1; linear1 identity [3x3], bias = [1,2,3].
    final p = ProjectorWeights(
      linear1: _id(3),
      bias1: _f([1, 2, 3]),
      linear2: null,
      bias2: null,
      inputDim: 3,
      hiddenDim: 3,
      outputDim: 3,
    );
    final out = projectOne(_f([10, 20, 30]), p);
    expect(out, _f([11, 22, 33]));
  });

  test('two-layer mlp2x_gelu projector applies linear->gelu->linear', () {
    // linear1 = identity, linear2 = identity, no biases -> out = gelu(v).
    final p = ProjectorWeights(
      linear1: _id(3),
      bias1: null,
      linear2: _id(3),
      bias2: null,
      inputDim: 3,
      hiddenDim: 3,
      outputDim: 3,
    );
    final v = _f([0, 1, -1]);
    final out = projectOne(v, p);
    final ref = gelu(v);
    for (var i = 0; i < 3; i++) {
      expect(out[i], closeTo(ref[i], 1e-12));
    }
    expect(p.isTwoLayer, isTrue);
  });

  test(
    'bindProjector resolves LLaVA linear_1/linear_2 and projects to text dim',
    () {
      const visionH = 4, textH = 6;
      final shapes = <String, List<int>>{
        'multi_modal_projector.linear_1.weight': [textH, visionH],
        'multi_modal_projector.linear_1.bias': [textH],
        'multi_modal_projector.linear_2.weight': [textH, textH],
        'multi_modal_projector.linear_2.bias': [textH],
      };
      final data = BytesBuilder();
      final header = <String, dynamic>{};
      var off = 0;
      for (final n in shapes.keys.toList()..sort()) {
        final shape = shapes[n]!;
        final count = shape.reduce((a, b) => a * b);
        final bd = ByteData(count * 4);
        for (var i = 0; i < count; i++) {
          bd.setFloat32(i * 4, 0.05 * (i % 5 - 2), Endian.little);
        }
        data.add(bd.buffer.asUint8List());
        header[n] = {
          'dtype': 'F32',
          'shape': shape,
          'data_offsets': [off, off + count * 4],
        };
        off += count * 4;
      }
      final hb = utf8.encode(jsonEncode(header));
      final st = BytesBuilder()
        ..add(
          (ByteData(
            8,
          )..setUint64(0, hb.length, Endian.little)).buffer.asUint8List(),
        )
        ..add(hb)
        ..add(data.toBytes());

      final store = SafetensorsStore.parse(st.toBytes());
      const spec = ProjectorSpec(numLayers: 2, activation: ActivationKind.gelu);
      final p = bindProjector(spec, store, visionH, textH);
      expect(p.isTwoLayer, isTrue);
      expect(p.inputDim, visionH);
      expect(p.outputDim, textH);
      final out = projectOne(_f([0.1, -0.2, 0.3, 0.4]), p);
      expect(out.length, textH);
      for (final x in out) {
        expect(x.isFinite, isTrue);
      }
    },
  );

  test('projectVision maps every embedding', () {
    final p = ProjectorWeights(
      linear1: _id(2),
      bias1: null,
      linear2: null,
      bias2: null,
      inputDim: 2,
      hiddenDim: 2,
      outputDim: 2,
    );
    final out = projectVision([
      _f([1, 2]),
      _f([3, 4]),
    ], p);
    expect(out.length, 2);
    expect(out[0], _f([1, 2]));
    expect(out[1], _f([3, 4]));
  });
}
