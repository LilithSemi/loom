import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// A tiny llama-style model (hidden 8, 2 layers, 2 heads headDim 4, inter 16,
// vocab 10) with 2 MTP modules at layer indices 2 and 3. [ehSeed] offsets the
// eh_proj fill so two stores differ only in the MTP fusion projection.
Uint8List _mtpSafetensors({double ehSeed = 0.0}) {
  const h = 8, inter = 16, layers = 2, mtp = 2;
  final shapes = <String, List<int>>{
    'model.embed_tokens.weight': [10, h],
    'model.norm.weight': [h],
  };
  void block(int i) {
    shapes['model.layers.$i.input_layernorm.weight'] = [h];
    shapes['model.layers.$i.self_attn.q_proj.weight'] = [h, h];
    shapes['model.layers.$i.self_attn.k_proj.weight'] = [h, h];
    shapes['model.layers.$i.self_attn.v_proj.weight'] = [h, h];
    shapes['model.layers.$i.self_attn.o_proj.weight'] = [h, h];
    shapes['model.layers.$i.post_attention_layernorm.weight'] = [h];
    shapes['model.layers.$i.mlp.gate_proj.weight'] = [inter, h];
    shapes['model.layers.$i.mlp.up_proj.weight'] = [inter, h];
    shapes['model.layers.$i.mlp.down_proj.weight'] = [h, inter];
  }

  for (var i = 0; i < layers; i++) {
    block(i);
  }
  final ehNames = <String>{};
  for (var m = 0; m < mtp; m++) {
    final l = layers + m;
    block(l);
    shapes['model.layers.$l.enorm.weight'] = [h];
    shapes['model.layers.$l.hnorm.weight'] = [h];
    shapes['model.layers.$l.eh_proj.weight'] = [h, 2 * h];
    ehNames.add('model.layers.$l.eh_proj.weight');
  }

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = shapes.keys.toList()..sort();
  for (final n in names) {
    final shape = shapes[n]!;
    final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    final bias = ehNames.contains(n) ? ehSeed : 0.0;
    for (var i = 0; i < count; i++) {
      bd.setFloat32(i * 4, 0.02 * (i % 5 - 2) + bias, Endian.little);
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
  final out = BytesBuilder();
  final len = ByteData(8)..setUint64(0, hb.length, Endian.little);
  out.add(len.buffer.asUint8List());
  out.add(hb);
  out.add(data.toBytes());
  return out.toBytes();
}

ModelGraph _mtpGraph() => parseHfConfig({
  'model_type': 'llama',
  'hidden_size': 8,
  'num_hidden_layers': 2,
  'num_attention_heads': 2,
  'num_key_value_heads': 2,
  'intermediate_size': 16,
  'vocab_size': 10,
  'max_position_embeddings': 32,
  'num_nextn_predict_layers': 2,
  'tie_word_embeddings': true,
}, name: 'tiny-mtp');

void main() {
  test('parseHfConfig builds an MtpSpec from num_nextn_predict_layers', () {
    final mtp = _mtpGraph().mtp;
    expect(mtp, isNotNull);
    expect(mtp!.numModules, 2);
    // A model without the key has no MTP.
    final plain = parseHfConfig({
      'model_type': 'llama',
      'hidden_size': 8,
      'num_hidden_layers': 1,
      'num_attention_heads': 2,
      'intermediate_size': 16,
      'vocab_size': 10,
    });
    expect(plain.mtp, isNull);
  });

  test(
    'bindWeights resolves every MTP module (block + enorm/hnorm/eh_proj)',
    () {
      final model = bindWeights(
        _mtpGraph(),
        SafetensorsStore.parse(_mtpSafetensors()),
      );
      expect(model.mtp, isNotNull);
      expect(model.mtp!.modules.length, 2);
      final m0 = model.mtp!.modules.first;
      expect(m0.enorm.shape, [8]);
      expect(m0.hnorm.shape, [8]);
      expect(m0.ehProj.shape, [8, 16]);
      expect(m0.block.qProj.shape, [8, 8]);
      expect(m0.block.down!.shape, [8, 16]);
    },
  );

  test('mtpDraft: draft[0] equals the main model next-token prediction', () {
    final g = _mtpGraph();
    final runner = GoldenRunner(
      g,
      bindWeights(g, SafetensorsStore.parse(_mtpSafetensors())),
    );
    final tokens = [1, 2, 3];
    final draftLogits = runner.mtpDraftLogits(tokens);
    expect(draftLogits.length, 3); // main + 2 modules
    // The first drafted logit set is exactly the main forward's logits.
    expect(draftLogits.first, orderedEquals(runner.forward(tokens)));
    // Every drafted token is a valid id, and there are numModules+1 of them.
    final draft = runner.mtpDraft(tokens);
    expect(draft.length, 3);
    for (final t in draft) {
      expect(t, inInclusiveRange(0, 9));
    }
  });

  test(
    'speculative decoding output is identical to greedy (fewer forwards)',
    () {
      final g = _mtpGraph();
      final runner = GoldenRunner(
        g,
        bindWeights(g, SafetensorsStore.parse(_mtpSafetensors())),
      );
      final prompt = [1, 2, 3];
      const n = 20;
      final greedy = runner.generate(prompt, maxNewTokens: n);

      var rounds = 0;
      var accepted = 0;
      final spec = runner.generateSpeculative(
        prompt,
        maxNewTokens: n,
        onRound: (a, d) {
          rounds++;
          accepted += a;
        },
      );

      // The whole point: bit-identical to greedy, regardless of draft quality.
      expect(spec, orderedEquals(greedy));
      expect(spec.length, n);
      expect(accepted, n); // every committed token counted
      // Each verify forward commits >= 2 tokens (one draft + the bonus), so the
      // number of forwards is strictly fewer than the token count.
      expect(rounds, lessThan(n));
    },
  );

  test('generateSpeculative throws when the model has no MTP heads', () {
    final g = parseHfConfig({
      'model_type': 'llama',
      'hidden_size': 8,
      'num_hidden_layers': 2,
      'num_attention_heads': 2,
      'num_key_value_heads': 2,
      'intermediate_size': 16,
      'vocab_size': 10,
      'max_position_embeddings': 32,
      'tie_word_embeddings': true,
    });
    // Reuse the MTP safetensors (a superset) so binding succeeds. The graph just
    // has no MTP spec.
    final runner = GoldenRunner(
      g,
      bindWeights(g, SafetensorsStore.parse(_mtpSafetensors())),
    );
    expect(
      () => runner.generateSpeculative([1, 2], maxNewTokens: 4),
      throwsStateError,
    );
  });

  test('the MTP modules participate: eh_proj changes drafts past index 0', () {
    final g = _mtpGraph();
    final a = GoldenRunner(
      g,
      bindWeights(g, SafetensorsStore.parse(_mtpSafetensors())),
    );
    final b = GoldenRunner(
      g,
      bindWeights(g, SafetensorsStore.parse(_mtpSafetensors(ehSeed: 0.5))),
    );
    final tokens = [4, 5, 6];
    final da = a.mtpDraftLogits(tokens);
    final db = b.mtpDraftLogits(tokens);
    // draft[0] (main model) is unaffected by the MTP eh_proj.
    expect(da.first, orderedEquals(db.first));
    // draft[1] (first MTP module) IS affected.
    var maxDiff = 0.0;
    for (var i = 0; i < da[1].length; i++) {
      final d = (da[1][i] - db[1][i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    expect(
      maxDiff,
      greaterThan(1e-6),
      reason: 'eh_proj had no effect on the MTP draft',
    );
  });
}
