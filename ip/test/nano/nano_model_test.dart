import 'package:loom/src/nano/nano_model.dart';
import 'package:test/test.dart';

/// Builds the context state from the last [bits] bits of [seq] ending at the
/// element just before index [end].
int ctxFrom(List<int> seq, int end, int bits) {
  var c = 0;
  for (var i = end - bits; i < end; i++) {
    c = (c << 1) | (seq[i] & 1);
  }
  return c;
}

void main() {
  group('NanoConfig', () {
    test('derives state count and mask from contextBits', () {
      const c = NanoConfig(contextBits: 3);
      expect(c.vocab, 2);
      expect(c.nStates, 8);
      expect(c.contextMask, 7);
    });
  });

  group('NanoModel training + greedy generation', () {
    test('reproduces an alternating bitstream', () {
      const cfg = NanoConfig(contextBits: 3);
      final target = List<int>.generate(64, (i) => i % 2); // 0 1 0 1 ...
      final m = NanoModel.train(cfg, target);

      // Start from context "010" (the bits before index 3 == 0,1,0).
      final start = ctxFrom(target, 3, 3);
      final gen = m.generate(start, 16);
      // Output keeps alternating (the learned bigram-of-bits behaviour).
      for (var i = 1; i < gen.length; i++) {
        expect(gen[i], isNot(equals(gen[i - 1])), reason: 'should alternate');
      }
    });

    test('reproduces a period-4 pattern (order-3 context suffices)', () {
      const cfg = NanoConfig(contextBits: 3);
      final pattern = [0, 0, 1, 1];
      final target = [for (var i = 0; i < 64; i++) pattern[i % 4]];
      final m = NanoModel.train(cfg, target);

      final start = ctxFrom(target, 3, 3); // bits 0,0,1 -> 0b001
      final gen = m.generate(start, 12);
      // After context 0,0,1 the next bits continue the cycle: 1,0,0,1,1,0,0,...
      final expected = [for (var i = 3; i < 3 + 12; i++) pattern[i % 4]];
      expect(gen, expected);
    });

    test('logits select the weight column for a context state', () {
      const cfg = NanoConfig(contextBits: 3);
      final target = [for (var i = 0; i < 64; i++) pattern4(i)];
      final m = NanoModel.train(cfg, target);
      for (var s = 0; s < cfg.nStates; s++) {
        expect(m.logits(s), [
          m.weights[0 * cfg.nStates + s],
          m.weights[1 * cfg.nStates + s],
        ]);
      }
    });

    test('weights fit int8 and matrix is [vocab x nStates]', () {
      const cfg = NanoConfig(contextBits: 3);
      final m = NanoModel.train(cfg, [for (var i = 0; i < 32; i++) i % 2]);
      expect(m.weights.length, cfg.vocab * cfg.nStates);
      for (final w in m.weights) {
        expect(w, inInclusiveRange(-128, 127));
      }
    });
  });
}

int pattern4(int i) => [0, 0, 1, 1][i % 4];
