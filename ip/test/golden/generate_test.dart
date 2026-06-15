import 'package:test/test.dart';
import 'package:loom/loom.dart';

/// Builds a one-hot logit vector that makes argmax pick [token].
List<double> _oneHot(int token, {int vocab = 8}) {
  final v = List<double>.filled(vocab, 0.0);
  v[token] = 1.0;
  return v;
}

void main() {
  group('greedyGenerate', () {
    test('emits the scripted greedy sequence', () {
      // Fake model: next token = (last token + 1), wrapping under vocab 8.
      List<double> forward(List<int> toks) => _oneHot((toks.last + 1) % 8);

      final out = greedyGenerate(forward, [0], maxNewTokens: 3);
      expect(out, [1, 2, 3]);
    });

    test('stops early when a stop token is produced', () {
      // Sequence would be 1,2,3,4 but 3 is a stop token.
      List<double> forward(List<int> toks) => _oneHot((toks.last + 1) % 8);

      final out = greedyGenerate(
        forward,
        [0],
        maxNewTokens: 10,
        stopTokens: {3},
      );
      // 3 is included as the final (stop) token, generation halts there.
      expect(out, [1, 2, 3]);
    });

    test('streams each token through the callback as it is produced', () {
      List<double> forward(List<int> toks) => _oneHot((toks.last + 1) % 8);
      final seen = <int>[];

      greedyGenerate(forward, [0], maxNewTokens: 3, onToken: seen.add);
      expect(seen, [1, 2, 3]);
    });

    test('feeds generated tokens back into the context', () {
      final contexts = <List<int>>[];
      List<double> forward(List<int> toks) {
        contexts.add(List<int>.from(toks));
        return _oneHot((toks.last + 1) % 8);
      }

      greedyGenerate(forward, [5], maxNewTokens: 2);
      expect(contexts, [
        [5],
        [5, 6],
      ]);
    });
  });
}
