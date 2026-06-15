import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

Float64List _rnd(math.Random r, int n) =>
    Float64List.fromList([for (var i = 0; i < n; i++) r.nextDouble() * 2 - 1]);

MoeExpert _expert(math.Random r, int hidden, int inter) => MoeExpert(
  gate: _rnd(r, inter * hidden),
  up: _rnd(r, inter * hidden),
  down: _rnd(r, hidden * inter),
  moeInter: inter,
);

void main() {
  test(
    'normTopK combine makes identical experts topK-invariant (weights sum to 1)',
    () {
      final r = math.Random(1);
      const hidden = 4, inter = 6, n = 4;
      final e = _expert(r, hidden, inter);
      final experts = List.filled(n, e); // all experts identical
      final router = _rnd(r, n * hidden);
      final x = _rnd(r, hidden);

      final k1 = moeMlp(x, hidden, router, n, experts, 1);
      final k2 = moeMlp(x, hidden, router, n, experts, 2);
      final k4 = moeMlp(x, hidden, router, n, experts, 4);
      for (var i = 0; i < hidden; i++) {
        expect(k2[i], closeTo(k1[i], 1e-12), reason: 'topK=2 vs 1 at $i');
        expect(k4[i], closeTo(k1[i], 1e-12), reason: 'topK=4 vs 1 at $i');
      }
    },
  );

  test('only the top-k experts contribute (gating is exact)', () {
    const hidden = 2, inter = 3, n = 4;
    final r = math.Random(2);
    final experts = [for (var i = 0; i < n; i++) _expert(r, hidden, inter)];
    // Router logits(x) = n-e for x = [1,0], so experts rank 0>1>2>3; top-2 = {0,1}.
    final x = Float64List.fromList([1.0, 0.0]);
    final router = Float64List(n * hidden);
    for (var e = 0; e < n; e++) {
      router[e * hidden] = (n - e).toDouble();
    }
    final out1 = moeMlp(x, hidden, router, n, experts, 2);

    // Replacing a NON-selected expert (index 3) must not change the output.
    final experts2 = List<MoeExpert>.from(experts);
    experts2[3] = _expert(math.Random(999), hidden, inter);
    final out2 = moeMlp(x, hidden, router, n, experts2, 2);
    for (var i = 0; i < hidden; i++) {
      expect(
        out2[i],
        closeTo(out1[i], 1e-12),
        reason: 'non-selected leaked at $i',
      );
    }

    // Replacing a SELECTED expert (index 0) MUST change the output.
    final experts3 = List<MoeExpert>.from(experts);
    experts3[0] = _expert(math.Random(1234), hidden, inter);
    final out3 = moeMlp(x, hidden, router, n, experts3, 2);
    var changed = false;
    for (var i = 0; i < hidden; i++) {
      if ((out3[i] - out1[i]).abs() > 1e-9) changed = true;
    }
    expect(changed, isTrue, reason: 'changing a selected expert had no effect');
  });
}
