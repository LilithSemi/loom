import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('identical logits -> 0 KL', () {
    final a = [1.0, 2.0, 3.0, -1.0];
    expect(klDivergence(a, a), closeTo(0.0, 1e-12));
  });
  test('different logits -> positive KL', () {
    expect(klDivergence([2.0, 0.0, 0.0], [0.0, 0.0, 2.0]), greaterThan(0.1));
  });
  test('shift-invariant (softmax) -> 0 KL under constant offset', () {
    expect(
      klDivergence([1.0, 2.0, 3.0], [11.0, 12.0, 13.0]),
      closeTo(0.0, 1e-12),
    );
  });
}
