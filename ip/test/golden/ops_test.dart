import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

void main() {
  group('linear', () {
    test('no bias: weight [1,2,3,4] outDim=2 inDim=2, x=[1,1] -> [3,7]', () {
      final w = Float64List.fromList([1, 2, 3, 4]);
      final x = Float64List.fromList([1, 1]);
      final y = linear(w, 2, 2, x);
      expect(y.length, 2);
      expect(y[0], closeTo(3.0, 1e-9));
      expect(y[1], closeTo(7.0, 1e-9));
    });

    test('with bias [10,20] -> [13,27]', () {
      final w = Float64List.fromList([1, 2, 3, 4]);
      final x = Float64List.fromList([1, 1]);
      final b = Float64List.fromList([10, 20]);
      final y = linear(w, 2, 2, x, bias: b);
      expect(y[0], closeTo(13.0, 1e-9));
      expect(y[1], closeTo(27.0, 1e-9));
    });

    test('throws if x.length != inDim', () {
      final w = Float64List.fromList([1, 2, 3, 4]);
      final x = Float64List.fromList([1]);
      expect(() => linear(w, 2, 2, x), throwsArgumentError);
    });

    test('throws if weight.length != outDim*inDim', () {
      final w = Float64List.fromList([1, 2, 3]);
      final x = Float64List.fromList([1, 1]);
      expect(() => linear(w, 2, 2, x), throwsArgumentError);
    });

    test('throws if bias.length != outDim', () {
      final w = Float64List.fromList([1, 2, 3, 4]);
      final x = Float64List.fromList([1, 1]);
      final b = Float64List.fromList([10]);
      expect(() => linear(w, 2, 2, x, bias: b), throwsArgumentError);
    });
  });

  group('rmsNorm', () {
    test('x=[1,2,3,4], gamma=[1,1,1,1], eps=0', () {
      final x = Float64List.fromList([1, 2, 3, 4]);
      final gamma = Float64List.fromList([1, 1, 1, 1]);
      final y = rmsNorm(x, gamma, 0.0);
      expect(y.length, 4);
      expect(y[0], closeTo(0.36514837, 1e-6));
      expect(y[1], closeTo(0.73029674, 1e-6));
      expect(y[2], closeTo(1.09544511, 1e-6));
      expect(y[3], closeTo(1.46059349, 1e-6));
    });

    test('throws if gamma.length != x.length', () {
      final x = Float64List.fromList([1, 2, 3]);
      final gamma = Float64List.fromList([1, 1]);
      expect(() => rmsNorm(x, gamma, 1e-5), throwsArgumentError);
    });
  });

  group('silu', () {
    test('silu([0,1,-1]) -> [0.0, 0.73105858, -0.26894142]', () {
      final x = Float64List.fromList([0.0, 1.0, -1.0]);
      final y = silu(x);
      expect(y.length, 3);
      expect(y[0], closeTo(0.0, 1e-6));
      expect(y[1], closeTo(0.73105858, 1e-6));
      expect(y[2], closeTo(-0.26894142, 1e-6));
    });
  });

  group('softmax', () {
    test('softmax([1,2,3]) -> [0.09003057, 0.24472847, 0.66524096]', () {
      final x = Float64List.fromList([1.0, 2.0, 3.0]);
      final y = softmax(x);
      expect(y.length, 3);
      expect(y[0], closeTo(0.09003057, 1e-6));
      expect(y[1], closeTo(0.24472847, 1e-6));
      expect(y[2], closeTo(0.66524096, 1e-6));
    });

    test('result sums to 1.0', () {
      final x = Float64List.fromList([1.0, 2.0, 3.0]);
      final y = softmax(x);
      final sum = y.fold(0.0, (a, b) => a + b);
      expect(sum, closeTo(1.0, 1e-9));
    });

    test(
      'stability: softmax([1000,1001,1002]) is finite and same as softmax([0,1,2])',
      () {
        final large = softmax(Float64List.fromList([1000.0, 1001.0, 1002.0]));
        final small = softmax(Float64List.fromList([0.0, 1.0, 2.0]));
        for (final v in large) {
          expect(v.isFinite, isTrue);
        }
        for (var i = 0; i < 3; i++) {
          expect(large[i], closeTo(small[i], 1e-9));
        }
      },
    );
  });

  group('addInPlace', () {
    test('a=[1,2,3], b=[10,20,30] -> a=[11,22,33]', () {
      final a = Float64List.fromList([1.0, 2.0, 3.0]);
      final b = Float64List.fromList([10.0, 20.0, 30.0]);
      addInPlace(a, b);
      expect(a[0], closeTo(11.0, 1e-9));
      expect(a[1], closeTo(22.0, 1e-9));
      expect(a[2], closeTo(33.0, 1e-9));
    });

    test('throws if lengths differ', () {
      final a = Float64List.fromList([1.0, 2.0]);
      final b = Float64List.fromList([1.0, 2.0, 3.0]);
      expect(() => addInPlace(a, b), throwsArgumentError);
    });
  });

  group('mul', () {
    test('mul([1,2,3],[4,5,6]) -> [4,10,18]', () {
      final a = Float64List.fromList([1.0, 2.0, 3.0]);
      final b = Float64List.fromList([4.0, 5.0, 6.0]);
      final y = mul(a, b);
      expect(y.length, 3);
      expect(y[0], closeTo(4.0, 1e-9));
      expect(y[1], closeTo(10.0, 1e-9));
      expect(y[2], closeTo(18.0, 1e-9));
    });

    test('throws if lengths differ', () {
      final a = Float64List.fromList([1.0, 2.0]);
      final b = Float64List.fromList([1.0, 2.0, 3.0]);
      expect(() => mul(a, b), throwsArgumentError);
    });
  });
}
