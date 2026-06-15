import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

void main() {
  group('applyRopeHead', () {
    test('pos=0 leaves [1,0] unchanged', () {
      final head = Float64List.fromList([1.0, 0.0]);
      applyRopeHead(head, 0, 10000.0);
      expect(head[0], closeTo(1.0, 1e-9));
      expect(head[1], closeTo(0.0, 1e-9));
    });

    test('pos=0 leaves [1,2] unchanged', () {
      final head = Float64List.fromList([1.0, 2.0]);
      applyRopeHead(head, 0, 10000.0);
      expect(head[0], closeTo(1.0, 1e-9));
      expect(head[1], closeTo(2.0, 1e-9));
    });

    test('pos=1, hd=2: [1,0] rotates to [cos(1), sin(1)]', () {
      final head = Float64List.fromList([1.0, 0.0]);
      applyRopeHead(head, 1, 10000.0);
      expect(head[0], closeTo(math.cos(1.0), 1e-7));
      expect(head[1], closeTo(math.sin(1.0), 1e-7));
    });

    test('odd-length head throws ArgumentError', () {
      final head = Float64List.fromList([1.0, 2.0, 3.0]);
      expect(() => applyRopeHead(head, 0, 10000.0), throwsArgumentError);
    });

    test(
      'pos=1, hd=4: [1,1,0,0] rotates correctly (half-split HF convention)',
      () {
        // half=2
        // j=0: invFreq=1/pow(10000, 0/4)=1, angle=1
        //   x1=head[0]=1, x2=head[2]=0
        //   head[0]=cos(1), head[2]=sin(1)
        // j=1: invFreq=1/pow(10000, 2/4)=1/100=0.01, angle=0.01
        //   x1=head[1]=1, x2=head[3]=0
        //   head[1]=cos(0.01), head[3]=sin(0.01)
        final head = Float64List.fromList([1.0, 1.0, 0.0, 0.0]);
        applyRopeHead(head, 1, 10000.0);
        expect(head[0], closeTo(math.cos(1.0), 1e-6));
        expect(head[1], closeTo(math.cos(0.01), 1e-6));
        expect(head[2], closeTo(math.sin(1.0), 1e-6));
        expect(head[3], closeTo(math.sin(0.01), 1e-6));
      },
    );
  });

  group('causalGqaAttention', () {
    test('T=2, MHA: q attends causally to correct values', () {
      // numHeads=1, numKvHeads=1, headDim=2
      // q=[[1,0],[0,1]], k=[[1,0],[0,1]], v=[[1,2],[3,4]]
      // pos=0: only s=0, score=dot([1,0],[1,0])/sqrt(2)=1/sqrt(2)
      //        softmax([1/sqrt(2)]) = [1.0]
      //        out = 1*[1,2] = [1,2]
      // pos=1: scores: s=0: dot([0,1],[1,0])/sqrt(2)=0
      //                s=1: dot([0,1],[0,1])/sqrt(2)=1/sqrt(2)
      //        softmax([0, 1/sqrt(2)])
      //        weights = [exp(0), exp(1/sqrt(2))] / (exp(0)+exp(1/sqrt(2)))
      //                = [1, exp(0.70710678)] / (1+exp(0.70710678))
      //                = [1, 2.02811498] / 3.02811498
      //                = [0.33024439, 0.66975561]
      //        out = 0.33024439*[1,2] + 0.66975561*[3,4]
      //            = [0.33024+2.00926, 0.66048+2.67902]
      //            = [2.33951, 3.33951]
      final q = [
        Float64List.fromList([1.0, 0.0]),
        Float64List.fromList([0.0, 1.0]),
      ];
      final k = [
        Float64List.fromList([1.0, 0.0]),
        Float64List.fromList([0.0, 1.0]),
      ];
      final v = [
        Float64List.fromList([1.0, 2.0]),
        Float64List.fromList([3.0, 4.0]),
      ];
      final out = causalGqaAttention(q, k, v, 1, 1, 2);
      expect(out.length, 2);
      expect(out[0][0], closeTo(1.0, 1e-5));
      expect(out[0][1], closeTo(2.0, 1e-5));
      expect(out[1][0], closeTo(2.33951, 1e-4));
      expect(out[1][1], closeTo(3.33951, 1e-4));
    });

    test('T=1, GQA broadcast: 2 heads share 1 kv head', () {
      // numHeads=2, numKvHeads=1, headDim=2, group=2
      // q row=[1,0, 0,1]: head0=[1,0], head1=[0,1]
      // k row=[2,0], v row=[5,6]
      // T=1 so each head sees only s=0, softmax([score]) = [1.0]
      // head0 out = v[0] = [5,6]
      // head1 out = v[0] = [5,6]
      // output row = [5,6,5,6]
      final q = [
        Float64List.fromList([1.0, 0.0, 0.0, 1.0]),
      ];
      final k = [
        Float64List.fromList([2.0, 0.0]),
      ];
      final v = [
        Float64List.fromList([5.0, 6.0]),
      ];
      final out = causalGqaAttention(q, k, v, 2, 1, 2);
      expect(out.length, 1);
      expect(out[0][0], closeTo(5.0, 1e-9));
      expect(out[0][1], closeTo(6.0, 1e-9));
      expect(out[0][2], closeTo(5.0, 1e-9));
      expect(out[0][3], closeTo(6.0, 1e-9));
    });

    test('numHeads not divisible by numKvHeads throws ArgumentError', () {
      final q = [
        Float64List.fromList([1.0, 0.0, 0.0]),
      ];
      final k = [
        Float64List.fromList([1.0]),
      ];
      final v = [
        Float64List.fromList([1.0]),
      ];
      expect(() => causalGqaAttention(q, k, v, 3, 2, 1), throwsArgumentError);
    });

    test('mismatched q row length throws ArgumentError', () {
      // q rows should be numHeads*headDim = 2*2=4 but we pass length 3
      final q = [
        Float64List.fromList([1.0, 0.0, 0.0]),
      ];
      final k = [
        Float64List.fromList([1.0, 0.0]),
      ];
      final v = [
        Float64List.fromList([1.0, 0.0]),
      ];
      expect(() => causalGqaAttention(q, k, v, 2, 1, 2), throwsArgumentError);
    });

    test('mismatched kv row length throws ArgumentError', () {
      // k rows should be numKvHeads*headDim = 1*2=2 but we pass length 3
      final q = [
        Float64List.fromList([1.0, 0.0, 1.0, 0.0]),
      ];
      final k = [
        Float64List.fromList([1.0, 0.0, 0.0]),
      ];
      final v = [
        Float64List.fromList([1.0, 0.0]),
      ];
      expect(() => causalGqaAttention(q, k, v, 2, 1, 2), throwsArgumentError);
    });

    test('k and v lists different length T throws ArgumentError', () {
      final q = [
        Float64List.fromList([1.0, 0.0]),
      ];
      final k = [
        Float64List.fromList([1.0, 0.0]),
        Float64List.fromList([0.0, 1.0]),
      ];
      final v = [
        Float64List.fromList([1.0, 0.0]),
      ];
      expect(() => causalGqaAttention(q, k, v, 1, 1, 2), throwsArgumentError);
    });
  });
}
