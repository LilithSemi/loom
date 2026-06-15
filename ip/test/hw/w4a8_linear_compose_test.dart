// W4A8 fp-linear composition test: LoomActQuant -> int matmul -> LoomDequant
// reproduces the golden fp64 linear within W4A8 tolerance.
//
// The integer matmul core (LoomStreamMatmul) is separately verified bit-exact
// (stream_matmul_int4_test), so matmulInt stands in for it here. This test
// isolates the new fp16<->int quant boundary units and proves they bracket the
// int matmul into a real linear, the unit the layer sequencer will instantiate.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/act_quant.dart';
import 'package:loom/src/hw/dequant.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('W4A8 fp linear matches golden fp64 linear within tolerance', () async {
    const rows = 4;
    const cols = 6;
    // Deterministic weights/activation.
    final w = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.21;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.37,
    ]);

    final goldenFp = linear(w, rows, cols, x);
    final goldenQ = quantizedLinearW4A8(w, rows, cols, x);

    // Weights quantized to int4 with per-row scales (precomputed at provisioning).
    final qm = quantizeRowwiseInt4(w, rows, cols);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final compute = Logic(name: 'compute');
    final aq = LoomActQuant();
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('x_en', xEn),
      ('x_in', xInL),
      ('compute', compute),
    ]) {
      aq.input(p).srcConnection! <= s;
    }

    final accL = Logic(name: 'acc', width: 32);
    final rowScaleL = Logic(name: 'row_scale', width: 16);
    final actScaleL = Logic(name: 'act_scale', width: 16);
    final dqValidIn = Logic(name: 'valid_in');
    final dq = LoomDequant();
    dq.input('clk').srcConnection! <= clk;
    dq.input('reset').srcConnection! <= reset;
    dq.input('valid_in').srcConnection! <= dqValidIn;
    dq.input('acc').srcConnection! <= accL;
    dq.input('row_scale').srcConnection! <= rowScaleL;
    dq.input('act_scale').srcConnection! <= actScaleL;

    await aq.build();
    await dq.build();

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [xEn, compute, dqValidIn]) {
      l.inject(0);
    }
    xInL.inject(0);
    accL.inject(0);
    rowScaleL.inject(0);
    actScaleL.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(3000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // MAX pass.
    for (var c = 0; c < cols; c++) {
      xEn.inject(1);
      xInL.inject(e(x[c]));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    compute.inject(1);
    await clk.nextPosedge;
    compute.inject(0);
    var guard = 0;
    while (!aq.output('ready').value.toBool() && guard++ < 200) {
      await clk.nextPosedge;
    }
    expect(aq.output('ready').value.toBool(), isTrue);
    final actScaleBits = aq.output('scale_out').value;

    // QUANT pass -> int8 activations. q_out/q_valid lag x_in (clocked quant
    // multiply), so collect on q_valid in order.
    final a8 = Int8List(cols);
    var nq = 0;
    final collectA8 = () async {
      while (nq < cols) {
        await clk.nextNegedge;
        if (aq.output('q_valid').value.toBool()) {
          var raw = aq.output('q_out').value.toInt();
          if (raw >= 128) raw -= 256;
          a8[nq++] = raw;
        }
      }
    }();
    for (var c = 0; c < cols; c++) {
      xEn.inject(1);
      xInL.inject(e(x[c]));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    await collectA8;

    // Integer matmul (stands in for the verified LoomStreamMatmul int4 core):
    // acc[r] = sum_c w4[r,c] * a8[c].
    final acc = Int32List(rows);
    for (var r = 0; r < rows; r++) {
      var s = 0;
      for (var c = 0; c < cols; c++) {
        s += qm.values[r * cols + c] * a8[c];
      }
      acc[r] = s;
    }

    // Dequant each row through the pipelined DUT: stream rows in, collect
    // valid_out results in order.
    final got = <double>[];
    actScaleL.inject(actScaleBits);
    final collect = () async {
      var seen = 0;
      while (seen < rows) {
        await clk.nextNegedge;
        if (dq.output('valid_out').value.toBool()) {
          outFp.put(dq.output('y').value);
          got.add(outFp.floatingPointValue.toDouble());
          seen++;
        }
      }
    }();
    for (var r = 0; r < rows; r++) {
      dqValidIn.inject(1);
      accL.inject(acc[r] & 0xFFFFFFFF);
      rowScaleL.inject(e(qm.rowScales[r]));
      await clk.nextPosedge;
    }
    dqValidIn.inject(0);
    await collect;
    await Simulator.endSimulation();

    // Compare to the fp64 linear (the true target) and the quant oracle.
    for (var r = 0; r < rows; r++) {
      expect(
        got[r],
        closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
        reason: 'row $r: hw=${got[r]} fp=${goldenFp[r]} q=${goldenQ[r]}',
      );
      expect(
        got[r],
        closeTo(goldenQ[r], 0.12 + goldenQ[r].abs() * 0.15),
        reason: 'row $r vs quant oracle: hw=${got[r]} q=${goldenQ[r]}',
      );
    }
  });
}
