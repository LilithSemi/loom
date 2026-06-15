// LoomAttentionHead diff-test: single-head causal attention in fp16 vs the
// golden causalGqaAttention (numHeads = numKvHeads = 1), within fp16 tolerance.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/hw/attention_head.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('attention head matches golden causal attention', () async {
    const headDim = 4;
    const maxSeq = 4;
    const L = 3; // query position 2 attends to keys 0..2
    final scale = 1.0 / math.sqrt(headDim.toDouble());

    final q = [0.5, -1.0, 2.0, 1.0];
    final kRows = [
      [1.0, 0.0, -1.0, 0.5],
      [0.5, 2.0, -1.0, 1.0],
      [-1.0, 1.0, 0.5, -2.0],
    ];
    final vRows = [
      [1.0, -1.0, 2.0, 0.5],
      [0.5, 2.0, -1.0, 1.0],
      [-1.0, 1.0, 0.5, -2.0],
    ];

    // Golden: build T=L rows, only query row L-1 matters for result[L-1].
    final qg = [
      for (var t = 0; t < L; t++)
        Float64List.fromList(t == L - 1 ? q : List.filled(headDim, 0.0)),
    ];
    final kg = [for (final r in kRows) Float64List.fromList(r)];
    final vg = [for (final r in vRows) Float64List.fromList(r)];
    final golden = causalGqaAttention(qg, kg, vg, 1, 1, headDim)[L - 1];

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final seqLen = Logic(name: 'seq_len', width: maxSeq.bitLength);
    final qEn = Logic(name: 'q_en');
    final qInL = Logic(name: 'q_in', width: 16);
    final kEn = Logic(name: 'k_en');
    final kInL = Logic(name: 'k_in', width: 16);
    final vEn = Logic(name: 'v_en');
    final vInL = Logic(name: 'v_in', width: 16);
    final rdAddr = Logic(name: 'rd_addr', width: (headDim - 1).bitLength);

    final dut = LoomAttentionHead(headDim: headDim, maxSeq: maxSeq);
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('start', start),
      ('seq_len', seqLen),
      ('q_en', qEn),
      ('q_in', qInL),
      ('k_en', kEn),
      ('k_in', kInL),
      ('v_en', vEn),
      ('v_in', vInL),
      ('rd_addr', rdAddr),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [start, qEn, kEn, vEn]) {
      l.inject(0);
    }
    qInL.inject(0);
    kInL.inject(0);
    vInL.inject(0);
    rdAddr.inject(0);
    seqLen.inject(L);
    reset.inject(1);
    Simulator.setMaxSimTime(5000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // start pulse (re-inits MAC + softmax).
    start.inject(1);
    await clk.nextPosedge;
    start.inject(0);

    // LOADQ: feed pre-scaled q (scale folded into q).
    for (var d = 0; d < headDim; d++) {
      qEn.inject(1);
      qInL.inject(e(q[d] * scale));
      await clk.nextPosedge;
    }
    qEn.inject(0);

    // SCORE: per key, headDim k_en then one commit bubble.
    for (var s = 0; s < L; s++) {
      for (var d = 0; d < headDim; d++) {
        kEn.inject(1);
        kInL.inject(e(kRows[s][d]));
        await clk.nextPosedge;
      }
      kEn.inject(0);
      await clk.nextPosedge; // commit scores[s]
    }

    // Softmax runs internally. Wait for wv_ready.
    var guard = 0;
    while (!dut.output('wv_ready').value.toBool() && guard++ < 500) {
      await clk.nextPosedge;
    }
    expect(
      dut.output('wv_ready').value.toBool(),
      isTrue,
      reason: 'reached WV phase',
    );

    // WV: per dim lane, L v_en (v[0..L-1][d]) then one commit bubble.
    for (var d = 0; d < headDim; d++) {
      for (var s = 0; s < L; s++) {
        vEn.inject(1);
        vInL.inject(e(vRows[s][d]));
        await clk.nextPosedge;
      }
      vEn.inject(0);
      await clk.nextPosedge; // commit out[d]
    }

    guard = 0;
    while (!dut.output('ready').value.toBool() && guard++ < 100) {
      await clk.nextPosedge;
    }
    expect(dut.output('ready').value.toBool(), isTrue, reason: 'reached DONE');

    // Read out[d].
    final got = <double>[];
    for (var d = 0; d < headDim; d++) {
      rdAddr.inject(d);
      await clk.nextNegedge;
      outFp.put(dut.output('out_at').value);
      got.add(outFp.floatingPointValue.toDouble());
    }
    await Simulator.endSimulation();

    for (var d = 0; d < headDim; d++) {
      expect(
        got[d],
        closeTo(golden[d], 0.05 + golden[d].abs() * 0.1),
        reason: 'out[$d]=${got[d]} vs ${golden[d]}',
      );
    }
  });
}
