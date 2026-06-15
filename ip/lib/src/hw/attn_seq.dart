library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'attention_head.dart';

/// Balanced mux-tree select: `entries[index]`.
Logic _sel(Logic index, List<Logic> entries) {
  if (entries.length == 1) return entries[0];
  final next = <Logic>[];
  for (var i = 0; i < entries.length; i += 2) {
    next.add(
      i + 1 < entries.length
          ? mux(index[0], entries[i + 1], entries[i])
          : entries[i],
    );
  }
  return _sel(index.getRange(1, index.width), next);
}

/// Third sequencer brick: single-head causal attention driven end-to-end by an
/// on-chip FSM. The sequencer OWNS the K/V cache (flop store per position) and
/// replays it into a [LoomAttentionHead] in the exact order that block wants:
/// Q (pre-scaled), then K per key (headDim + a commit bubble), then V per DIM
/// LANE (seq_len + bubble, i.e. transposed), reading the per-dim result out.
/// This proves the KV-replay orchestration the transformer sequencer performs
/// at every attention step (the head itself stores no KV).
///
/// Load protocol (state LOAD): set `seq_len` (= L keys), stream `q` (headDim,
/// scale pre-folded) on `q_en`/`q_in`, and K then V as L*headDim values each in
/// [key][dim] row-major order on `k_en`/`k_in` and `v_en`/`v_in`. Pulse `start`.
/// Results stream out on `o`/`o_valid` (headDim values, dim order); `done`
/// pulses at the end.
class LoomAttnSeq extends BridgeModule {
  static const int _load = 0;
  static const int _sah = 1; // pulse head start
  static const int _lq = 2; // load Q
  static const int _sck = 3; // score: feed key
  static const int _scb = 4; // score: commit bubble
  static const int _wvw = 5; // wait wv_ready (softmax)
  static const int _wvk = 6; // weighted-V: feed value lane
  static const int _wvb = 7; // weighted-V: commit bubble
  static const int _dw = 8; // wait ready
  static const int _emit = 9; // stream results
  static const int _fin = 10;

  LoomAttnSeq({required int headDim, required int maxSeq, String? name})
    : super('LoomAttnSeq', name: name ?? 'loom_attn_seq') {
    final lenW = maxSeq.bitLength;
    final rdW = (headDim - 1).bitLength.clamp(1, 32);
    final kvLen = maxSeq * headDim;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('seq_len', PortDirection.input, width: lenW);
    createPort('q_en', PortDirection.input);
    createPort('q_in', PortDirection.input, width: 16);
    createPort('k_en', PortDirection.input);
    createPort('k_in', PortDirection.input, width: 16);
    createPort('v_en', PortDirection.input);
    createPort('v_in', PortDirection.input, width: 16);
    final oP = addOutput('o', width: 16);
    final oValidP = addOutput('o_valid');
    final doneP = addOutput('done');
    final busyP = addOutput('busy');

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');
    final one = Const(1, width: 16);

    final state = Logic(name: 'state', width: 4);
    final qc = Logic(name: 'qc', width: 16); // load q count
    final kc = Logic(name: 'kc', width: 16); // load k count
    final vc = Logic(name: 'vc', width: 16); // load v count
    final lReg = Logic(name: 'l_reg', width: 16); // key count L
    final sReg = Logic(name: 's_reg', width: 16); // key index
    final dReg = Logic(name: 'd_reg', width: 16); // dim index
    final qi = Logic(name: 'qi', width: 16); // Q feed index
    final ei = Logic(name: 'ei', width: 16); // emit index

    final qbuf = [
      for (var i = 0; i < headDim; i++) Logic(name: 'qb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 4));
    final hd = Const(headDim, width: 16);
    // Storage index for key s, dim d (row-major [key][dim]).
    Logic addr(Logic s, Logic d) => (s * hd).getRange(0, 16) + d;

    // K/V working buffers live in BRAM, not flop arrays: the [key][dim] replay
    // read was a kvLen:1 mux and the load a kvLen-way write decode (this block's
    // LUT hog). Load (state _load) and replay (score/weighted-V) are DISJOINT
    // phases, so there is never a same-cycle read+write - a plain registered-read
    // BRAM is safe. Its 1-cycle read latency is absorbed by delaying the head's
    // k_en/v_en one cycle (kEnD/vEnD) so the enable lines up with the read data.
    final aw = (kvLen - 1).bitLength.clamp(1, 32);
    final rdAddr = addr(sReg, dReg).getRange(0, aw);
    final kbram = HarborBram(
      clk,
      width: 16,
      depth: kvLen,
      wrEn: st(_load) & input('k_en'),
      wrAddr: kc.getRange(0, aw),
      wrData: input('k_in'),
      rdAddr: rdAddr,
      name: 'kbram',
    );
    final vbram = HarborBram(
      clk,
      width: 16,
      depth: kvLen,
      wrEn: st(_load) & input('v_en'),
      wrAddr: vc.getRange(0, aw),
      wrData: input('v_in'),
      rdAddr: rdAddr,
      name: 'vbram',
    );
    // Read enables delayed one cycle to match the BRAM read latency.
    final kEnD = Logic(name: 'k_en_d');
    final vEnD = Logic(name: 'v_en_d');

    final ah = LoomAttentionHead(headDim: headDim, maxSeq: maxSeq);
    ah.input('clk').srcConnection! <= clk;
    ah.input('reset').srcConnection! <= reset;
    ah.input('start').srcConnection! <= st(_sah);
    ah.input('seq_len').srcConnection! <= lReg.getRange(0, lenW);
    ah.input('q_en').srcConnection! <= st(_lq);
    ah.input('q_in').srcConnection! <= _sel(qi, qbuf);
    ah.input('k_en').srcConnection! <= kEnD;
    ah.input('k_in').srcConnection! <= kbram.rdData;
    ah.input('v_en').srcConnection! <= vEnD;
    ah.input('v_in').srcConnection! <= vbram.rdData;
    ah.input('rd_addr').srcConnection! <= ei.getRange(0, rdW);

    oP <= ah.output('out_at');
    oValidP <= st(_emit);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final qLoad = <Conditional>[
      for (var i = 0; i < headDim; i++)
        If(
          st(_load) & input('q_en') & qc.eq(Const(i, width: 16)),
          then: [qbuf[i] < input('q_in')],
        ),
    ];
    // K/V loads are the kbram/vbram synchronous writes above. No flop decode.

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          qc < Const(0, width: 16),
          kc < Const(0, width: 16),
          vc < Const(0, width: 16),
          lReg < Const(0, width: 16),
          sReg < Const(0, width: 16),
          dReg < Const(0, width: 16),
          qi < Const(0, width: 16),
          ei < Const(0, width: 16),
          for (final b in qbuf) b < Const(0, width: 16),
          kEnD < Const(0),
          vEnD < Const(0),
        ],
        orElse: [
          ...qLoad,
          // Delay the head's K/V enables one cycle to match the BRAM read.
          kEnD < st(_sck),
          vEnD < st(_wvk),
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('q_en'), then: [qc < qc + one]),
              If(input('k_en'), then: [kc < kc + one]),
              If(input('v_en'), then: [vc < vc + one]),
              If(
                start,
                then: [
                  lReg < input('seq_len').zeroExtend(16),
                  qi < Const(0, width: 16),
                  state < Const(_sah, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_sah, width: 4), [state < Const(_lq, width: 4)]),
            CaseItem(Const(_lq, width: 4), [
              If(
                qi.eq(hd - one),
                then: [
                  sReg < Const(0, width: 16),
                  dReg < Const(0, width: 16),
                  state < Const(_sck, width: 4),
                ],
                orElse: [qi < qi + one],
              ),
            ]),
            CaseItem(Const(_sck, width: 4), [
              If(
                dReg.eq(hd - one),
                then: [
                  dReg < Const(0, width: 16),
                  state < Const(_scb, width: 4),
                ],
                orElse: [dReg < dReg + one],
              ),
            ]),
            CaseItem(Const(_scb, width: 4), [
              If(
                sReg.eq(lReg - one),
                then: [state < Const(_wvw, width: 4)],
                orElse: [sReg < sReg + one, state < Const(_sck, width: 4)],
              ),
            ]),
            CaseItem(Const(_wvw, width: 4), [
              If(
                ah.output('wv_ready'),
                then: [
                  sReg < Const(0, width: 16),
                  dReg < Const(0, width: 16),
                  state < Const(_wvk, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_wvk, width: 4), [
              If(
                sReg.eq(lReg - one),
                then: [
                  sReg < Const(0, width: 16),
                  state < Const(_wvb, width: 4),
                ],
                orElse: [sReg < sReg + one],
              ),
            ]),
            CaseItem(Const(_wvb, width: 4), [
              If(
                dReg.eq(hd - one),
                then: [state < Const(_dw, width: 4)],
                orElse: [dReg < dReg + one, state < Const(_wvk, width: 4)],
              ),
            ]),
            CaseItem(Const(_dw, width: 4), [
              If(
                ah.output('ready'),
                then: [
                  ei < Const(0, width: 16),
                  state < Const(_emit, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_emit, width: 4), [
              If(
                ei.eq(hd - one),
                then: [state < Const(_fin, width: 4)],
                orElse: [ei < ei + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              qc < Const(0, width: 16),
              kc < Const(0, width: 16),
              vc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
