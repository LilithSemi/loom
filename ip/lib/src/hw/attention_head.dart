import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_mac.dart';
import 'fp_softmax.dart';

/// Selects `entries[index]` with a balanced binary mux tree (depth ~log2 N).
Logic _muxTree(Logic index, List<Logic> entries) {
  if (entries.length == 1) return entries[0];
  final next = <Logic>[];
  for (var i = 0; i < entries.length; i += 2) {
    next.add(
      i + 1 < entries.length
          ? mux(index[0], entries[i + 1], entries[i])
          : entries[i],
    );
  }
  return _muxTree(index.getRange(1, index.width), next);
}

/// LoomAttentionHead: single-head causal attention for one query position,
/// matching the golden `causalGqaAttention` for one head:
///
///   scores[s] = sum_d q[d] * k[s][d]      (s = 0..L-1, L keys = qPos+1)
///   weights   = softmax(scores)
///   out[d]    = sum_s weights[s] * v[s][d]
///
/// The 1/sqrt(headDim) score scale is folded into `q` by the caller (linear in
/// q, so scaling q == scaling the dot), keeping this datapath multiplier-light.
///
/// Self-contained FSM that drives one shared LoomFpMac (score dots + weighted-V
/// sum) and one embedded LoomSoftmax. Buffers q[headDim], scores[maxSeq]
/// (reused to hold the softmax weights), out[headDim] as flop register files.
/// For SmolLM2 dims (headDim 64, ctx-length keys) those buffers should move to
/// BRAM. Flops are fine at the sizes the diff-test exercises.
///
/// Host/sequencer protocol (one cycle per element, a bubble cycle commits each
/// key / output lane):
///   1. pulse `start` (re-inits the MAC + softmax).
///   2. LOADQ: `q_en` + `q_in` x headDim  -> auto-advances to SCORE.
///   3. SCORE: per key s: `k_en` + `k_in` x headDim, then ONE idle cycle to
///      commit scores[s]. After L keys the FSM runs softmax internally (no host
///      input) and asserts `wv_ready`.
///   4. WV (once `wv_ready`): per dim d: `v_en` + `v_in` x L (v[0..L-1][d]),
///      then ONE idle cycle to commit out[d]. After headDim lanes `ready`
///      asserts.
///   5. read out[d] via `rd_addr` -> `out_at` (combinational).
class LoomAttentionHead extends BridgeModule {
  static const int _loadQ = 0;
  static const int _score = 1;
  static const int _smaxMax = 2;
  static const int _smaxSum = 3;
  static const int _smaxCalc = 4;
  static const int _smaxWait = 5;
  static const int _smaxNorm = 6;
  static const int _wv = 7;
  static const int _done = 8;

  LoomAttentionHead({
    required int headDim,
    required int maxSeq,
    int recipIterations = 4,
    String? name,
  }) : super('LoomAttentionHead', name: name ?? 'loom_attention_head') {
    final dW = (headDim - 1).bitLength.clamp(1, 32);
    final sW = (maxSeq - 1).bitLength.clamp(1, 32);
    final lenW = maxSeq.bitLength.clamp(1, 32);

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
    createPort('rd_addr', PortDirection.input, width: dW);
    final readyP = addOutput('ready');
    final wvReadyP = addOutput('wv_ready');
    final outAtP = addOutput('out_at', width: 16);

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');

    final state = Logic(name: 'state', width: 4);
    final dc = Logic(name: 'dc', width: dW); // dim counter
    final sc = Logic(name: 'sc', width: sW); // seq counter
    final pending = Logic(name: 'pending'); // SCORE key commit pending
    final pendingV = Logic(name: 'pending_v'); // WV lane commit pending

    final qReg = [
      for (var i = 0; i < headDim; i++) Logic(name: 'q$i', width: 16),
    ];
    final scReg = [
      for (var i = 0; i < maxSeq; i++) Logic(name: 'sc$i', width: 16),
    ];
    final outReg = [
      for (var i = 0; i < headDim; i++) Logic(name: 'o$i', width: 16),
    ];

    final lastKey = (input('seq_len') - Const(1, width: lenW)).getRange(0, sW);
    final lastDim = Const(headDim - 1, width: dW);

    Logic st(int s) => state.eq(Const(s, width: 4));

    final qSel = _muxTree(dc, qReg);
    final scSel = _muxTree(sc, scReg);

    final mac = LoomFpMac();
    mac.input('clk').srcConnection! <= clk;
    mac.input('reset').srcConnection! <= reset;
    final macEn = (st(_score) & input('k_en')) | (st(_wv) & input('v_en'));
    final macClear =
        st(_loadQ) |
        (st(_score) & pending & ~input('k_en')) |
        (st(_wv) & pendingV & ~input('v_en'));
    mac.input('clear').srcConnection! <= macClear;
    mac.input('en').srcConnection! <= macEn;
    mac.input('a').srcConnection! <= mux(st(_score), qSel, scSel);
    mac.input('b').srcConnection! <=
        mux(st(_score), input('k_in'), input('v_in'));
    final macAcc = mac.output('acc');

    final smax = LoomSoftmax(recipIterations: recipIterations);
    smax.input('clk').srcConnection! <= clk;
    smax.input('reset').srcConnection! <= reset | start; // re-init per query
    smax.input('max_en').srcConnection! <= st(_smaxMax);
    smax.input('sum_en').srcConnection! <= st(_smaxSum);
    smax.input('compute').srcConnection! <= st(_smaxCalc);
    smax.input('norm_en').srcConnection! <= st(_smaxNorm);
    smax.input('x_in').srcConnection! <= scSel;
    final smaxY = smax.output('y');
    final smaxReady = smax.output('ready');

    final qWe = st(_loadQ) & input('q_en');
    final scWe = (st(_score) & pending & ~input('k_en')) | st(_smaxNorm);
    final scVal = mux(st(_smaxNorm), smaxY, macAcc);
    final outWe = st(_wv) & pendingV & ~input('v_en');

    final writes = <Conditional>[];
    for (var i = 0; i < headDim; i++) {
      writes.add(
        If(qWe & dc.eq(Const(i, width: dW)), then: [qReg[i] < input('q_in')]),
      );
      writes.add(
        If(outWe & dc.eq(Const(i, width: dW)), then: [outReg[i] < macAcc]),
      );
    }
    for (var i = 0; i < maxSeq; i++) {
      writes.add(
        If(scWe & sc.eq(Const(i, width: sW)), then: [scReg[i] < scVal]),
      );
    }

    final atLastKey = sc.eq(lastKey);
    final scInc = mux(atLastKey, Const(0, width: sW), sc + Const(1, width: sW));
    final atLastDim = dc.eq(lastDim);

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_loadQ, width: 4),
          dc < Const(0, width: dW),
          sc < Const(0, width: sW),
          pending < Const(0),
          pendingV < Const(0),
        ],
        orElse: [
          If(
            start,
            then: [
              state < Const(_loadQ, width: 4),
              dc < Const(0, width: dW),
              sc < Const(0, width: sW),
              pending < Const(0),
              pendingV < Const(0),
            ],
          ),
          ...writes,
          Case(state, [
            CaseItem(Const(_loadQ, width: 4), [
              If(
                input('q_en'),
                then: [
                  If(
                    dc.eq(lastDim),
                    then: [
                      state < Const(_score, width: 4),
                      dc < Const(0, width: dW),
                      sc < Const(0, width: sW),
                    ],
                    orElse: [dc < dc + Const(1, width: dW)],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_score, width: 4), [
              If(
                input('k_en'),
                then: [
                  If(
                    dc.eq(lastDim),
                    then: [
                      dc < Const(0, width: dW),
                      pending < Const(1),
                    ],
                    orElse: [dc < dc + Const(1, width: dW)],
                  ),
                ],
                orElse: [
                  If(
                    pending,
                    then: [
                      pending < Const(0),
                      dc < Const(0, width: dW),
                      sc < scInc,
                      If(atLastKey, then: [state < Const(_smaxMax, width: 4)]),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_smaxMax, width: 4), [
              sc < scInc,
              If(atLastKey, then: [state < Const(_smaxSum, width: 4)]),
            ]),
            CaseItem(Const(_smaxSum, width: 4), [
              sc < scInc,
              If(atLastKey, then: [state < Const(_smaxCalc, width: 4)]),
            ]),
            CaseItem(Const(_smaxCalc, width: 4), [
              state < Const(_smaxWait, width: 4),
            ]),
            CaseItem(Const(_smaxWait, width: 4), [
              If(
                smaxReady,
                then: [
                  state < Const(_smaxNorm, width: 4),
                  sc < Const(0, width: sW),
                ],
              ),
            ]),
            CaseItem(Const(_smaxNorm, width: 4), [
              sc < scInc,
              If(
                atLastKey,
                then: [
                  state < Const(_wv, width: 4),
                  dc < Const(0, width: dW),
                ],
              ),
            ]),
            CaseItem(Const(_wv, width: 4), [
              If(
                input('v_en'),
                then: [
                  sc < scInc,
                  If(atLastKey, then: [pendingV < Const(1)]),
                ],
                orElse: [
                  If(
                    pendingV,
                    then: [
                      pendingV < Const(0),
                      sc < Const(0, width: sW),
                      If(
                        atLastDim,
                        then: [state < Const(_done, width: 4)],
                        orElse: [dc < dc + Const(1, width: dW)],
                      ),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_done, width: 4), []),
          ]),
        ],
      ),
    ]);

    readyP <= st(_done);
    wvReadyP <= st(_wv);
    outAtP <= _muxTree(input('rd_addr'), outReg);
  }
}
