library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_linear.dart';
import 'fp_rmsnorm.dart';

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

/// The attention front: RMSNorm(x) feeds THREE projections - Q [qDim x H], K
/// and V [kvDim x H] - through ONE reused [LoomFpLinear] engine, the Q/K/V
/// pattern the full decoder layer needs.
///
/// Load (state LOAD): stream `x`+`gamma` on `x_en`/`x_in`/`gamma_in`. The three
/// scale tables on `sq_en`/`sk_en`/`sv_en`. Set `wb_q`/`wb_k`/`wb_v`, `inv_n`
/// (=1/H), `eps`. Pulse `start`. Results stream on `y`/`y_valid` with `y_phase`
/// (0=Q,1=K,2=V); `done` pulses at the end.
class LoomQkvNorm extends BridgeModule {
  static const int _load = 0;
  static const int _rnacc = 1;
  static const int _rncomp = 2;
  static const int _rnwait = 3;
  static const int _rnnorm = 4;
  static const int _feed = 5;
  static const int _fire = 6;
  static const int _run = 7;
  static const int _fin = 8;

  late final WishboneInterface mem;

  LoomQkvNorm({
    required int hidden,
    required int qDim,
    required int kvDim,
    int recipIterations = 4,
    String? name,
  }) : super('LoomQkvNorm', name: name ?? 'loom_qkv_norm') {
    const aw = 32;
    final ct = (hidden + 1) ~/ 2; // shared inner-dim tiles (inDim = hidden)
    final rbQ = (qDim + 1) ~/ 2;
    final rbKV = (kvDim + 1) ~/ 2;
    final maxTiles = ct;
    final maxBlocks = rbQ > rbKV ? rbQ : rbKV;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('wb_q', PortDirection.input, width: aw);
    createPort('wb_k', PortDirection.input, width: aw);
    createPort('wb_v', PortDirection.input, width: aw);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('gamma_in', PortDirection.input, width: 16);
    createPort('sq_en', PortDirection.input);
    createPort('sq_in', PortDirection.input, width: 16);
    createPort('sk_en', PortDirection.input);
    createPort('sk_in', PortDirection.input, width: 16);
    createPort('sv_en', PortDirection.input);
    createPort('sv_in', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);
    final yValidP = addOutput('y_valid');
    final yPhaseP = addOutput('y_phase', width: 2);
    final doneP = addOutput('done');
    final busyP = addOutput('busy');

    final memRef = addInterface(
      WishboneInterface(WishboneConfig(addressWidth: aw, dataWidth: 32)),
      name: 'mem',
      role: PairRole.provider,
    );
    mem = memRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');
    final one = Const(1, width: 16);

    final state = Logic(name: 'state', width: 4);
    final phase = Logic(name: 'phase', width: 2); // 0=Q, 1=K, 2=V
    final xc = Logic(name: 'xc', width: 16);
    final sqc = Logic(name: 'sqc', width: 16);
    final skc = Logic(name: 'skc', width: 16);
    final svc = Logic(name: 'svc', width: 16);
    final k = Logic(name: 'k', width: 16); // norm element index
    final fk = Logic(name: 'fk', width: 16); // feed index
    final nEl = Logic(name: 'n_el', width: 16); // = hidden

    final xbuf = [
      for (var i = 0; i < hidden; i++) Logic(name: 'xb$i', width: 16),
    ];
    final gbuf = [
      for (var i = 0; i < hidden; i++) Logic(name: 'gb$i', width: 16),
    ];
    final nbuf = [
      for (var i = 0; i < hidden; i++) Logic(name: 'nb$i', width: 16),
    ];
    final sqbuf = [
      for (var i = 0; i < qDim; i++) Logic(name: 'sqb$i', width: 16),
    ];
    final skbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'skb$i', width: 16),
    ];
    final svbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'svb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 4));
    final isQ = phase.eq(Const(0, width: 2));
    final isK = phase.eq(Const(1, width: 2));

    // outDim of the current phase, and the scale source.
    final outDim = mux(isQ, Const(qDim, width: 16), Const(kvDim, width: 16));
    final feedLen = mux(
      nEl.gt(outDim),
      nEl,
      outDim,
    ); // acts(H) vs scales(outDim)

    // RMSNorm front.
    final rms = LoomRmsNorm(recipIterations: recipIterations);
    rms.input('clk').srcConnection! <= clk;
    rms.input('reset').srcConnection! <= reset;
    rms.input('acc_en').srcConnection! <= st(_rnacc);
    rms.input('compute').srcConnection! <= st(_rncomp);
    rms.input('norm_en').srcConnection! <= st(_rnnorm);
    rms.input('x_in').srcConnection! <= _sel(k, xbuf);
    rms.input('gamma_in').srcConnection! <= _sel(k, gbuf);
    rms.input('eps').srcConnection! <= input('eps');
    rms.input('inv_n').srcConnection! <= input('inv_n');

    // Reused matmul engine (acts = normed vector from nbuf, weights from mem).
    final fpl = LoomFpLinear(
      maxColTiles: maxTiles,
      maxRowBlocks: maxBlocks,
      recipIterations: recipIterations,
    );
    fpl.input('clk').srcConnection! <= clk;
    fpl.input('reset').srcConnection! <= reset;
    fpl.input('start').srcConnection! <= st(_fire);
    fpl.input('col_tiles').srcConnection! <= Const(ct, width: 16);
    fpl.input('row_blocks').srcConnection! <=
        mux(isQ, Const(rbQ, width: 16), Const(rbKV, width: 16));
    fpl.input('weight_base').srcConnection! <=
        mux(isQ, input('wb_q'), mux(isK, input('wb_k'), input('wb_v')));
    final feedingX = st(_feed) & fk.lt(nEl);
    final feedingS = st(_feed) & fk.lt(outDim);
    fpl.input('x_en').srcConnection! <= feedingX;
    fpl.input('x_in').srcConnection! <= _sel(fk, nbuf);
    fpl.input('rs_en').srcConnection! <= feedingS;
    fpl.input('rs_in').srcConnection! <=
        mux(isQ, _sel(fk, sqbuf), mux(isK, _sel(fk, skbuf), _sel(fk, svbuf)));

    mem.cyc <= fpl.output('mem_CYC');
    mem.stb <= fpl.output('mem_STB');
    mem.we <= fpl.output('mem_WE');
    mem.adr <= fpl.output('mem_ADR');
    mem.datMosi <= fpl.output('mem_DAT_MOSI');
    mem.sel <= fpl.output('mem_SEL');
    fpl.input('mem_ACK').srcConnection! <= mem.ack;
    fpl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    yP <= fpl.output('y');
    yValidP <= st(_run) & fpl.output('y_valid');
    yPhaseP <= phase;
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < hidden; i++)
        If(
          st(_load) & input('x_en') & xc.eq(Const(i, width: 16)),
          then: [xbuf[i] < input('x_in'), gbuf[i] < input('gamma_in')],
        ),
      for (var i = 0; i < qDim; i++)
        If(
          st(_load) & input('sq_en') & sqc.eq(Const(i, width: 16)),
          then: [sqbuf[i] < input('sq_in')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_load) & input('sk_en') & skc.eq(Const(i, width: 16)),
          then: [skbuf[i] < input('sk_in')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_load) & input('sv_en') & svc.eq(Const(i, width: 16)),
          then: [svbuf[i] < input('sv_in')],
        ),
    ];
    // Capture the normalized vector into nbuf during RNNORM.
    final normWrites = <Conditional>[
      for (var i = 0; i < hidden; i++)
        If(
          st(_rnnorm) & rms.output('y_valid') & k.eq(Const(i, width: 16)),
          then: [nbuf[i] < rms.output('y')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          phase < Const(0, width: 2),
          xc < Const(0, width: 16),
          sqc < Const(0, width: 16),
          skc < Const(0, width: 16),
          svc < Const(0, width: 16),
          k < Const(0, width: 16),
          fk < Const(0, width: 16),
          nEl < Const(0, width: 16),
          for (final b in xbuf) b < Const(0, width: 16),
          for (final b in gbuf) b < Const(0, width: 16),
          for (final b in nbuf) b < Const(0, width: 16),
          for (final b in sqbuf) b < Const(0, width: 16),
          for (final b in skbuf) b < Const(0, width: 16),
          for (final b in svbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...normWrites,
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('x_en'), then: [xc < xc + one]),
              If(input('sq_en'), then: [sqc < sqc + one]),
              If(input('sk_en'), then: [skc < skc + one]),
              If(input('sv_en'), then: [svc < svc + one]),
              If(
                start,
                then: [
                  nEl < xc,
                  k < Const(0, width: 16),
                  state < Const(_rnacc, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_rnacc, width: 4), [
              If(
                k.eq(nEl - one),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_rncomp, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_rncomp, width: 4), [
              state < Const(_rnwait, width: 4),
            ]),
            CaseItem(Const(_rnwait, width: 4), [
              If(
                rms.output('ready'),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_rnnorm, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_rnnorm, width: 4), [
              If(
                k.eq(nEl - one),
                then: [
                  phase < Const(0, width: 2),
                  fk < Const(0, width: 16),
                  state < Const(_feed, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_feed, width: 4), [
              If(
                fk.eq(feedLen),
                then: [state < Const(_fire, width: 4)],
                orElse: [fk < fk + one],
              ),
            ]),
            CaseItem(Const(_fire, width: 4), [state < Const(_run, width: 4)]),
            CaseItem(Const(_run, width: 4), [
              If(
                fpl.output('done'),
                then: [
                  If(
                    phase.eq(Const(2, width: 2)),
                    then: [state < Const(_fin, width: 4)],
                    orElse: [
                      phase < phase + Const(1, width: 2),
                      fk < Const(0, width: 16),
                      state < Const(_feed, width: 4),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              xc < Const(0, width: 16),
              sqc < Const(0, width: 16),
              skc < Const(0, width: 16),
              svc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
