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

/// Computes `y = W @ rmsNorm(x, gamma)`. An on-chip FSM buffers the input
/// vector `x`, its norm weights `gamma`, and the matrix's per-row scales, runs
/// the two-pass [LoomRmsNorm] over `x`, pipes each normalized element straight
/// into an internal [LoomFpLinear] activation buffer, then fires the matmul.
/// This is the nonlinear->linear handoff the transformer sequencer performs at
/// every projection (a norm feeds Q/K/V and gate/up).
///
/// Load protocol (state LOAD): stream `x`+`gamma` together on `x_en`/`x_in`/
/// `gamma_in` (one pair per cycle), the per-row scales on `rs_en`/`rs_in`, set
/// `col_tiles`/`row_blocks`/`weight_base` and the norm constants `inv_n` (=1/N)
/// and `eps`, then pulse `start`. Results appear on `y`/`y_valid`; `done` pulses
/// at the end.
class LoomNormLinear extends BridgeModule {
  static const int _load = 0;
  static const int _rnacc = 1;
  static const int _rncomp = 2;
  static const int _rnwait = 3;
  static const int _rnnorm = 4;
  static const int _feeds = 5;
  static const int _fire = 6;
  static const int _run = 7;
  static const int _fin = 8;

  late final WishboneInterface mem;

  LoomNormLinear({
    int maxColTiles = 4,
    int maxRowBlocks = 4,
    int recipIterations = 4,
    String? name,
  }) : super('LoomNormLinear', name: name ?? 'loom_norm_linear') {
    const aw = 32;
    final maxCols = maxColTiles * 2;
    final maxRows = maxRowBlocks * 2;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('col_tiles', PortDirection.input, width: 16);
    createPort('row_blocks', PortDirection.input, width: 16);
    createPort('weight_base', PortDirection.input, width: aw);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('gamma_in', PortDirection.input, width: 16);
    createPort('rs_en', PortDirection.input);
    createPort('rs_in', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);
    final yValidP = addOutput('y_valid');
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

    final state = Logic(name: 'state', width: 4);
    final ci = Logic(name: 'ci', width: 16); // load x/gamma count
    final ri = Logic(name: 'ri', width: 16); // load scale count
    final k = Logic(name: 'k', width: 16); // phase element index
    final nEl = Logic(name: 'n_el', width: 16); // vector length (= inDim)
    final nSc = Logic(name: 'n_sc', width: 16); // scale count (= outDim)
    final ctReg = Logic(name: 'ct_reg', width: 16);
    final rbReg = Logic(name: 'rb_reg', width: 16);
    final wbReg = Logic(name: 'wb_reg', width: aw);

    final xbuf = [
      for (var i = 0; i < maxCols; i++) Logic(name: 'xb$i', width: 16),
    ];
    final gbuf = [
      for (var i = 0; i < maxCols; i++) Logic(name: 'gb$i', width: 16),
    ];
    final rsbuf = [
      for (var i = 0; i < maxRows; i++) Logic(name: 'rsb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 4));
    final one = Const(1, width: 16);

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
    final rmsReady = rms.output('ready');

    final fpl = LoomFpLinear(
      maxColTiles: maxColTiles,
      maxRowBlocks: maxRowBlocks,
      recipIterations: recipIterations,
    );
    fpl.input('clk').srcConnection! <= clk;
    fpl.input('reset').srcConnection! <= reset;
    fpl.input('start').srcConnection! <= st(_fire);
    fpl.input('col_tiles').srcConnection! <= ctReg;
    fpl.input('row_blocks').srcConnection! <= rbReg;
    fpl.input('weight_base').srcConnection! <= wbReg;
    // Pipe each normalized element straight into the activation buffer.
    fpl.input('x_en').srcConnection! <= st(_rnnorm);
    fpl.input('x_in').srcConnection! <= rms.output('y');
    fpl.input('rs_en').srcConnection! <= st(_feeds);
    fpl.input('rs_in').srcConnection! <= _sel(k, rsbuf);

    mem.cyc <= fpl.output('mem_CYC');
    mem.stb <= fpl.output('mem_STB');
    mem.we <= fpl.output('mem_WE');
    mem.adr <= fpl.output('mem_ADR');
    mem.datMosi <= fpl.output('mem_DAT_MOSI');
    mem.sel <= fpl.output('mem_SEL');
    fpl.input('mem_ACK').srcConnection! <= mem.ack;
    fpl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    yP <= fpl.output('y');
    yValidP <= fpl.output('y_valid');
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final xLoadWrites = <Conditional>[
      for (var i = 0; i < maxCols; i++)
        If(
          st(_load) & input('x_en') & ci.eq(Const(i, width: 16)),
          then: [xbuf[i] < input('x_in'), gbuf[i] < input('gamma_in')],
        ),
    ];
    final rsLoadWrites = <Conditional>[
      for (var i = 0; i < maxRows; i++)
        If(
          st(_load) & input('rs_en') & ri.eq(Const(i, width: 16)),
          then: [rsbuf[i] < input('rs_in')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          ci < Const(0, width: 16),
          ri < Const(0, width: 16),
          k < Const(0, width: 16),
          nEl < Const(0, width: 16),
          nSc < Const(0, width: 16),
          ctReg < Const(0, width: 16),
          rbReg < Const(0, width: 16),
          wbReg < Const(0, width: aw),
          for (final b in xbuf) b < Const(0, width: 16),
          for (final b in gbuf) b < Const(0, width: 16),
          for (final b in rsbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...xLoadWrites,
          ...rsLoadWrites,
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('x_en'), then: [ci < ci + one]),
              If(input('rs_en'), then: [ri < ri + one]),
              If(
                start,
                then: [
                  nEl < ci,
                  nSc < ri,
                  ctReg < input('col_tiles'),
                  rbReg < input('row_blocks'),
                  wbReg < input('weight_base'),
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
                rmsReady,
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
                  k < Const(0, width: 16),
                  state < Const(_feeds, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_feeds, width: 4), [
              If(
                k.eq(nSc - one),
                then: [state < Const(_fire, width: 4)],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_fire, width: 4), [state < Const(_run, width: 4)]),
            CaseItem(Const(_run, width: 4), [
              If(fpl.output('done'), then: [state < Const(_fin, width: 4)]),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              ci < Const(0, width: 16),
              ri < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
