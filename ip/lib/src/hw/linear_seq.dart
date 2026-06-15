library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_linear.dart';

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

/// Hardwired FSM that buffers an activation vector plus per-row weight scales,
/// then autonomously streams them into an internal [LoomFpLinear] and surfaces
/// the fp16 results, forwarding the block's Wishbone weight-read master to the
/// top.
///
/// Load protocol (state LOAD): stream the activation vector on `x_en`/`x_in`
/// (one fp16 per cycle) and the per-row weight scales on `rs_en`/`rs_in`, set
/// `col_tiles`/`row_blocks`/`weight_base`, then pulse `start`. The FSM latches
/// the counts, replays both streams into the engine, fires it, and passes `y`/
/// `y_valid` straight through. `done` pulses when the matmul completes.
class LoomLinearSeq extends BridgeModule {
  static const int _load = 0;
  static const int _feed = 1;
  static const int _fire = 2;
  static const int _run = 3;
  static const int _fin = 4;

  late final WishboneInterface mem;

  LoomLinearSeq({
    int maxColTiles = 4,
    int maxRowBlocks = 4,
    int recipIterations = 4,
    String? name,
  }) : super('LoomLinearSeq', name: name ?? 'loom_linear_seq') {
    const aw = 32;
    final maxCols = maxColTiles * 2;
    final maxRows = maxRowBlocks * 2;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('col_tiles', PortDirection.input, width: 16);
    createPort('row_blocks', PortDirection.input, width: 16);
    createPort('weight_base', PortDirection.input, width: aw);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
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

    final state = Logic(name: 'state', width: 3);
    final ci = Logic(name: 'ci', width: 16); // acts buffered
    final ri = Logic(name: 'ri', width: 16); // scales buffered
    final fk = Logic(name: 'fk', width: 16); // feed index
    final nActs = Logic(name: 'n_acts', width: 16);
    final nScales = Logic(name: 'n_scales', width: 16);
    final ctReg = Logic(name: 'ct_reg', width: 16);
    final rbReg = Logic(name: 'rb_reg', width: 16);
    final wbReg = Logic(name: 'wb_reg', width: aw);

    final xbuf = [
      for (var i = 0; i < maxCols; i++) Logic(name: 'xb$i', width: 16),
    ];
    final rsbuf = [
      for (var i = 0; i < maxRows; i++) Logic(name: 'rsb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 3));

    final feedLen = mux(nActs.gt(nScales), nActs, nScales);
    final feedingX = st(_feed) & fk.lt(nActs);
    final feedingS = st(_feed) & fk.lt(nScales);

    // The reused matmul engine.
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
    fpl.input('x_en').srcConnection! <= feedingX;
    fpl.input('x_in').srcConnection! <= _sel(fk, xbuf);
    fpl.input('rs_en').srcConnection! <= feedingS;
    fpl.input('rs_in').srcConnection! <= _sel(fk, rsbuf);

    // Forward the engine's weight-read master to our 'mem' interface.
    mem.cyc <= fpl.output('mem_CYC');
    mem.stb <= fpl.output('mem_STB');
    mem.we <= fpl.output('mem_WE');
    mem.adr <= fpl.output('mem_ADR');
    mem.datMosi <= fpl.output('mem_DAT_MOSI');
    mem.sel <= fpl.output('mem_SEL');
    fpl.input('mem_ACK').srcConnection! <= mem.ack;
    fpl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    // Results pass straight through.
    yP <= fpl.output('y');
    yValidP <= fpl.output('y_valid');
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final xLoadWrites = <Conditional>[
      for (var i = 0; i < maxCols; i++)
        If(
          st(_load) & input('x_en') & ci.eq(Const(i, width: 16)),
          then: [xbuf[i] < input('x_in')],
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
          state < Const(_load, width: 3),
          ci < Const(0, width: 16),
          ri < Const(0, width: 16),
          fk < Const(0, width: 16),
          nActs < Const(0, width: 16),
          nScales < Const(0, width: 16),
          ctReg < Const(0, width: 16),
          rbReg < Const(0, width: 16),
          wbReg < Const(0, width: aw),
          for (final b in xbuf) b < Const(0, width: 16),
          for (final b in rsbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...xLoadWrites,
          ...rsLoadWrites,
          Case(state, [
            CaseItem(Const(_load, width: 3), [
              If(input('x_en'), then: [ci < ci + Const(1, width: 16)]),
              If(input('rs_en'), then: [ri < ri + Const(1, width: 16)]),
              If(
                start,
                then: [
                  nActs < ci,
                  nScales < ri,
                  ctReg < input('col_tiles'),
                  rbReg < input('row_blocks'),
                  wbReg < input('weight_base'),
                  fk < Const(0, width: 16),
                  state < Const(_feed, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_feed, width: 3), [
              If(
                fk.eq(feedLen),
                then: [state < Const(_fire, width: 3)],
                orElse: [fk < fk + Const(1, width: 16)],
              ),
            ]),
            CaseItem(Const(_fire, width: 3), [state < Const(_run, width: 3)]),
            CaseItem(Const(_run, width: 3), [
              If(fpl.output('done'), then: [state < Const(_fin, width: 3)]),
            ]),
            CaseItem(Const(_fin, width: 3), [
              state < Const(_load, width: 3),
              ci < Const(0, width: 16),
              ri < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
