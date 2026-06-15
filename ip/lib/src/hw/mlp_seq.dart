library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_linear.dart';
import 'fp_silu.dart';

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

/// Gated-SiLU (SwiGLU) MLP body: `y = W_down @ (SiLU(W_gate @ x) * (W_up @ x))`.
/// One reused [LoomFpLinear] is time-shared across the three matmuls (gate, up,
/// down), with an elementwise `SiLU(gate) * up` between the second and third.
/// `x` is the (already normalized) MLP input. `H`/`iSize` are compile-time so
/// the tile geometry is baked.
///
/// Load protocol (state LOAD): stream `x` (H values) on `x_en`/`x_in`, the three
/// scale tables on `sg_en`/`su_en`/`sd_en` (iSize, iSize, H fp16 scales), set the
/// three `wb_*` weight-base addresses, then pulse `start`. Results stream out on
/// `o`/`o_valid` (H values). `done` pulses at the end.
class LoomMlpSeq extends BridgeModule {
  static const int _load = 0;
  static const int _feed = 1;
  static const int _fire = 2;
  static const int _run = 3;
  static const int _act = 4;
  static const int _fin = 5;

  late final WishboneInterface mem;

  LoomMlpSeq({
    required int hidden,
    required int iSize,
    int recipIterations = 4,
    String? name,
  }) : super('LoomMlpSeq', name: name ?? 'loom_mlp_seq') {
    const aw = 32;
    final ctGU = (hidden + 1) ~/ 2; // gate/up: cols = H
    final rbGU = (iSize + 1) ~/ 2; // gate/up: rows = iSize
    final ctD = (iSize + 1) ~/ 2; // down: cols = iSize
    final rbD = (hidden + 1) ~/ 2; // down: rows = H
    final maxTiles = ctGU > ctD ? ctGU : ctD;
    final maxBlocks = rbGU > rbD ? rbGU : rbD;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('wb_gate', PortDirection.input, width: aw);
    createPort('wb_up', PortDirection.input, width: aw);
    createPort('wb_down', PortDirection.input, width: aw);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('sg_en', PortDirection.input);
    createPort('sg_in', PortDirection.input, width: 16);
    createPort('su_en', PortDirection.input);
    createPort('su_in', PortDirection.input, width: 16);
    createPort('sd_en', PortDirection.input);
    createPort('sd_in', PortDirection.input, width: 16);
    final oP = addOutput('o', width: 16);
    final oValidP = addOutput('o_valid');
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

    final state = Logic(name: 'state', width: 3);
    final phase = Logic(name: 'phase', width: 2); // 0 gate, 1 up, 2 down
    final xc = Logic(name: 'xc', width: 16);
    final sgc = Logic(name: 'sgc', width: 16);
    final suc = Logic(name: 'suc', width: 16);
    final sdc = Logic(name: 'sdc', width: 16);
    final fk = Logic(name: 'fk', width: 16); // feed index
    final ry = Logic(name: 'ry', width: 16); // result index
    final ai = Logic(name: 'ai', width: 16); // act index

    final xbuf = [
      for (var i = 0; i < hidden; i++) Logic(name: 'xb$i', width: 16),
    ];
    final gbuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'gb$i', width: 16),
    ];
    final ubuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'ub$i', width: 16),
    ];
    final hbuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'hb$i', width: 16),
    ];
    final sgbuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'sgb$i', width: 16),
    ];
    final subuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'sub$i', width: 16),
    ];
    final sdbuf = [
      for (var i = 0; i < hidden; i++) Logic(name: 'sdb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 3));
    final isDown = phase.eq(Const(2, width: 2));
    final isGate = phase.eq(Const(0, width: 2));

    final nActs = mux(
      isDown,
      Const(iSize, width: 16),
      Const(hidden, width: 16),
    );
    final nScales = mux(
      isDown,
      Const(hidden, width: 16),
      Const(iSize, width: 16),
    );
    final feedLen = mux(nActs.gt(nScales), nActs, nScales);

    // Reused matmul engine.
    final fpl = LoomFpLinear(
      maxColTiles: maxTiles,
      maxRowBlocks: maxBlocks,
      recipIterations: recipIterations,
    );
    fpl.input('clk').srcConnection! <= clk;
    fpl.input('reset').srcConnection! <= reset;
    fpl.input('start').srcConnection! <= st(_fire);
    fpl.input('col_tiles').srcConnection! <=
        mux(isDown, Const(ctD, width: 16), Const(ctGU, width: 16));
    fpl.input('row_blocks').srcConnection! <=
        mux(isDown, Const(rbD, width: 16), Const(rbGU, width: 16));
    fpl.input('weight_base').srcConnection! <=
        mux(
          isDown,
          input('wb_down'),
          mux(isGate, input('wb_gate'), input('wb_up')),
        );
    fpl.input('x_en').srcConnection! <= st(_feed) & fk.lt(nActs);
    // gate/up read x. Down reads the SiLU(gate)*up product h.
    fpl.input('x_in').srcConnection! <=
        mux(isDown, _sel(fk, hbuf), _sel(fk, xbuf));
    fpl.input('rs_en').srcConnection! <= st(_feed) & fk.lt(nScales);
    fpl.input('rs_in').srcConnection! <=
        mux(
          isDown,
          _sel(fk, sdbuf),
          mux(isGate, _sel(fk, sgbuf), _sel(fk, subuf)),
        );

    mem.cyc <= fpl.output('mem_CYC');
    mem.stb <= fpl.output('mem_STB');
    mem.we <= fpl.output('mem_WE');
    mem.adr <= fpl.output('mem_ADR');
    mem.datMosi <= fpl.output('mem_DAT_MOSI');
    mem.sel <= fpl.output('mem_SEL');
    fpl.input('mem_ACK').srcConnection! <= mem.ack;
    fpl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    final flY = fpl.output('y');
    final flYValid = fpl.output('y_valid');
    final flDone = fpl.output('done');

    // Elementwise SiLU(gate[ai]) * up[ai] during ACT.
    final silu = LoomSiLU();
    silu.input('x').srcConnection! <= _sel(ai, gbuf);
    final hProd = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(silu.output('y')),
      FloatingPoint16()..gets(_sel(ai, ubuf)),
    ).product.packed;

    // down results stream straight out. Gate/up results are buffered.
    oP <= flY;
    oValidP <= st(_run) & isDown & flYValid;
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < hidden; i++)
        If(
          st(_load) & input('x_en') & xc.eq(Const(i, width: 16)),
          then: [xbuf[i] < input('x_in')],
        ),
      for (var i = 0; i < iSize; i++)
        If(
          st(_load) & input('sg_en') & sgc.eq(Const(i, width: 16)),
          then: [sgbuf[i] < input('sg_in')],
        ),
      for (var i = 0; i < iSize; i++)
        If(
          st(_load) & input('su_en') & suc.eq(Const(i, width: 16)),
          then: [subuf[i] < input('su_in')],
        ),
      for (var i = 0; i < hidden; i++)
        If(
          st(_load) & input('sd_en') & sdc.eq(Const(i, width: 16)),
          then: [sdbuf[i] < input('sd_in')],
        ),
    ];
    // Buffer gate/up matmul results by row index.
    final resWrites = <Conditional>[
      for (var i = 0; i < iSize; i++)
        If(
          st(_run) & isGate & flYValid & ry.eq(Const(i, width: 16)),
          then: [gbuf[i] < flY],
        ),
      for (var i = 0; i < iSize; i++)
        If(
          st(_run) &
              phase.eq(Const(1, width: 2)) &
              flYValid &
              ry.eq(Const(i, width: 16)),
          then: [ubuf[i] < flY],
        ),
    ];
    // Elementwise write during ACT.
    final actWrites = <Conditional>[
      for (var i = 0; i < iSize; i++)
        If(st(_act) & ai.eq(Const(i, width: 16)), then: [hbuf[i] < hProd]),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 3),
          phase < Const(0, width: 2),
          xc < Const(0, width: 16),
          sgc < Const(0, width: 16),
          suc < Const(0, width: 16),
          sdc < Const(0, width: 16),
          fk < Const(0, width: 16),
          ry < Const(0, width: 16),
          ai < Const(0, width: 16),
          for (final b in xbuf) b < Const(0, width: 16),
          for (final b in gbuf) b < Const(0, width: 16),
          for (final b in ubuf) b < Const(0, width: 16),
          for (final b in hbuf) b < Const(0, width: 16),
          for (final b in sgbuf) b < Const(0, width: 16),
          for (final b in subuf) b < Const(0, width: 16),
          for (final b in sdbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...resWrites,
          ...actWrites,
          Case(state, [
            CaseItem(Const(_load, width: 3), [
              If(input('x_en'), then: [xc < xc + one]),
              If(input('sg_en'), then: [sgc < sgc + one]),
              If(input('su_en'), then: [suc < suc + one]),
              If(input('sd_en'), then: [sdc < sdc + one]),
              If(
                start,
                then: [
                  phase < Const(0, width: 2),
                  fk < Const(0, width: 16),
                  state < Const(_feed, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_feed, width: 3), [
              If(
                fk.eq(feedLen),
                then: [
                  ry < Const(0, width: 16),
                  state < Const(_fire, width: 3),
                ],
                orElse: [fk < fk + one],
              ),
            ]),
            CaseItem(Const(_fire, width: 3), [state < Const(_run, width: 3)]),
            CaseItem(Const(_run, width: 3), [
              If(flYValid, then: [ry < ry + one]),
              If(
                flDone,
                then: [
                  If(
                    isGate,
                    then: [
                      phase < Const(1, width: 2),
                      fk < Const(0, width: 16),
                      state < Const(_feed, width: 3),
                    ],
                  ),
                  If(
                    phase.eq(Const(1, width: 2)),
                    then: [
                      ai < Const(0, width: 16),
                      state < Const(_act, width: 3),
                    ],
                  ),
                  If(isDown, then: [state < Const(_fin, width: 3)]),
                ],
              ),
            ]),
            CaseItem(Const(_act, width: 3), [
              If(
                ai.eq(Const(iSize - 1, width: 16)),
                then: [
                  phase < Const(2, width: 2),
                  fk < Const(0, width: 16),
                  state < Const(_feed, width: 3),
                ],
                orElse: [ai < ai + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 3), [
              state < Const(_load, width: 3),
              xc < Const(0, width: 16),
              sgc < Const(0, width: 16),
              suc < Const(0, width: 16),
              sdc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
