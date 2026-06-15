import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'act_quant.dart';
import 'dequant.dart';
import 'stream_matmul.dart';

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

Logic _sel(List<Logic> regs, Logic idx) {
  if (regs.length == 1) return regs[0];
  final need = (regs.length - 1).bitLength;
  return _muxTree(idx.getRange(0, need), regs);
}

/// Replace byte [p] (0..3) of the 32-bit [word] with [b] (8-bit).
Logic _replaceByte(Logic word, int p, Logic b) {
  final parts = <Logic>[];
  if (p < 3) parts.add(word.getRange((p + 1) * 8, 32)); // high
  parts.add(b);
  if (p > 0) parts.add(word.getRange(0, p * 8)); // low
  return parts.swizzle();
}

/// LoomFpLinear: an fp16-in / fp16-out W4A8 linear layer, the reusable compute
/// brick the transformer sequencer instantiates for every projection and MLP
/// matrix. It brackets the verified integer streaming matmul with the fp16<->int
/// quant boundary units:
///
///   1. buffer the fp16 activation vector + the per-row fp16 weight scales
///      (streamed in by the sequencer/host),
///   2. LoomActQuant (two passes over the buffer) -> int8 activations packed
///      into the matmul's `acts_ext` registers + the fp16 dequant scale,
///   3. LoomStreamMatmul (int4 weights from `mem`, acts from registers, emitAcc)
///      -> int32 accumulators per row block,
///   4. LoomDequant each row (acc * rowScale * actScale) -> fp16 result stream.
///
/// Because the matmul takes activations from registers (actsExternal), this
/// module needs only ONE Wishbone master (`mem`, for weights), no act-staging
/// master and no arbiter. Weights live wherever `mem` decodes to (SRAM
/// scratchpad or DDR). Buffers are flop register files sized by maxColTiles/
/// maxRowBlocks. At full SmolLM2 dims they belong in BRAM.
///
/// Protocol: stream the fp16 activations (`x_en`/`x_in`, col_tiles*2 values) and
/// the fp16 row scales (`rs_en`/`rs_in`, row_blocks*2 values), set `col_tiles`/
/// `row_blocks`/`weight_base`, pulse `start`. Results arrive on `y`/`y_valid`
/// in row order; `done` pulses at the end.
class LoomFpLinear extends BridgeModule {
  static const int _load = 0;
  static const int _qmax = 1;
  static const int _qcomp = 2;
  static const int _qwait = 3;
  static const int _qout = 4;
  static const int _mm = 5;
  static const int _done = 6;

  late final WishboneInterface mem;

  LoomFpLinear({
    int maxColTiles = 4,
    int maxRowBlocks = 4,
    int recipIterations = 4,
    bool ternaryWeights = false,
    Logic? hostActScale,
    String? name,
  }) : super('LoomFpLinear', name: name ?? 'loom_fp_linear') {
    const peR = 2;
    const accW = 32;
    const aw = 32;
    final maxCols = maxColTiles * 2;
    final maxRows = maxRowBlocks * 2;
    final xWordCap = (maxColTiles + 1) ~/ 2;

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
    // Host act-scale override: OPTIONAL. Only wired up (a real input port
    // created) when a caller passes a signal (col-tiling: one shared act-scale
    // across every col-block's device op). When nonzero, it is fed into
    // LoomActQuant itself as ITS scale override, so the int8 acts are
    // quantized on the SAME shared grid as the dequant multiplier (both come
    // out of aqScale below). Quantizing on a different grid than the dequant
    // multiplier would corrupt the col-tiled partial sums. Callers that pass
    // nothing get no port at all; actScaleReg is driven straight from aqScale.
    Logic? hostActScaleIn;
    if (hostActScale != null) {
      createPort('host_act_scale', PortDirection.input, width: 16);
      input('host_act_scale').srcConnection! <= hostActScale;
      hostActScaleIn = input('host_act_scale');
    }
    final yP = addOutput('y', width: 16);
    final yValidP = addOutput('y_valid');
    final doneP = addOutput('done');
    final busyP = addOutput('busy');
    // fp32 pre-narrow dequant product, aligned to y/y_valid on the same
    // valid_out cycle (LoomDequant already aligns y_acc with y internally).
    final yAccP = addOutput('y_acc', width: 32);

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
    final ci = Logic(name: 'ci', width: 16); // col index (load + quant passes)
    final pcnt = Logic(name: 'pcnt', width: 16); // QOUT pack index (on q_valid)
    final ri = Logic(name: 'ri', width: 16); // row-scale load index
    final colTilesReg = Logic(name: 'col_tiles_reg', width: 16);
    final rowBlocksReg = Logic(name: 'row_blocks_reg', width: 16);
    final weightBaseReg = Logic(name: 'weight_base_reg', width: aw);
    final actScaleReg = Logic(name: 'act_scale_reg', width: 16);
    final mmStarted = Logic(name: 'mm_started');
    final doneLatch = Logic(name: 'done_latch');
    final pending1 = Logic(name: 'pending1'); // row 1 of the current block
    final blkReg = Logic(name: 'blk_reg', width: 16);
    final inflight = Logic(
      name: 'inflight',
      width: 8,
    ); // rows in the dequant pipe

    final xfBuf = [
      for (var i = 0; i < maxCols; i++) Logic(name: 'xf$i', width: 16),
    ];
    final rsBuf = [
      for (var i = 0; i < maxRows; i++) Logic(name: 'rs$i', width: 16),
    ];
    final actWords = [
      for (var i = 0; i < xWordCap; i++) Logic(name: 'aw$i', width: 32),
    ];

    Logic st(int s) => state.eq(Const(s, width: 3));
    final colCount = colTilesReg << 1; // padded cols = col_tiles * 2
    final atLastCol = ci.eq((colCount - Const(1, width: 16)).getRange(0, 16));

    // hostActScaleIn (when wired) is passed straight through as the quantizer's
    // OWN scale override: every col-block must quantize its int8 acts on the
    // SAME shared grid the dequant uses, not its own local max-abs. A per-block
    // quant scale paired with a shared dequant scale corrupts the col-tiled
    // partial sums. aqScale below equals the override whenever one is active,
    // so int8 acts + dequant agree.
    final aq = LoomActQuant(
      recipIterations: recipIterations,
      scaleOverride: hostActScaleIn,
    );
    aq.input('clk').srcConnection! <= clk;
    aq.input('reset').srcConnection! <= reset | start; // fresh per linear
    // MAX feeds all cols. QOUT feeds while ci < colCount (then drains the
    // pipelined quantizer). The quant multiply is clocked, so q_out/q_valid lag
    // x_in. Pack on q_valid via pcnt rather than the feed index.
    final ciLtCol = ci.lt(colCount);
    final feeding = st(_qmax) | (st(_qout) & ciLtCol);
    aq.input('x_en').srcConnection! <= feeding;
    aq.input('x_in').srcConnection! <= _sel(xfBuf, ci);
    aq.input('compute').srcConnection! <= st(_qcomp);
    final aqReady = aq.output('ready');
    final aqScale = aq.output('scale_out');
    final aqQ = aq.output('q_out');
    final aqQValid = aq.output('q_valid');

    final mm = LoomStreamMatmul(
      config: LoomStreamMatmulConfig(
        int4Weights: true,
        emitAcc: true,
        actsExternal: true,
        requantLatency: 0,
        maxColTiles: maxColTiles,
        maxRowBlocks: maxRowBlocks,
        ternaryWeights: ternaryWeights,
      ),
    );
    mm.input('clk').srcConnection! <= clk;
    mm.input('reset').srcConnection! <= reset;
    mm.input('start').srcConnection! <= st(_mm) & ~mmStarted;
    mm.input('row_blocks').srcConnection! <= rowBlocksReg;
    mm.input('col_tiles').srcConnection! <= colTilesReg;
    mm.input('shift').srcConnection! <= Const(0, width: 6);
    mm.input('weight_base').srcConnection! <= weightBaseReg;
    mm.input('act_base').srcConnection! <= Const(0, width: aw);
    mm.input('mult_base').srcConnection! <= Const(0, width: aw);
    mm.input('acts_ext').srcConnection! <= actWords.reversed.toList().swizzle();

    // Forward the matmul's master to the external 'mem' interface.
    mem.cyc <= mm.output('bus_CYC');
    mem.stb <= mm.output('bus_STB');
    mem.we <= mm.output('bus_WE');
    mem.adr <= mm.output('bus_ADR');
    mem.datMosi <= mm.output('bus_DAT_MOSI');
    mem.sel <= mm.output('bus_SEL');
    mm.input('bus_ACK').srcConnection! <= mem.ack;
    mm.input('bus_DAT_MISO').srcConnection! <= mem.datMiso;

    final mmResultValid = mm.output('result_valid');
    final mmResultBlock = mm.output('result_block');
    final mmResultAcc = mm.output('result_acc'); // peR*accW = 64
    final mmDone = mm.output('done');

    // Feed both rows of each result block into the dequant pipe in order (row 0
    // on the result_valid cycle, row 1 the next cycle via pending1). Collect
    // them `valid_out` cycles later. result_acc holds stable until the next
    // block, so row 1's accumulator is still valid on the pending1 cycle.
    final feed = st(_mm) & (mmResultValid | pending1);
    final rowIdx = mux(
      pending1,
      ((blkReg << 1) | Const(1, width: 16)).getRange(0, 16),
      (mmResultBlock << 1).getRange(0, 16),
    );
    final accSel = mux(
      pending1,
      mmResultAcc.getRange(accW, 2 * accW),
      mmResultAcc.getRange(0, accW),
    );
    final dq = LoomDequant();
    dq.input('clk').srcConnection! <= clk;
    dq.input('reset').srcConnection! <= reset;
    dq.input('valid_in').srcConnection! <= feed;
    dq.input('acc').srcConnection! <= accSel;
    dq.input('row_scale').srcConnection! <= _sel(rsBuf, rowIdx);
    dq.input('act_scale').srcConnection! <= actScaleReg;
    final dqValidOut = dq.output('valid_out');

    // Pack on the quantizer's q_valid (delayed by the clocked multiply), at the
    // pack index pcnt (not the feed index ci).
    final packWrites = <Conditional>[
      for (var wi = 0; wi < xWordCap; wi++)
        for (var p = 0; p < 4; p++)
          If(
            st(_qout) & aqQValid & pcnt.eq(Const(wi * 4 + p, width: 16)),
            then: [actWords[wi] < _replaceByte(actWords[wi], p, aqQ)],
          ),
    ];

    final xLoadWrites = <Conditional>[
      for (var i = 0; i < maxCols; i++)
        If(
          st(_load) & input('x_en') & ci.eq(Const(i, width: 16)),
          then: [xfBuf[i] < input('x_in')],
        ),
    ];
    final rsLoadWrites = <Conditional>[
      for (var i = 0; i < maxRows; i++)
        If(
          st(_load) & input('rs_en') & ri.eq(Const(i, width: 16)),
          then: [rsBuf[i] < input('rs_in')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 3),
          ci < Const(0, width: 16),
          ri < Const(0, width: 16),
          colTilesReg < Const(0, width: 16),
          rowBlocksReg < Const(0, width: 16),
          weightBaseReg < Const(0, width: aw),
          actScaleReg < Const(0, width: 16),
          mmStarted < Const(0),
          doneLatch < Const(0),
          pending1 < Const(0),
          blkReg < Const(0, width: 16),
          inflight < Const(0, width: 8),
          pcnt < Const(0, width: 16),
          for (final w in actWords) w < Const(0, width: 32),
        ],
        orElse: [
          ...xLoadWrites,
          ...rsLoadWrites,
          ...packWrites,
          Case(state, [
            CaseItem(Const(_load, width: 3), [
              If(input('x_en'), then: [ci < ci + Const(1, width: 16)]),
              If(input('rs_en'), then: [ri < ri + Const(1, width: 16)]),
              If(
                start,
                then: [
                  colTilesReg < input('col_tiles'),
                  rowBlocksReg < input('row_blocks'),
                  weightBaseReg < input('weight_base'),
                  ci < Const(0, width: 16),
                  mmStarted < Const(0),
                  doneLatch < Const(0),
                  pending1 < Const(0),
                  inflight < Const(0, width: 8),
                  state < Const(_qmax, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_qmax, width: 3), [
              If(
                atLastCol,
                then: [
                  ci < Const(0, width: 16),
                  state < Const(_qcomp, width: 3),
                ],
                orElse: [ci < ci + Const(1, width: 16)],
              ),
            ]),
            CaseItem(Const(_qcomp, width: 3), [
              state < Const(_qwait, width: 3),
            ]),
            CaseItem(Const(_qwait, width: 3), [
              If(
                aqReady,
                then: [
                  // aqScale IS the override already (LoomActQuant folds the
                  // shared-scale override into its own scale_out when active,
                  // and falls back to its on-chip max-abs scale otherwise), so
                  // no separate mux is needed here.
                  actScaleReg < aqScale,
                  ci < Const(0, width: 16),
                  pcnt < Const(0, width: 16),
                  state < Const(_qout, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_qout, width: 3), [
              // Feed while ci < colCount. Pack on q_valid via pcnt. Advance to
              // MM once the last quantized element is packed (pipeline drained).
              If(ciLtCol, then: [ci < ci + Const(1, width: 16)]),
              If(
                aqQValid,
                then: [
                  pcnt < pcnt + Const(1, width: 16),
                  If(
                    pcnt.eq((colCount - Const(1, width: 16)).getRange(0, 16)),
                    then: [state < Const(_mm, width: 3)],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_mm, width: 3), [
              mmStarted < Const(1),
              If(mmDone, then: [doneLatch < Const(1)]),
              If(
                mmResultValid,
                then: [blkReg < mmResultBlock, pending1 < Const(1)],
                orElse: [
                  If(pending1, then: [pending1 < Const(0)]),
                ],
              ),
              // Track rows in the dequant pipe: +1 per feed, -1 per valid_out.
              inflight <
                  (inflight +
                          mux(feed, Const(1, width: 8), Const(0, width: 8))) -
                      mux(dqValidOut, Const(1, width: 8), Const(0, width: 8)),
              // Finish only when the matmul is done AND the pipe is drained.
              If(
                doneLatch &
                    ~feed &
                    ~mmResultValid &
                    inflight.eq(Const(0, width: 8)) &
                    ~dqValidOut,
                then: [state < Const(_done, width: 3)],
              ),
            ]),
            CaseItem(Const(_done, width: 3), [
              ci < Const(0, width: 16),
              ri < Const(0, width: 16),
              state < Const(_load, width: 3),
            ]),
          ]),
        ],
      ),
    ]);

    yP <= dq.output('y');
    yAccP <= dq.output('y_acc'); // same valid_out cycle as y, fp32 precision
    yValidP <= dqValidOut; // results stream out of the dequant pipe, in order
    doneP <= st(_done);
    busyP <= ~st(_load);
  }
}
