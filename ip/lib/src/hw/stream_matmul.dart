// LoomStreamMatmul: a Wishbone-master int8 matmul that streams its weight
// tiles out of memory, so the inner/outer dimensions are NOT capped by on-chip
// register files (unlike LoomAccelerator's 8x8). This is the scale-up path
// toward running SmolLM2-sized linear layers with weights in DDR3 (staged
// through an on-chip SRAM scratchpad).
//
// Fixed 2x2 PE tile for v0: each weight tile is exactly one 32-bit word, which
// keeps the streaming bookkeeping simple. Activations and per-row requant
// multipliers are small, so they are read once into registers. The (large)
// weight matrix is streamed per row-block and fed straight into LoomLinear as
// each word arrives. Wider PEs and SRAM-backed activations come later.
//
// Memory layout (all little-endian, int8 packed 4/word):
//   weights : TILE-MAJOR. tile (rb, ct) at weightBase + (rb*colTiles + ct)*4,
//             one word = {w[1,1],w[1,0],w[0,1],w[0,0]} (byte0 = w[0,0]).
//   acts    : x[0..cols-1] packed 4/word at actBase. cols = colTiles*2.
//   mults   : uint16 per row, 2/word at multBase. word rb = {mult[2rb+1],mult[2rb]}.
// The host zero-pads rows/cols up to multiples of 2. Only the real rows matter.
//
// Result: each row-block's two int8 outputs are emitted on `result`
// (peRows*outWidth) with a one-cycle `result_valid` and the block index on
// `result_block`. `done` pulses after the last block.

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'linear.dart';

/// Selects `entries[index]` with a BALANCED binary mux tree (depth
/// ~ceil(log2(entries.length))), instead of a linear `mux` fold (depth N which
/// becomes the critical path). `index` is consumed LSB-first per level.
Logic _muxTree(Logic index, List<Logic> entries) {
  if (entries.isEmpty) return Const(0, width: 32);
  if (entries.length == 1) return entries[0];
  final next = <Logic>[];
  for (var i = 0; i < entries.length; i += 2) {
    if (i + 1 < entries.length) {
      next.add(mux(index[0], entries[i + 1], entries[i]));
    } else {
      next.add(entries[i]);
    }
  }
  return _muxTree(index.getRange(1, index.width), next);
}

class LoomStreamMatmulConfig {
  final int peRows;
  final int peCols;
  final int inWidth;
  final int accWidth;
  final int multWidth;
  final int shiftWidth;
  final int outWidth;
  final int addressWidth;
  final int peLatency;
  final int requantLatency;

  /// Register-array capacity (v0 holds activations + mults on chip).
  final int maxColTiles;
  final int maxRowBlocks;

  /// W4A8: weights are signed int4 packed 2-per-byte in memory (8 per 32-bit
  /// word = TWO 2x2 tiles), sign-extended to int8 on the fly before the int8
  /// PE. Halves weight storage so a 135M-param model fits 128MB DDR3.
  /// Activations stay int8. When false, weights are int8 (1 tile/word).
  final bool int4Weights;

  /// When true, emit the raw signed int32 accumulators on `result_acc`
  /// (peRows*accWidth) alongside `result_valid`, instead of relying only on the
  /// requantized int8 `result`. The fp16 W4A8 path (LoomFpLinear) dequantizes
  /// these directly. REQUIRES requantLatency == 0 so the captured accumulator is
  /// aligned with `result_valid` (the int8 requant tap lags by requantLatency).
  final bool emitAcc;

  /// When true, activations are NOT loaded from memory. Instead they are driven
  /// on the `acts_ext` input (xWordCap*32, int8 packed 4/word, same layout the
  /// memory load used). LoomFpLinear quantizes fp16 activations into these
  /// registers, so the matmul never needs a separate act-staging master. The
  /// `_phLoadX` phase is skipped. With emitAcc the `_phLoadM` (int8 requant
  /// multiplier) load is skipped too, since the fp path dequantizes instead.
  final bool actsExternal;

  /// Multiply-free BitNet path: weights are ternary {-1,0,+1} (still int4-packed
  /// in memory, ternary is a subset of int4), so the inner PE uses select/negate
  /// instead of a DSP multiply. Forwarded to LoomLinear -> LoomMatmul -> PE.
  final bool ternaryWeights;

  const LoomStreamMatmulConfig({
    this.peRows = 2,
    this.peCols = 2,
    this.inWidth = 8,
    this.accWidth = 32,
    this.multWidth = 16,
    this.shiftWidth = 6,
    this.outWidth = 8,
    this.addressWidth = 32,
    this.peLatency = 2,
    this.requantLatency = 2,
    this.maxColTiles = 64,
    this.maxRowBlocks = 64,
    this.int4Weights = false,
    this.emitAcc = false,
    this.actsExternal = false,
    this.ternaryWeights = false,
  });

  int get xWordCap => (maxColTiles + 1) ~/ 2; // 2 cols/tile, 4 cols/word

  void validate() {
    if (peRows != 2 || peCols != 2) {
      throw ArgumentError('LoomStreamMatmul v0 requires peRows==peCols==2');
    }
    if (inWidth != 8 || multWidth != 16) {
      throw ArgumentError(
        'LoomStreamMatmul v0 requires inWidth=8, multWidth=16',
      );
    }
  }
}

// Phase encoding.
const _phIdle = 0;
const _phLoadX = 1;
const _phLoadM = 2;
const _phRowIssue = 3;
const _phRowStream = 4;
const _phRowCap = 5;
const _phDone = 6;

class LoomStreamMatmul extends BridgeModule {
  final LoomStreamMatmulConfig config;
  late final WishboneInterface bus;

  LoomStreamMatmul({required this.config, String? name})
    : super('LoomStreamMatmul', name: name ?? 'loom_stream_matmul') {
    config.validate();
    final cfg = config;
    final aw = cfg.addressWidth;
    if (cfg.emitAcc && cfg.requantLatency != 0) {
      throw ArgumentError(
        'LoomStreamMatmul.emitAcc requires requantLatency == 0 so the captured '
        'int32 accumulator stays aligned with result_valid '
        '(got requantLatency=${cfg.requantLatency})',
      );
    }

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('row_blocks', PortDirection.input, width: 16);
    createPort('col_tiles', PortDirection.input, width: 16);
    createPort('shift', PortDirection.input, width: cfg.shiftWidth);
    createPort('weight_base', PortDirection.input, width: aw);
    createPort('act_base', PortDirection.input, width: aw);
    createPort('mult_base', PortDirection.input, width: aw);
    if (cfg.actsExternal) {
      createPort('acts_ext', PortDirection.input, width: cfg.xWordCap * 32);
    }

    final resultOut = addOutput('result', width: cfg.peRows * cfg.outWidth);
    final resultBlock = addOutput('result_block', width: 16);
    final resultValid = addOutput('result_valid');
    final doneOut = addOutput('done');
    final busyOut = addOutput('busy');
    // Raw int32 accumulators per row block (fp16 W4A8 path). Only present when
    // emitAcc. Aligned with result_valid (requantLatency must be 0).
    final Logic? resultAccOut = cfg.emitAcc
        ? addOutput('result_acc', width: cfg.peRows * cfg.accWidth)
        : null;

    final busRef = addInterface(
      WishboneInterface(WishboneConfig(addressWidth: aw, dataWidth: 32)),
      name: 'bus',
      role: PairRole.provider,
    );
    bus = busRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');
    final rowBlocksIn = input('row_blocks');
    final colTilesIn = input('col_tiles');
    final shiftIn = input('shift');
    final weightBaseIn = input('weight_base');
    final actBaseIn = input('act_base');
    final multBaseIn = input('mult_base');

    // On-chip storage for activations + mults.
    final xWords = List.generate(
      cfg.xWordCap,
      (i) => Logic(name: 'xword_$i', width: 32),
    );
    final multWords = List.generate(
      cfg.maxRowBlocks,
      (i) => Logic(name: 'mword_$i', width: 32),
    );

    // FSM registers.
    final phase = Logic(name: 'phase', width: 3);
    final cyc = Logic(name: 'cyc');
    final stb = Logic(name: 'stb');
    final adr = Logic(name: 'adr', width: aw);
    final remaining = Logic(name: 'remaining', width: 16);
    final loadIdx = Logic(name: 'load_idx', width: 16);
    final rb = Logic(name: 'rb', width: 16);
    final ct = Logic(name: 'ct', width: 16);
    // int4 only: tphase=1 means "feed the high tile of the latched word next".
    // wReg holds the just-read word so the high tile survives the stb pause.
    final tphase = Logic(name: 'tphase');
    final wReg = Logic(name: 'w_reg', width: 32);
    final latchedRowBlocks = Logic(name: 'l_row_blocks', width: 16);
    final latchedColTiles = Logic(name: 'l_col_tiles', width: 16);
    final latchedShift = Logic(name: 'l_shift', width: cfg.shiftWidth);
    final wRowAddr = Logic(
      name: 'w_row_addr',
      width: aw,
    ); // base of current block
    final colBytes = Logic(name: 'col_bytes', width: aw); // colTiles*4 stride

    final ackNow = cyc & stb & bus.ack;
    final inStream = phase.eq(Const(_phRowStream, width: 3));

    // These index a flop array by a runtime value. Written as a balanced mux
    // TREE (depth ~log2(N)), NOT a linear fold (depth N): the linear version was
    // the critical path (a ~64-deep L6MUX21 cascade -> 24 MHz). For large N
    // these should ultimately be BRAM-backed. The tree closes timing meanwhile.
    // x tile for the current colTile: word (ct>>1), half by ct[0]. In
    // actsExternal mode the words come from the acts_ext port, not the
    // memory-loaded xWords registers.
    final actSrc = cfg.actsExternal
        ? [
            for (var i = 0; i < cfg.xWordCap; i++)
              input('acts_ext').getRange(i * 32, (i + 1) * 32),
          ]
        : xWords;
    final xWordSel = _muxTree(ct.getRange(1, 16), actSrc);
    final xTile = mux(
      ct[0],
      xWordSel.getRange(16, 32),
      xWordSel.getRange(0, 16),
    );

    // rowMult word for the current row-block.
    final rowMultSel = _muxTree(rb, multWords);

    // Unpack a 16-bit packed int4 tile (4 nibbles) into a 32-bit int8 wTile
    // (sign-extended), matching the int8 wTile byte layout w[r*peCols+c].
    Logic unpackTile(Logic half16) => [
      half16.getRange(12, 16).signExtend(8),
      half16.getRange(8, 12).signExtend(8),
      half16.getRange(4, 8).signExtend(8),
      half16.getRange(0, 4).signExtend(8),
    ].swizzle();

    // Feed cadence:
    //   int8: one tile per word, fed on each ack.
    //   int4: two tiles per word - the low tile on ack, the high tile the next
    //         cycle (tphase=1, bus paused). The weight source differs per phase.
    final Logic feed;
    final Logic wTileFeed;
    if (cfg.int4Weights) {
      final lowFeed = inStream & ~tphase & ackNow; // low tile, this word
      final highFeed = inStream & tphase; // high tile, latched word
      feed = lowFeed | highFeed;
      wTileFeed = mux(
        tphase,
        unpackTile(wReg.getRange(16, 32)),
        unpackTile(bus.datMiso.getRange(0, 16)),
      );
    } else {
      feed = inStream & ackNow;
      wTileFeed = bus.datMiso; // each weight word IS one int8 2x2 tile
    }
    final linValid = feed;
    final linFirst = feed & ct.eq(Const(0, width: 16));
    final linLast = feed & (ct + Const(1, width: 16)).eq(latchedColTiles);

    final linear = LoomLinear(
      clk: clk,
      reset: reset,
      wTile: wTileFeed,
      xTile: xTile,
      valid: linValid,
      first: linFirst,
      last: linLast,
      rowMult: rowMultSel,
      shift: latchedShift,
      peRows: cfg.peRows,
      peCols: cfg.peCols,
      inWidth: cfg.inWidth,
      accWidth: cfg.accWidth,
      multWidth: cfg.multWidth,
      shiftWidth: cfg.shiftWidth,
      outWidth: cfg.outWidth,
      peLatency: cfg.peLatency,
      requantLatency: cfg.requantLatency,
      ternaryWeights: cfg.ternaryWeights,
    );

    // Per-phase ack stores for the load phases.
    final loadXStores = <Conditional>[
      for (var i = 0; i < xWords.length; i++)
        If(loadIdx.eq(Const(i, width: 16)), then: [xWords[i] < bus.datMiso]),
    ];
    final loadMStores = <Conditional>[
      for (var i = 0; i < multWords.length; i++)
        If(loadIdx.eq(Const(i, width: 16)), then: [multWords[i] < bus.datMiso]),
    ];

    Conditional issueRead(Logic base, Logic count) => If(
      count.gt(Const(0, width: 16)),
      then: [
        adr < base,
        remaining < count,
        cyc < Const(1),
        stb < Const(1),
        loadIdx < Const(0, width: 16),
      ],
    );

    final lastOfBurst = remaining.eq(Const(1, width: 16));

    // Words to read per row block: int8 = colTiles (1 tile/word). int4 =
    // ceil(colTiles/2) (2 tiles/word).
    Logic ceilHalf(Logic v, int w) =>
        (v + Const(1, width: 16)).getRange(1, 16).zeroExtend(w);
    final wordsPerRow = cfg.int4Weights
        ? ceilHalf(latchedColTiles, 16)
        : latchedColTiles;
    final isLastTile = (ct + Const(1, width: 16)).eq(latchedColTiles);

    Sequential(clk, [
      If(
        reset,
        then: [
          phase < Const(_phIdle, width: 3),
          cyc < Const(0),
          stb < Const(0),
          adr < Const(0, width: aw),
          remaining < Const(0, width: 16),
          loadIdx < Const(0, width: 16),
          rb < Const(0, width: 16),
          ct < Const(0, width: 16),
          tphase < Const(0),
          wReg < Const(0, width: 32),
          latchedRowBlocks < Const(0, width: 16),
          latchedColTiles < Const(0, width: 16),
          latchedShift < Const(0, width: cfg.shiftWidth),
          wRowAddr < Const(0, width: aw),
          colBytes < Const(0, width: aw),
          resultOut < Const(0, width: cfg.peRows * cfg.outWidth),
          resultBlock < Const(0, width: 16),
          resultValid < Const(0),
          doneOut < Const(0),
          if (resultAccOut != null)
            resultAccOut < Const(0, width: cfg.peRows * cfg.accWidth),
          for (final w in xWords) w < Const(0, width: 32),
          for (final w in multWords) w < Const(0, width: 32),
        ],
        orElse: [
          resultValid < Const(0),
          doneOut < Const(0),
          Case(phase, [
            CaseItem(Const(_phIdle, width: 3), [
              If(
                start,
                then: [
                  latchedRowBlocks < rowBlocksIn,
                  latchedColTiles < colTilesIn,
                  latchedShift < shiftIn,
                  wRowAddr < weightBaseIn,
                  colBytes <
                      ((cfg.int4Weights
                              ? ceilHalf(colTilesIn, aw)
                              : colTilesIn.zeroExtend(aw)) <<
                          2),
                  rb < Const(0, width: 16),
                  if (cfg.actsExternal) ...[
                    // Acts already in registers (acts_ext). Skip the act load.
                    // Skip the int8 mult load too when dequantizing (emitAcc).
                    if (cfg.emitAcc)
                      phase < Const(_phRowIssue, width: 3)
                    else ...[
                      issueRead(multBaseIn, latchedRowBlocks),
                      phase < Const(_phLoadM, width: 3),
                    ],
                  ] else ...[
                    // Load activations: ceil(colTiles*2 / 4) = ceil(colTiles/2).
                    issueRead(
                      actBaseIn,
                      (colTilesIn + Const(1, width: 16))
                          .getRange(1, 16)
                          .zeroExtend(16),
                    ),
                    phase < Const(_phLoadX, width: 3),
                  ],
                ],
              ),
            ]),

            CaseItem(Const(_phLoadX, width: 3), [
              If(
                ackNow,
                then: [
                  ...loadXStores,
                  If(
                    lastOfBurst,
                    then: [
                      cyc < Const(0),
                      stb < Const(0),
                      issueRead(multBaseIn, latchedRowBlocks),
                      phase < Const(_phLoadM, width: 3),
                    ],
                    orElse: [
                      loadIdx < (loadIdx + Const(1, width: 16)),
                      adr < (adr + Const(4, width: aw)),
                      remaining < (remaining - Const(1, width: 16)),
                    ],
                  ),
                ],
              ),
            ]),

            CaseItem(Const(_phLoadM, width: 3), [
              If(
                ackNow,
                then: [
                  ...loadMStores,
                  If(
                    lastOfBurst,
                    then: [
                      cyc < Const(0),
                      stb < Const(0),
                      phase < Const(_phRowIssue, width: 3),
                    ],
                    orElse: [
                      loadIdx < (loadIdx + Const(1, width: 16)),
                      adr < (adr + Const(4, width: aw)),
                      remaining < (remaining - Const(1, width: 16)),
                    ],
                  ),
                ],
              ),
            ]),

            // Issue the weight burst for row-block `rb`.
            CaseItem(Const(_phRowIssue, width: 3), [
              ct < Const(0, width: 16),
              tphase < Const(0),
              adr < wRowAddr,
              remaining < wordsPerRow,
              cyc < Const(1),
              stb < Const(1),
              phase < Const(_phRowStream, width: 3),
            ]),

            // Stream weight tiles into LoomLinear.
            CaseItem(Const(_phRowStream, width: 3), [
              if (!cfg.int4Weights)
                // int8: one tile per word, fed on each ack.
                If(
                  ackNow,
                  then: [
                    If(
                      lastOfBurst,
                      then: [
                        cyc < Const(0),
                        stb < Const(0),
                        phase < Const(_phRowCap, width: 3),
                      ],
                      orElse: [
                        ct < (ct + Const(1, width: 16)),
                        adr < (adr + Const(4, width: aw)),
                        remaining < (remaining - Const(1, width: 16)),
                      ],
                    ),
                  ],
                )
              else
                // int4: low tile on ack (tphase 0), high tile next cycle
                // (tphase 1, bus paused). Stop as soon as ct reaches colTiles
                // (handles odd colTiles: the last word's high tile is unused).
                If(
                  ~tphase,
                  then: [
                    If(
                      ackNow,
                      then: [
                        wReg < bus.datMiso,
                        ct < (ct + Const(1, width: 16)),
                        If(
                          isLastTile,
                          then: [
                            cyc < Const(0),
                            stb < Const(0),
                            phase < Const(_phRowCap, width: 3),
                          ],
                          orElse: [tphase < Const(1), stb < Const(0)],
                        ),
                      ],
                    ),
                  ],
                  orElse: [
                    ct < (ct + Const(1, width: 16)),
                    tphase < Const(0),
                    If(
                      isLastTile,
                      then: [
                        cyc < Const(0),
                        stb < Const(0),
                        phase < Const(_phRowCap, width: 3),
                      ],
                      orElse: [
                        stb < Const(1),
                        adr < (adr + Const(4, width: aw)),
                        remaining < (remaining - Const(1, width: 16)),
                      ],
                    ),
                  ],
                ),
            ]),

            // Wait for the requantized row outputs, emit them.
            CaseItem(Const(_phRowCap, width: 3), [
              If(
                linear.outValid,
                then: [
                  resultOut < linear.out,
                  if (resultAccOut != null) resultAccOut < linear.acc,
                  resultBlock < rb,
                  resultValid < Const(1),
                  rb < (rb + Const(1, width: 16)),
                  wRowAddr < (wRowAddr + colBytes),
                  If(
                    (rb + Const(1, width: 16)).eq(latchedRowBlocks),
                    then: [phase < Const(_phDone, width: 3)],
                    orElse: [phase < Const(_phRowIssue, width: 3)],
                  ),
                ],
              ),
            ]),

            CaseItem(Const(_phDone, width: 3), [
              doneOut < Const(1),
              phase < Const(_phIdle, width: 3),
            ]),
          ]),
        ],
      ),
    ]);

    bus.cyc <= cyc;
    bus.stb <= stb;
    bus.we <= Const(0);
    bus.adr <= adr;
    bus.datMosi <= Const(0, width: 32);
    bus.sel <= Const(0xF, width: 4);
    busyOut <= ~phase.eq(Const(_phIdle, width: 3));
  }
}
