// LoomAccelerator: Wishbone-driven top-level int8 linear-layer accelerator.
//
// Wraps LoomLinear behind:
//   - On-chip buffers (weight, activation, rowMult, result)
//   - A CSR register block (VERSION, ROWS, COLS, SHIFT, CONTROL, STATUS)
//   - A sequencer FSM that streams tiles from the buffers into LoomLinear
//     and captures results back into the result buffer
//   - A Wishbone slave (harbor BridgeModule pattern)
//
// Address map (byte addresses, 12-bit bus):
//   CSR region (bits [11:8] == 0x0):
//     0x000  VERSION  RO  0x4C4F4F4D
//     0x004  ROWS     RW  runtime row count (1..maxRows)
//     0x008  COLS     RW  runtime col count (1..maxCols)
//     0x00C  SHIFT    RW  requant shift (0..63)
//     0x010  CONTROL  RW  bit 0: start (self-clears after one cycle)
//     0x014  STATUS   RO  bit 0: busy, bit 1: done (sticky until next start)
//
//   Buffer regions:
//     0x100..0x1FF  weight buffer    WO  int8 packed 4/word, row-major W[r,c]
//     0x200..0x2FF  activation buf   WO  int8 packed 4/word, x[c]
//     0x300..0x3FF  rowMult buffer   WO  uint16 packed 2/word, mult[r]
//     0x400..0x4FF  result buffer    RO  int8 packed 4/word, y[r]
//
// Zero-padding: when ROWS is not a multiple of peRows or COLS not a multiple
// of peCols, the last tile is padded with zeros. Zero elements contribute 0
// to the accumulator, so partial sums for in-bound elements are correct.
//
// Sequencer FSM states (seqState register):
//   IDLE(0)     -> on CONTROL.start: latch ROWS/COLS/SHIFT, set busy, rowBlk=0,
//                  go to ROW_INIT
//   ROW_INIT(1) -> set colTile=0, go to TILE
//   TILE(2)     -> linValid driven high combinationally from state. At posedge:
//                  increment colTile, go to WAIT
//   WAIT(3)     -> if colTile (post-increment) == numColTiles, go to CAPTURE_W;
//                  else go back to TILE
//   CAPTURE_W(4)-> wait for LoomLinear outValid (which fires 1 cycle after last
//                  tile posedge). Sample out into result buffer, advance rowBlk,
//                  go to DONE or ROW_INIT
//   DONE(5)     -> clear busy, set done, return to IDLE
//
// Signal timing:
//   linValid/linFirst/linLast are driven COMBINATIONALLY from seqState and
//   colTile (before the posedge increment). This ensures LoomLinear sees them
//   in the same cycle the state is TILE, matching the "tile at posedge N"
//   contract for outValid at posedge N+1.
//   In CAPTURE_W we check linear.outValid (combinational from LoomLinear).

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'linear.dart';

// Configuration

/// Immutable configuration for [LoomAccelerator].
class LoomAcceleratorConfig {
  /// Number of PE rows (output rows per LoomLinear invocation).
  final int peRows;

  /// Number of PE columns (inner-dim tile size).
  final int peCols;

  /// Maximum number of runtime rows supported by the weight buffer.
  final int maxRows;

  /// Maximum number of runtime cols supported by the activation buffer.
  final int maxCols;

  /// Bit width of each signed input element (W and x).
  final int inWidth;

  /// Bit width of the signed accumulator.
  final int accWidth;

  /// Bit width of the unsigned per-row requant multiplier.
  final int multWidth;

  /// Bit width of the shared requant shift.
  final int shiftWidth;

  /// Bit width of the saturated signed output per row.
  final int outWidth;

  /// Base address in the SoC memory map.
  final int baseAddress;

  /// Version/magic constant returned from VERSION register.
  final int versionMagic;

  /// Human-readable model identity baked into the silicon and reported over the
  /// bus (MODEL_LEN CSR + NAME region). The runtime reads this so it never has
  /// to hardcode which model the device is, e.g. 'SmolLM2-135M'. ASCII only.
  final String modelName;

  /// Maximum baked model-name length in bytes (NAME region capacity).
  static const int maxModelNameBytes = 64;

  /// Loom Nano: order-N binary on-chip inference. When > 0, the accelerator
  /// bakes a tiny binary Markov LM (context = last [nanoContextBits] bits) and
  /// exposes an autoregressive step interface: the device computes
  /// argmax(W . onehot(context)) on chip and owns the context window, so the
  /// runtime stays model-agnostic (it just steps and reads tokens). 0 disables.
  final int nanoContextBits;

  /// Baked Nano weights, row-major `[vocab=2][nStates=2^nanoContextBits]`,
  /// int8. Empty when Nano is disabled.
  final List<int> nanoWeights;

  /// Number of Nano context states.
  int get nanoStates => 1 << nanoContextBits;

  /// Pipeline latency (clock cycles) of the inner PE array. Pipelining the
  /// signed int8 multiply -> adder tree is what lets the design close 48 MHz on
  /// the OrangeCrab ECP5 (the combinational path was ~22 MHz). The sequencer
  /// polls LoomLinear.outValid, so any latency is tolerated transparently.
  final int peLatency;

  /// Pipeline latency (clock cycles, 0..2) of the per-row requantize. The wide
  /// requant multiply + round/saturate chain became the critical path once the
  /// PE array was pipelined. Stage 1 registers the product. Stage 2 registers
  /// the rounded value (off the saturation compares). latency 2 is what closes
  /// 48 MHz. Tolerated transparently by the polling FSM.
  final int requantLatency;

  const LoomAcceleratorConfig({
    required this.baseAddress,
    this.peRows = 2,
    this.peCols = 2,
    this.maxRows = 8,
    this.maxCols = 8,
    this.inWidth = 8,
    this.accWidth = 32,
    this.multWidth = 16,
    this.shiftWidth = 6,
    this.outWidth = 8,
    this.versionMagic = 0x4C4F4F4D,
    this.peLatency = 2,
    this.requantLatency = 2,
    this.modelName = 'loom',
    this.nanoContextBits = 0,
    this.nanoWeights = const [],
  });

  /// Number of row blocks needed to cover maxRows.
  int get maxRowBlocks => (maxRows + peRows - 1) ~/ peRows;

  /// Number of col tiles needed to cover maxCols.
  int get maxColTiles => (maxCols + peCols - 1) ~/ peCols;

  /// Validates the configuration. Throws [ArgumentError] on failure.
  void validate() {
    if (peRows <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.peRows must be > 0, got $peRows',
      );
    }
    if (peCols <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.peCols must be > 0, got $peCols',
      );
    }
    if (maxRows <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.maxRows must be > 0, got $maxRows',
      );
    }
    if (maxCols <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.maxCols must be > 0, got $maxCols',
      );
    }
    if (maxRows < peRows) {
      throw ArgumentError(
        'LoomAcceleratorConfig.maxRows ($maxRows) must be >= peRows ($peRows)',
      );
    }
    if (maxCols < peCols) {
      throw ArgumentError(
        'LoomAcceleratorConfig.maxCols ($maxCols) must be >= peCols ($peCols)',
      );
    }
    if (inWidth <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.inWidth must be > 0, got $inWidth',
      );
    }
    if (multWidth <= 0) {
      throw ArgumentError(
        'LoomAcceleratorConfig.multWidth must be > 0, got $multWidth',
      );
    }
    if (modelName.length > maxModelNameBytes) {
      throw ArgumentError(
        'LoomAcceleratorConfig.modelName must be <= $maxModelNameBytes bytes, '
        'got ${modelName.length}',
      );
    }
    for (final cu in modelName.codeUnits) {
      if (cu > 0x7F) {
        throw ArgumentError(
          'LoomAcceleratorConfig.modelName must be ASCII, got "$modelName"',
        );
      }
    }
    if (nanoContextBits < 0 || nanoContextBits > 8) {
      throw ArgumentError(
        'LoomAcceleratorConfig.nanoContextBits must be in 0..8, '
        'got $nanoContextBits',
      );
    }
    if (nanoContextBits > 0 && nanoWeights.length != 2 * nanoStates) {
      throw ArgumentError(
        'LoomAcceleratorConfig.nanoWeights must have ${2 * nanoStates} entries '
        '(2 x $nanoStates), got ${nanoWeights.length}',
      );
    }
  }
}

// Register map descriptor (for SVD)

const _loomAcceleratorRegisterMap = HarborDeviceRegisterMap(
  name: 'loom_accelerator',
  fields: [
    HarborDeviceField(
      name: 'VERSION',
      width: 4,
      offset: 0x00,
      readOnly: true,
      resetValue: 0x4C4F4F4D,
    ),
    HarborDeviceField(name: 'ROWS', width: 4, offset: 0x04),
    HarborDeviceField(name: 'COLS', width: 4, offset: 0x08),
    HarborDeviceField(name: 'SHIFT', width: 4, offset: 0x0C),
    HarborDeviceField(name: 'CONTROL', width: 4, offset: 0x10),
    HarborDeviceField(name: 'STATUS', width: 4, offset: 0x14, readOnly: true),
    HarborDeviceField(
      name: 'MODEL_LEN',
      width: 4,
      offset: 0x18,
      readOnly: true,
    ),
  ],
);

// CSR byte offsets.
const _regVersion = 0x000;
const _regRows = 0x004;
const _regCols = 0x008;
const _regShift = 0x00C;
const _regControl = 0x010;
const _regStatus = 0x014;
const _regModelLen = 0x018;

// Loom Nano autoregressive inference CSRs (CSR region, nibble 0x0).
//   0x020 INFER_RESET  W  any write clears the on-chip context + output
//   0x024 INFER_PUSH   W  shift dataIn[0] into the context (prompt priming)
//   0x028 INFER_STEP   W  generate one token on chip: next=argmax(W.onehot(ctx)),
//                          shift it into the context, latch it to INFER_OUT
//   0x02C INFER_OUT    RO bit0: last generated token. Bits[..]: current context
const _regInferReset = 0x020;
const _regInferPush = 0x024;
const _regInferStep = 0x028;
const _regInferOut = 0x02C;

// NAME region nibble (bits [11:8] of the byte address): 0x5xx, read-only,
// ASCII model name packed 4 chars/word (char 4*i in the word LSB).
const _regionName = 0x5;

// FSM state encoding.
const _stIdle = 0;
const _stRowInit = 1;
const _stTile = 2;
const _stWait = 3;
const _stCaptureW = 4;
const _stDone = 5;

// Module

/// Wishbone-driven int8 linear-layer accelerator.
///
/// See the file-level comment for the address map and FSM description.
class LoomAccelerator extends BridgeModule
    with HarborDeviceTreeNodeProvider, HarborSvdPeripheralProvider {
  /// Size of the slave address window in bytes.
  ///
  /// The bus is 12-bit (covers 0x000..0xFFF). The harbor Wishbone decoder gates
  /// each slave with `adr < Const(start+size, width: addressWidth)`, so the
  /// EXCLUSIVE end (start+size) MUST be representable in the bus address width.
  /// A full 4 KiB window (end 0x1000) wraps to 0x000 in 12 bits and collapses
  /// the decode to empty (the accelerator becomes unreachable). 0x800 (end
  /// 0x800) fits in 12 bits and still covers every real register (the highest
  /// used region, the result buffer, ends at 0x4FF).
  static const int windowSize = 0x800;

  /// Configuration.
  final LoomAcceleratorConfig config;

  /// Wishbone slave port.
  late final BusSlavePort bus;

  LoomAccelerator({
    required this.config,
    BusProtocol protocol = BusProtocol.wishbone,
    String? name,
  }) : super('LoomAccelerator', name: name ?? 'loom_accelerator') {
    config.validate();

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    // 12-bit address bus covers 0x000..0xFFF (4 KiB).
    bus = BusSlavePort.create(
      module: this,
      name: 'bus',
      protocol: protocol,
      addressWidth: 12,
      dataWidth: 32,
    );

    final clk = input('clk');
    final reset = input('reset');
    final cfg = config;

    // On-chip buffers as register arrays (32-bit words).
    final wWeightWords = (cfg.maxRows * cfg.maxCols + 3) ~/ 4;
    final wActWords = (cfg.maxCols + 3) ~/ 4;
    final wMultWords = (cfg.maxRows + 1) ~/ 2;
    final wResultWords = (cfg.maxRows + 3) ~/ 4;

    final weightBuf = List.generate(
      wWeightWords,
      (i) => Logic(name: 'wbuf_$i', width: 32),
    );
    final actBuf = List.generate(
      wActWords,
      (i) => Logic(name: 'abuf_$i', width: 32),
    );
    final multBuf = List.generate(
      wMultWords,
      (i) => Logic(name: 'mbuf_$i', width: 32),
    );
    final resultBuf = List.generate(
      wResultWords,
      (i) => Logic(name: 'rbuf_$i', width: 32),
    );

    // Baked model identity: MODEL_LEN + NAME region constant words.
    final nameBytes = cfg.modelName.codeUnits;
    final nameLen = nameBytes.length;
    final nameWordCount = ((nameLen + 3) ~/ 4).clamp(1, 16);
    final nameWords = List.generate(nameWordCount, (i) {
      var w = 0;
      for (var b = 0; b < 4; b++) {
        final idx = i * 4 + b;
        if (idx < nameLen) w |= (nameBytes[idx] & 0xFF) << (b * 8);
      }
      return Const(w, width: 32);
    });

    // CSR registers
    final rowsReg = Logic(name: 'rows_reg', width: 32);
    final colsReg = Logic(name: 'cols_reg', width: 32);
    final shiftReg = Logic(name: 'shift_reg', width: 32);
    final controlReg = Logic(name: 'control_reg', width: 32);
    final busyReg = Logic(name: 'busy_reg');
    final doneReg = Logic(name: 'done_reg');

    // Sequencer registers
    final seqState = Logic(name: 'seq_state', width: 3);
    final rowBlk = Logic(name: 'row_blk', width: 8);
    final colTile = Logic(name: 'col_tile', width: 8);
    final latchedRows = Logic(name: 'latched_rows', width: 32);
    final latchedCols = Logic(name: 'latched_cols', width: 32);
    final latchedShift = Logic(name: 'latched_shift', width: cfg.shiftWidth);

    // Combinational: numRowBlocks and numColTiles from latched values.
    //
    // numRowBlocks = max r such that (r-1)*peRows < latchedRows, clamp to
    // [1, maxRowBlocks]. Built as mux chain (lower r = default).
    Logic numRowBlocks = Const(1, width: 8);
    for (var r = 2; r <= cfg.maxRowBlocks; r++) {
      numRowBlocks = mux(
        latchedRows.gt(Const((r - 1) * cfg.peRows, width: 32)),
        Const(r, width: 8),
        numRowBlocks,
      );
    }

    Logic numColTiles = Const(1, width: 8);
    for (var c = 2; c <= cfg.maxColTiles; c++) {
      numColTiles = mux(
        latchedCols.gt(Const((c - 1) * cfg.peCols, width: 32)),
        Const(c, width: 8),
        numColTiles,
      );
    }

    // Combinational tile fetch helpers.
    Logic wBufByte(int byteIdx) {
      if (byteIdx >= wWeightWords * 4) return Const(0, width: 8);
      final wordIdx = byteIdx ~/ 4;
      final bitOff = (byteIdx % 4) * 8;
      return weightBuf[wordIdx].getRange(bitOff, bitOff + 8);
    }

    Logic aBufByte(int byteIdx) {
      if (byteIdx >= wActWords * 4) return Const(0, width: 8);
      final wordIdx = byteIdx ~/ 4;
      final bitOff = (byteIdx % 4) * 8;
      return actBuf[wordIdx].getRange(bitOff, bitOff + 8);
    }

    Logic mBufHalf(int rowIdx) {
      if (rowIdx >= wMultWords * 2) return Const(0, width: cfg.multWidth);
      final wordIdx = rowIdx ~/ 2;
      final bitOff = (rowIdx % 2) * 16;
      return multBuf[wordIdx].getRange(bitOff, bitOff + cfg.multWidth);
    }

    // Combinational tile buses.
    //
    // For each PE element, build a mux tree over all (rowBlk, colTile) values.
    // Out-of-bound rows/cols produce 0 (zero-padding).
    final wTileSlices = <Logic>[];
    final xTileSlices = <Logic>[];
    final rowMultSlices = <Logic>[];

    for (var peR = 0; peR < cfg.peRows; peR++) {
      for (var peC = 0; peC < cfg.peCols; peC++) {
        Logic result = Const(0, width: cfg.inWidth);
        for (var rb = 0; rb < cfg.maxRowBlocks; rb++) {
          for (var ct = 0; ct < cfg.maxColTiles; ct++) {
            final globalRow = rb * cfg.peRows + peR;
            final globalCol = ct * cfg.peCols + peC;
            final byteIdx = globalRow * cfg.maxCols + globalCol;
            if (byteIdx >= wWeightWords * 4) continue;

            final sel =
                rowBlk.eq(Const(rb, width: 8)) &
                colTile.eq(Const(ct, width: 8));
            final inBound =
                latchedRows.gt(Const(globalRow, width: 32)) &
                latchedCols.gt(Const(globalCol, width: 32));
            result = mux(
              sel,
              mux(inBound, wBufByte(byteIdx), Const(0, width: cfg.inWidth)),
              result,
            );
          }
        }
        wTileSlices.add(result);
      }
    }

    for (var peC = 0; peC < cfg.peCols; peC++) {
      Logic result = Const(0, width: cfg.inWidth);
      for (var ct = 0; ct < cfg.maxColTiles; ct++) {
        final globalCol = ct * cfg.peCols + peC;
        if (globalCol >= wActWords * 4) continue;

        final sel = colTile.eq(Const(ct, width: 8));
        final colInBound = latchedCols.gt(Const(globalCol, width: 32));
        result = mux(
          sel,
          mux(colInBound, aBufByte(globalCol), Const(0, width: cfg.inWidth)),
          result,
        );
      }
      xTileSlices.add(result);
    }

    for (var peR = 0; peR < cfg.peRows; peR++) {
      Logic result = Const(0, width: cfg.multWidth);
      for (var rb = 0; rb < cfg.maxRowBlocks; rb++) {
        final globalRow = rb * cfg.peRows + peR;
        if (globalRow >= cfg.maxRows) continue;

        final sel = rowBlk.eq(Const(rb, width: 8));
        final rowInBound = latchedRows.gt(Const(globalRow, width: 32));
        result = mux(
          sel,
          mux(rowInBound, mBufHalf(globalRow), Const(0, width: cfg.multWidth)),
          result,
        );
      }
      rowMultSlices.add(result);
    }

    // LoomLinear drive signals.
    //
    // linValid/linFirst/linLast are driven COMBINATIONALLY from seqState so
    // that LoomLinear samples them in the same cycle the FSM is in TILE state.
    // The mux-tree tile buses (wTileSlices etc.) are already combinational.
    //
    // linValid  = (seqState == TILE)
    // linFirst  = (seqState == TILE) && (colTile == 0)
    // linLast   = (seqState == TILE) && (colTile + 1 == numColTiles)
    //   (colTile here is the CURRENT value, before the posedge increment)
    final isTile = seqState.eq(Const(_stTile, width: 3));
    final linValid = isTile;
    final linFirst = isTile & colTile.eq(Const(0, width: 8));
    final linLast = isTile & (colTile + Const(1, width: 8)).eq(numColTiles);

    final linWTile = wTileSlices.reversed.toList().swizzle();
    final linXTile = xTileSlices.reversed.toList().swizzle();
    final linRowMult = rowMultSlices.reversed.toList().swizzle();

    final linear = LoomLinear(
      clk: clk,
      reset: reset,
      wTile: linWTile,
      xTile: linXTile,
      valid: linValid,
      first: linFirst,
      last: linLast,
      rowMult: linRowMult,
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
    );

    // Pre-build CAPTURE write conditionals.
    //
    // When outValid fires, we need to write all PE row outputs into the
    // result buffer simultaneously. The critical rule: multiple PE rows may
    // land in the same 32-bit result word (e.g. rows 0 and 1 both go into
    // word 0 at bytes 0 and 1). In ROHD Sequential, last-write-wins, and
    // each write reads the PRE-posedge value of resultBuf[wordIdx].
    //
    // Fix: group writes BY result word. For each (rb, wordIdx) combination,
    // compute a single newWord expression that incorporates ALL the PE rows
    // landing in that word at that rowBlk. Issue one assignment per word.
    final captureWrites = <Conditional>[];

    // Enumerate all (rb) values. For each rb, group (peR) by wordIdx.
    for (var rb = 0; rb < cfg.maxRowBlocks; rb++) {
      // Build map: wordIdx -> list of (peR, globalRow, byteOff)
      final wordToRows = <int, List<(int, int, int)>>{};
      for (var peR = 0; peR < cfg.peRows; peR++) {
        final globalRow = rb * cfg.peRows + peR;
        if (globalRow >= cfg.maxRows) continue;
        final wordIdx = globalRow ~/ 4;
        final byteOff = (globalRow % 4) * 8;
        wordToRows.putIfAbsent(wordIdx, () => []).add((
          peR,
          globalRow,
          byteOff,
        ));
      }

      // For each result word this rowBlk contributes to, build one write.
      // Read-modify-write the result buffer word so prior row blocks' data
      // in the same word is preserved. resultBuf[wordIdx] at this posedge
      // holds data from earlier row blocks (written in previous CAPTURE_W
      // cycles).
      for (final entry in wordToRows.entries) {
        final wordIdx = entry.key;
        final rows = entry.value;

        // Start from the EXISTING word (preserves earlier row block bytes).
        // Then OR in each new row's byte, masked to its lane.
        Logic newWord = resultBuf[wordIdx];
        for (final (peR, globalRow, byteOff) in rows) {
          final outByte = linear.out
              .getRange(peR * cfg.outWidth, (peR + 1) * cfg.outWidth)
              .zeroExtend(32);
          final shiftedByte = outByte << byteOff;
          // First, clear the byte lane for this row.
          final mask = 0xFFFFFFFF ^ (0xFF << byteOff);
          final cleared = newWord & Const(mask, width: 32);
          // Include this byte if globalRow < latchedRows.
          final rowInBound = latchedRows.gt(Const(globalRow, width: 32));
          newWord = cleared | mux(rowInBound, shiftedByte, Const(0, width: 32));
        }

        captureWrites.add(
          If(
            rowBlk.eq(Const(rb, width: 8)),
            then: [resultBuf[wordIdx] < newWord],
          ),
        );
      }
    }

    // Pre-build buffer write/read conditionals for bus handler.
    final weightWrites = <Conditional>[
      for (var i = 0; i < wWeightWords; i++)
        If(
          bus.addr.getRange(2, 8).eq(Const(i, width: 6)),
          then: [weightBuf[i] < bus.dataIn],
        ),
    ];

    final actWrites = <Conditional>[
      for (var i = 0; i < wActWords; i++)
        If(
          bus.addr.getRange(2, 8).eq(Const(i, width: 6)),
          then: [actBuf[i] < bus.dataIn],
        ),
    ];

    final multWrites = <Conditional>[
      for (var i = 0; i < wMultWords; i++)
        If(
          bus.addr.getRange(2, 8).eq(Const(i, width: 6)),
          then: [multBuf[i] < bus.dataIn],
        ),
    ];

    final resultReads = <Conditional>[
      for (var i = 0; i < wResultWords; i++)
        If(
          bus.addr.getRange(2, 8).eq(Const(i, width: 6)),
          then: [bus.dataOut < resultBuf[i]],
        ),
    ];

    // Loom Nano: on-chip autoregressive binary inference.
    //
    // The device owns the context window and computes the model forward
    // (argmax(W . onehot(ctx))) entirely on chip. The runtime only writes
    // RESET/PUSH/STEP and reads INFER_OUT, so it never knows the model.
    //
    // For a one-hot activation, W . onehot(ctx) == column `ctx` of W, so the
    // logits are a mux over the baked weights. Argmax over the two of them is
    // a signed compare. A one-hot activation degenerates the matmul to a
    // column select rather than exercising the full PE array.
    final nanoEnabled = cfg.nanoContextBits > 0;
    final nanoResets = <Conditional>[];
    final nanoCsrItems = <CaseItem>[];
    if (nanoEnabled) {
      final ctxW = cfg.nanoContextBits;
      final nStates = cfg.nanoStates;
      final nanoCtx = Logic(name: 'nano_ctx', width: ctxW);
      final nanoOut = Logic(name: 'nano_out');

      // Combinational logits = column `nanoCtx` of the baked weight matrix.
      Logic l0 = Const(0, width: 8);
      Logic l1 = Const(0, width: 8);
      for (var c = 0; c < nStates; c++) {
        final sel = nanoCtx.eq(Const(c, width: ctxW));
        l0 = mux(sel, Const(cfg.nanoWeights[c] & 0xFF, width: 8), l0);
        l1 = mux(sel, Const(cfg.nanoWeights[nStates + c] & 0xFF, width: 8), l1);
      }
      // argmax over two int8 logits: signed gt via MSB-flip + unsigned compare.
      final nanoNext = (l1 ^ Const(0x80, width: 8)).gt(
        l0 ^ Const(0x80, width: 8),
      );

      // Shift a new bit into the context (newest bit = LSB, oldest drops).
      Logic shiftIn(Logic bit) =>
          [nanoCtx.getRange(0, ctxW - 1), bit].swizzle();

      nanoResets
        ..add(nanoCtx < Const(0, width: ctxW))
        ..add(nanoOut < Const(0));

      nanoCsrItems
        ..add(
          CaseItem(Const(_regInferReset, width: 8), [
            If(
              bus.we,
              then: [
                nanoCtx < Const(0, width: ctxW),
                nanoOut < Const(0),
              ],
            ),
          ]),
        )
        ..add(
          CaseItem(Const(_regInferPush, width: 8), [
            If(bus.we, then: [nanoCtx < shiftIn(bus.dataIn[0])]),
          ]),
        )
        ..add(
          CaseItem(Const(_regInferStep, width: 8), [
            If(bus.we, then: [nanoCtx < shiftIn(nanoNext), nanoOut < nanoNext]),
          ]),
        )
        ..add(
          CaseItem(Const(_regInferOut, width: 8), [
            bus.dataOut <
                [Const(0, width: 32 - 1 - ctxW), nanoCtx, nanoOut].swizzle(),
          ]),
        );
    }

    // Sequential block: FSM + bus + buffer writes + result capture.
    Sequential(clk, [
      If(
        reset,
        then: [
          rowsReg < Const(0, width: 32),
          colsReg < Const(0, width: 32),
          shiftReg < Const(0, width: 32),
          controlReg < Const(0, width: 32),
          busyReg < Const(0),
          doneReg < Const(0),
          seqState < Const(_stIdle, width: 3),
          rowBlk < Const(0, width: 8),
          colTile < Const(0, width: 8),
          latchedRows < Const(0, width: 32),
          latchedCols < Const(0, width: 32),
          latchedShift < Const(0, width: cfg.shiftWidth),
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
          for (var i = 0; i < wWeightWords; i++)
            weightBuf[i] < Const(0, width: 32),
          for (var i = 0; i < wActWords; i++) actBuf[i] < Const(0, width: 32),
          for (var i = 0; i < wMultWords; i++) multBuf[i] < Const(0, width: 32),
          for (var i = 0; i < wResultWords; i++)
            resultBuf[i] < Const(0, width: 32),
          ...nanoResets,
        ],
        orElse: [
          // Defaults each cycle.
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),

          // Self-clear CONTROL.start one cycle after it was written.
          If(
            controlReg[0],
            then: [controlReg < (controlReg & Const(0xFFFFFFFE, width: 32))],
          ),

          // Sequencer FSM.
          Case(seqState, [
            // IDLE: wait for CONTROL.start.
            CaseItem(Const(_stIdle, width: 3), [
              If(
                controlReg[0],
                then: [
                  busyReg < Const(1),
                  doneReg < Const(0),
                  latchedRows < rowsReg,
                  latchedCols < colsReg,
                  latchedShift < shiftReg.getRange(0, cfg.shiftWidth),
                  rowBlk < Const(0, width: 8),
                  seqState < Const(_stRowInit, width: 3),
                ],
              ),
            ]),

            // ROW_INIT: reset colTile, go to TILE.
            CaseItem(Const(_stRowInit, width: 3), [
              colTile < Const(0, width: 8),
              seqState < Const(_stTile, width: 3),
            ]),

            // TILE: linValid is driven HIGH combinationally this cycle.
            // LoomLinear samples the tile at this posedge.
            // If this is the last tile (colTile+1 == numColTiles), go directly
            // to CAPTURE_W so outValid is checked on the very next posedge
            // (one cycle after the last tile's posedge, matching LoomLinear's
            // timing contract). For non-last tiles, go to WAIT then back to TILE.
            CaseItem(Const(_stTile, width: 3), [
              colTile < (colTile + Const(1, width: 8)),
              If(
                (colTile + Const(1, width: 8)).eq(numColTiles),
                then: [seqState < Const(_stCaptureW, width: 3)],
                orElse: [seqState < Const(_stWait, width: 3)],
              ),
            ]),

            // WAIT: inter-tile gap. linValid is LOW. Always go back to TILE.
            CaseItem(Const(_stWait, width: 3), [
              seqState < Const(_stTile, width: 3),
            ]),

            // CAPTURE_W: wait for LoomLinear outValid, then write result.
            // outValid pulses one cycle after the last tile's posedge.
            // Last tile posedge was TILE state; WAIT was the next cycle;
            // CAPTURE_W is the cycle after that => outValid should fire now.
            CaseItem(Const(_stCaptureW, width: 3), [
              If(
                linear.outValid,
                then: [
                  ...captureWrites,
                  rowBlk < (rowBlk + Const(1, width: 8)),
                  If(
                    (rowBlk + Const(1, width: 8)).eq(numRowBlocks),
                    then: [seqState < Const(_stDone, width: 3)],
                    orElse: [seqState < Const(_stRowInit, width: 3)],
                  ),
                ],
              ),
            ]),

            // DONE: clear busy, set done, return to IDLE.
            CaseItem(Const(_stDone, width: 3), [
              busyReg < Const(0),
              doneReg < Const(1),
              seqState < Const(_stIdle, width: 3),
            ]),
          ]),

          // Bus register interface.
          // Region decode: bits [11:8] of byte address.
          If(
            bus.stb & ~bus.ack,
            then: [
              bus.ack < Const(1),

              Case(bus.addr.getRange(8, 12), [
                // CSR region.
                CaseItem(Const(0x0, width: 4), [
                  Case(bus.addr.getRange(0, 8), [
                    CaseItem(Const(_regVersion, width: 8), [
                      bus.dataOut < Const(config.versionMagic, width: 32),
                    ]),
                    CaseItem(Const(_regRows, width: 8), [
                      If(
                        bus.we,
                        then: [rowsReg < bus.dataIn],
                        orElse: [bus.dataOut < rowsReg],
                      ),
                    ]),
                    CaseItem(Const(_regCols, width: 8), [
                      If(
                        bus.we,
                        then: [colsReg < bus.dataIn],
                        orElse: [bus.dataOut < colsReg],
                      ),
                    ]),
                    CaseItem(Const(_regShift, width: 8), [
                      If(
                        bus.we,
                        then: [shiftReg < bus.dataIn],
                        orElse: [bus.dataOut < shiftReg],
                      ),
                    ]),
                    CaseItem(Const(_regControl, width: 8), [
                      If(
                        bus.we,
                        then: [controlReg < bus.dataIn],
                        orElse: [bus.dataOut < controlReg],
                      ),
                    ]),
                    CaseItem(Const(_regStatus, width: 8), [
                      bus.dataOut <
                          [Const(0, width: 30), doneReg, busyReg].swizzle(),
                    ]),
                    CaseItem(Const(_regModelLen, width: 8), [
                      bus.dataOut < Const(nameLen, width: 32),
                    ]),
                    ...nanoCsrItems,
                  ]),
                ]),

                // Weight buffer (WO).
                CaseItem(Const(0x1, width: 4), [
                  If(bus.we, then: weightWrites),
                ]),

                // Activation buffer (WO).
                CaseItem(Const(0x2, width: 4), [If(bus.we, then: actWrites)]),

                // RowMult buffer (WO).
                CaseItem(Const(0x3, width: 4), [If(bus.we, then: multWrites)]),

                // Result buffer (RO).
                CaseItem(Const(0x4, width: 4), [...resultReads]),

                // NAME region (RO): baked ASCII model name, 4 chars/word.
                CaseItem(Const(_regionName, width: 4), [
                  Case(bus.addr.getRange(2, 8), [
                    for (var i = 0; i < nameWords.length; i++)
                      CaseItem(Const(i, width: 6), [
                        bus.dataOut < nameWords[i],
                      ]),
                  ]),
                ]),
              ]),
            ],
          ),
        ],
      ),
    ]);
  }

  @override
  HarborDeviceTreeNode get dtNode => HarborDeviceTreeNode(
    compatible: ['midstall,loom-accelerator'],
    reg: BusAddressRange(config.baseAddress, windowSize),
    properties: {
      'midstall,pe-rows': config.peRows,
      'midstall,pe-cols': config.peCols,
      'midstall,max-rows': config.maxRows,
      'midstall,max-cols': config.maxCols,
      '#address-cells': 1,
      '#size-cells': 1,
    },
  );

  @override
  HarborSvdPeripheral get svdPeripheral => HarborSvdPeripheral(
    name: 'LOOM_ACCELERATOR',
    groupName: 'LOOM',
    description: 'Loom int8 linear-layer accelerator',
    baseAddress: config.baseAddress,
    size: windowSize,
    registers: _loomAcceleratorRegisterMap,
  );
}
