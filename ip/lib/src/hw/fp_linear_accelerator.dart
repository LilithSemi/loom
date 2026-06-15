// LoomFpLinearAccelerator: a host-drivable peripheral wrapping LoomFpLinear, so
// the model-agnostic runtime can run a full fp16-in/fp16-out W4A8 linear over a
// transport (UART/USB). The device does the LLM's heavy matmul math on-chip;
// the host streams activations and reads results.
//
// Two bus interfaces:
//   - 'bus' : Wishbone SLAVE  - host configures dims/weight_base, pushes fp16
//             activations + per-row fp16 weight scales, strobes start, polls
//             status, reads back the fp16 result rows.
//   - 'mem' : Wishbone MASTER - the inner LoomFpLinear reads int4 weights from
//             memory (SRAM scratchpad or DDR).
//
// CSR map (byte offsets, region nibble [11:8]):
//   0x000 VERSION     RO 0x4C4F4F4D
//   0x004 COL_TILES   RW inner-dim tiles (cols = col_tiles*2)
//   0x008 ROW_BLOCKS  RW output-row blocks (rows = row_blocks*2)
//   0x00C WEIGHT_BASE RW byte addr of tile-major int4 weights in 'mem'
//   0x010 CONTROL     RW bit0 start (self-clears, also resets the result ptr)
//   0x014 STATUS      RO bit0 busy, bit1 done (sticky until next start)
//   0x018 ACT_PUSH    WO write low16 = next fp16 activation (host pushes cols)
//   0x024 ACT_PUSH2   WO write = TWO packed fp16 activations (low16 then high16);
//                        halves the host->device activation bytes on the wire.
//   0x01C SCALE_PUSH  WO write low16 = next fp16 row scale  (host pushes rows)
//   0x020 SCALE_BASE  RW byte addr in 'mem' of the per-row fp16 scales (one per
//                        32-bit word, low16). When != 0, the accelerator fetches
//                        the scales from flash itself on start (RESIDENT scales);
//                        the host pushes none. When 0, host uses SCALE_PUSH.
//   0x100.. RESULT    RO PACKED fp16 results: word w at 0x100 + w*4 = {row 2w+1
//                        in high16, row 2w in low16}, halving the result bytes
//                        the host reads back.

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_linear.dart';
import 'fp_silu.dart';

const _faVersion = 0x000;
const _faColTiles = 0x004;
const _faRowBlocks = 0x008;
const _faWeightBase = 0x00C;
const _faControl = 0x010;
const _faStatus = 0x014;
const _faActPush = 0x018;
const _faScalePush = 0x01C;
const _faActPush2 = 0x024; // one write = two packed fp16 activations
// Byte addr in 'mem' of this matrix's per-row fp16 scales (2 packed per 32-bit
// word). When nonzero, the accelerator reads the scales from flash itself on
// start (RESIDENT scales) instead of the host pushing them via SCALE_PUSH.
const _faScaleBase = 0x020;
// Fusion MODE: 0 = normal, 1 = capture_gate (results -> gateBuf, host does not
// read them), 2 = fuse_up (results are the "up" matmul; after it finishes the
// combine FSM overwrites resultBuf[i] with silu(gateBuf[i]) * resultBuf[i], so
// the host reads back silu(gate)*up in one vector instead of gate and up).
const _faMode = 0x028;
// Host act-scale override (16-bit fp16): when nonzero, overrides the inner
// LoomFpLinear's act-scale end to end - both the int8 activation quantization
// AND the dequant multiplier - so every col-block of a column-tiled matmul
// quantizes on the SAME shared grid before their partial sums are added.
const _faActScale = 0x02C;
// Column-group address offset: added to SCALE_BASE so each col-tile block
// loads its own group's per-row scales instead of always group 0's.
const _faGroupOff = 0x030;

// Scale-fetch FSM states.
const _saIdle = 0;
const _saRead = 1;
const _saRun = 2;

// SwiGLU combine FSM states (fuse_up mode).
const _cbIdle = 0;
const _cbRun = 1; // sweeping read index ci over the rows
const _cbDrain = 2; // flush the last L products out of the pipelined datapath
const _cbDone = 3; // set doneReg, return to idle
// Read-to-product latency of the combine datapath: 1 result-BRAM registered read
// + 1 operand-select register (keeps the slow BRAM read output out of the SiLU/
// multiply timing path - without it the design only closes ~27.8MHz) + 1
// clocked-multiply internal flop. The write index trails the read by this many.
const _cbLatency = 3;

class LoomFpLinearAccelerator extends BridgeModule
    with HarborDeviceTreeNodeProvider {
  final int maxColTiles;
  final int maxRowBlocks;
  final int addressWidth;
  final int versionMagic;
  final int baseAddress;

  static const int windowSize = 0x800;

  late final BusSlavePort bus;
  late final WishboneInterface mem;

  LoomFpLinearAccelerator({
    this.baseAddress = 0x10000000,
    this.maxColTiles = 4,
    this.maxRowBlocks = 4,
    this.addressWidth = 32,
    this.versionMagic = 0x4C4F4F4D,
    int recipIterations = 4,
    bool ternaryWeights = false,
    BusProtocol protocol = BusProtocol.wishbone,
    String? name,
  }) : super(
         'LoomFpLinearAccelerator',
         name: name ?? 'loom_fp_linear_accelerator',
       ) {
    final maxRows = maxRowBlocks * 2;
    final aw = addressWidth;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    bus = BusSlavePort.create(
      module: this,
      name: 'bus',
      protocol: protocol,
      addressWidth: aw,
      dataWidth: 32,
    );

    final memRef = addInterface(
      WishboneInterface(WishboneConfig(addressWidth: aw, dataWidth: 32)),
      name: 'mem',
      role: PairRole.provider,
    );
    mem = memRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');

    final colTilesReg = Logic(name: 'col_tiles_reg', width: 16);
    final rowBlocksReg = Logic(name: 'row_blocks_reg', width: 16);
    final weightBaseReg = Logic(name: 'weight_base_reg', width: aw);
    final scaleBaseReg = Logic(name: 'scale_base_reg', width: aw);
    // 0 normal / 1 gate / 2 up / 3-5 reserved for col-tiling modes.
    final modeReg = Logic(name: 'mode_reg', width: 3);
    final actScaleReg = Logic(name: 'host_act_scale_reg', width: 16);
    final groupOffReg = Logic(name: 'group_off_reg', width: aw);
    final startPulse = Logic(name: 'start_pulse');
    final doneReg = Logic(name: 'done_reg');
    final rowCtr = Logic(name: 'row_ctr', width: 16);

    // SwiGLU combine FSM (fuse_up): sweeps ci over the rows, feeding
    // silu(gateBuf[ci]) * resultBuf[ci] through the pipelined combine datapath
    // and writing the (_cbLatency-cycle-delayed) product back into
    // resultBuf[ci - _cbLatency].
    final cbState = Logic(name: 'cb_state', width: 2);
    final cbCi = Logic(
      name: 'cb_ci',
      width: 16,
    ); // read index (into gate/result)
    final cbcw = (_cbLatency + 1).bitLength.clamp(1, 16);
    final cbDrainCnt = Logic(
      name: 'cb_drain_cnt',
      width: cbcw,
    ); // DRAIN countdown
    // Operand-select register: latch the wide resultBuf mux output so it is NOT
    // in series with the multiply (timing hardening). The gate operand comes from
    // gateBram's registered read directly (its readLatency IS this stage).
    // Parity of the combine read index cbCi, delayed one cycle to match the
    // even/odd result BRAM read latency (picks resOdd vs resEven for the operand).
    final cbCiLsbD = Logic(name: 'cb_ci_lsb_d');
    // Operand-select registers: latch the (slow) result-BRAM read outputs one
    // cycle before SiLU/multiply, so the BRAM clock-to-out delay is not in series
    // with the SiLU LUT + multiply. gate feeds SiLU, result feeds the multiply.
    final gateSelReg = Logic(name: 'gate_sel_reg', width: 16);
    final resultSelReg = Logic(name: 'result_sel_reg', width: 16);
    // Host RESULT read is a registered BRAM read: assert on the stb cycle, then
    // deliver data + ack the next cycle (readLatency 1).
    final resPend = Logic(name: 'res_pend');
    // Read-index / valid shift pipeline, _cbLatency deep, so the write index and
    // valid land in lockstep with the product. cbWr/cbWrVld are the tail.
    final cbIdxPipe = [
      for (var i = 0; i < _cbLatency; i++) Logic(name: 'cb_idx_$i', width: 16),
    ];
    final cbVldPipe = [
      for (var i = 0; i < _cbLatency; i++) Logic(name: 'cb_vld_$i'),
    ];
    final cbWr = cbIdxPipe.last; // delayed write index (= ci - _cbLatency)
    final cbWrVld = cbVldPipe.last; // a valid product is landing now
    final combineActive = cbState
        .neq(Const(_cbIdle, width: 2))
        .named('combine_active');

    // Scale-fetch FSM: when SCALE_BASE != 0, on start the accelerator reads the
    // `rows` per-row fp16 scales from flash itself (one 32-bit word per scale,
    // low16) instead of the host pushing them, then fires the inner linear.
    final saState = Logic(name: 'sa_state', width: 2);
    final scaleAddr = Logic(
      name: 'scale_addr',
      width: aw,
    ); // current scale word addr
    final sidx = Logic(name: 'sidx', width: 16); // scales fetched so far
    final rowsTarget = (rowBlocksReg << 1).named(
      'rows_target',
    ); // rows = row_blocks*2
    final scaleReading = saState
        .eq(Const(_saRead, width: 2))
        .named('scale_reading');
    // Push a fetched scale into the inner linear on the flash ack cycle.
    final scaleAck = (scaleReading & mem.ack).named('scale_ack');

    // Combinational push pulses: assert during the bus write-accept cycle.
    final writeAccept = bus.stb & ~bus.ack & bus.we;
    final actPush =
        writeAccept & bus.addr.getRange(0, 12).eq(Const(_faActPush, width: 12));
    final actPush2 =
        writeAccept &
        bus.addr.getRange(0, 12).eq(Const(_faActPush2, width: 12));
    final scalePush =
        writeAccept &
        bus.addr.getRange(0, 12).eq(Const(_faScalePush, width: 12));

    // Packed-pair activation push: a write to ACT_PUSH2 buffers low16 this cycle
    // and high16 the next (the command engine's next write is >=2 cycles away).
    final actHiReg = Logic(name: 'act_hi_reg', width: 16);
    final actHiPending = Logic(name: 'act_hi_pending');

    // Inner fp16 W4A8 linear (the bus master for weights).
    final fl = LoomFpLinear(
      maxColTiles: maxColTiles,
      maxRowBlocks: maxRowBlocks,
      recipIterations: recipIterations,
      ternaryWeights: ternaryWeights,
      hostActScale: actScaleReg,
    );
    fl.input('clk').srcConnection! <= clk;
    fl.input('reset').srcConnection! <= reset;
    fl.input('start').srcConnection! <= startPulse;
    fl.input('col_tiles').srcConnection! <= colTilesReg;
    fl.input('row_blocks').srcConnection! <= rowBlocksReg;
    fl.input('weight_base').srcConnection! <= weightBaseReg;
    // Activation pushes: legacy single (actPush), the low half of a packed pair
    // (actPush2), or the buffered high half one cycle later (actHiPending).
    fl.input('x_en').srcConnection! <= (actPush | actPush2 | actHiPending);
    fl.input('x_in').srcConnection! <=
        mux(actHiPending, actHiReg, bus.dataIn.getRange(0, 16));
    // Scales come from the flash read (scaleAck) while fetching, else the host
    // SCALE_PUSH (legacy / non-resident). fl buffers either into rsBuf in _load.
    fl.input('rs_en').srcConnection! <= mux(scaleReading, scaleAck, scalePush);
    fl.input('rs_in').srcConnection! <=
        mux(
          scaleReading,
          mem.datMiso.getRange(0, 16),
          bus.dataIn.getRange(0, 16),
        );

    // 'mem' master is muxed: the wrapper drives it while reading scales, then the
    // inner linear drives it for the weight reads (they never overlap).
    mem.cyc <= mux(scaleReading, Const(1), fl.output('mem_CYC'));
    mem.stb <= mux(scaleReading, Const(1), fl.output('mem_STB'));
    mem.we <= mux(scaleReading, Const(0), fl.output('mem_WE'));
    mem.adr <= mux(scaleReading, scaleAddr, fl.output('mem_ADR'));
    mem.datMosi <= fl.output('mem_DAT_MOSI');
    mem.sel <= mux(scaleReading, Const(0xF, width: 4), fl.output('mem_SEL'));
    // The inner linear only sees acks during its own (weight-read) phase.
    fl.input('mem_ACK').srcConnection! <= (mem.ack & ~scaleReading);
    fl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    final flBusy = fl.output('busy');
    final flDone = fl.output('done');
    final flYValid = fl.output('y_valid');
    final flY = fl.output('y');

    // Result buffer: split into EVEN-row and ODD-row block RAMs. This keeps the
    // host's PACKED read (word w = {row 2w+1, row 2w}) to one access per BRAM,
    // while per-row writes (capture at rowCtr, combine at cbWr) hit exactly one
    // half by row parity. A single wide buffer's read/write muxes would be the
    // LUT bottleneck capping maxRows on the 25F. word w = {resOdd[w],
    // resEven[w]}. Row r lives in {even|odd}[r>>1]. yosys maps each to DP16KD.
    final resultWords = maxRows ~/ 2;
    final resAw = (resultWords - 1).bitLength;
    final resultWordIdx = (bus.addr - Const(0x100, width: aw)).getRange(
      2,
      2 + resAw,
    );
    final resultRange =
        bus.addr.gte(Const(0x100, width: aw)) &
        bus.addr.lt(Const(0x100 + resultWords * 4, width: aw));

    // Gate buffer BRAM (capture_gate): read by the combine at cbCi, written on
    // capture. readLatency 1 IS the operand-select stage, so _cbLatency stays 2.
    final brAw = (maxRows - 1).bitLength;
    final gateBram = HarborBram(
      clk,
      width: 16,
      depth: maxRows,
      wrEn: flYValid & modeReg.eq(Const(1, width: 3)),
      wrAddr: rowCtr.getRange(0, brAw),
      wrData: flY,
      rdAddr: cbCi.getRange(0, brAw),
      name: 'gate_bram',
    );

    // Column-tile fp32 accumulator (COLTILE_FIRST/MID/LAST): sums a matmul's
    // col-blocks on-chip, in fp32, so the host reads ONE fp16 result after the
    // LAST block instead of assembling partial sums itself. Mirrors the
    // SwiGLU combine's delay-alignment above: accBram's read is REGISTERED
    // (1 cycle, like gateBram) and the fp32 adder is CLOCKED (1 more cycle),
    // so the write-side row index / mode flags / dequant value are each
    // delayed 2 cycles to land together with the adder's output.
    final isFirst = modeReg.eq(Const(3, width: 3));
    final isMid = modeReg.eq(Const(4, width: 3));
    final isLast = modeReg.eq(Const(5, width: 3));

    Logic coltileDly(String name, Logic d) {
      final r = Logic(name: name, width: d.width);
      r <= flop(clk, d, reset: reset, resetValue: 0);
      return r;
    }

    // Stage 1: registered once. accBram.rdData (below) is ALSO registered
    // once off the SAME rowCtr, so it lands on the same cycle as these.
    final yAccD1 = coltileDly('coltile_yacc_d1', fl.output('y_acc'));
    final flYValidD1 = coltileDly('coltile_valid_d1', flYValid);
    final rowCtrD1 = coltileDly('coltile_row_d1', rowCtr);
    final isFirstD1 = coltileDly('coltile_first_d1', isFirst);
    final isMidD1 = coltileDly('coltile_mid_d1', isMid);
    final isLastD1 = coltileDly('coltile_last_d1', isLast);

    // Stage 2: registered again to match the fp32 adder's own 1-cycle
    // internal pipeline (FloatingPointAdderSinglePath is CLOCKED), so the
    // FIRST path (no add: y_acc passes straight through) and the MID/LAST
    // path (adder output) land on the SAME cycle for the accNew mux.
    final yAccD2 = coltileDly('coltile_yacc_d2', yAccD1);
    final flYValidD2 = coltileDly('coltile_valid_d2', flYValidD1);
    final rowCtrD2 = coltileDly('coltile_row_d2', rowCtrD1);
    final isFirstD2 = coltileDly('coltile_first_d2', isFirstD1);
    final isMidD2 = coltileDly('coltile_mid_d2', isMidD1);
    final isLastD2 = coltileDly('coltile_last_d2', isLastD1);

    final accNew = Logic(name: 'coltile_acc_new', width: 32);
    final accWrEn = flYValidD2 & (isFirstD2 | isMidD2 | isLastD2);
    final accBram = HarborBram(
      clk,
      width: 32,
      depth: maxRows,
      wrEn: accWrEn,
      wrAddr: rowCtrD2.getRange(0, brAw),
      wrData: accNew,
      // Continuously tracks the CURRENT row. ReadLatency 1 means rd_data
      // lands one cycle from now, in lockstep with yAccD1/flYValidD1/etc.
      rdAddr: rowCtr.getRange(0, brAw),
      name: 'coltile_acc',
      // FLOPS, not DP16KD: this read-modify-write bank can read AND write the
      // same row in one cycle when a slow flash read stalls the pipeline. A
      // DP16KD returns undefined on that collision (scrambles the accumulate on
      // silicon while passing sim). Flops are deterministic and sim-faithful.
      useFlops: true,
    );
    // MID/LAST: fp32-add the running sum (accBram, read off rowCtr one cycle
    // ago) to this block's fresh dequant (yAccD1). FIRST: no add, just the
    // fresh value carried one more cycle to match the adder's latency.
    final accSum = FloatingPointAdderSinglePath(
      FloatingPoint32()..gets(accBram.rdData),
      FloatingPoint32()..gets(yAccD1),
      clk: clk,
    ).sum.packed;
    accNew <= mux(isFirstD2, yAccD2, accSum);

    // LAST narrows the fp32 running sum to fp16 for the host-visible result.
    final accF16 = FloatingPoint16();
    FloatingPointConverter(FloatingPoint32()..gets(accNew), accF16);
    // A LAST row's narrowed sum is ready to land in the result BRAM 2 cycles
    // after ITS OWN flYValid (this is what the result-write gating below uses
    // instead of the plain, undelayed flYValid/rowCtr that NORMAL/FUSE use).
    final coltileLastWrite = flYValidD2 & isLastD2;

    // SwiGLU combine datapath (fuse_up), pipelined _cbLatency=2:
    //   gateBram.rdData (readLatency 1) -> LoomSiLU (comb) -> clocked multiply
    //   flop -> combineProd. The result operand comes from the even/odd BRAM
    //   read, parity-selected by cbCiLsbD (cbCi[0] delayed to match readLatency).
    final silu = LoomSiLU();
    silu.input('x').srcConnection! <= gateSelReg;
    final resultCombOperand = Logic(name: 'result_comb_operand', width: 16);
    final combineProd = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(silu.output('y')),
      FloatingPoint16()..gets(resultSelReg),
      clk: clk,
    ).product.packed;

    // Result BRAM write port: the combine (cbWr, combineProd) when a product is
    // landing, else result capture. The three never overlap (capture during
    // the matmul, combine after. Col-tile modes are a separate matmul run
    // from fuse_up/normal). Address = row index >> 1; the row's LSB picks the
    // even or odd half.
    //
    // NORMAL (0) / FUSE_UP (2) capture immediately, same cycle as flYValid.
    // COLTILE FIRST/MID (3/4) write NOTHING here (only the fp32 accumulator
    // above). COLTILE LAST (5) writes the
    // narrowed running sum, gated on the accumulate pipeline's OWN delayed
    // valid/row (coltileLastWrite/rowCtrD2), not the immediate flYValid/rowCtr.
    final isNormalOrFuse =
        modeReg.eq(Const(0, width: 3)) | modeReg.eq(Const(2, width: 3));
    final capWriteNormal = flYValid & isNormalOrFuse;
    final capWrite = capWriteNormal | coltileLastWrite;
    final resWrData = mux(
      cbWrVld,
      combineProd,
      mux(coltileLastWrite, accF16.packed, flY),
    );
    final resWrAddr = mux(
      cbWrVld,
      cbWr.getRange(1, 1 + resAw),
      mux(
        coltileLastWrite,
        rowCtrD2.getRange(1, 1 + resAw),
        rowCtr.getRange(1, 1 + resAw),
      ),
    );
    final resWrLsb = mux(
      cbWrVld,
      cbWr.getRange(0, 1),
      mux(coltileLastWrite, rowCtrD2.getRange(0, 1), rowCtr.getRange(0, 1)),
    );
    final resWrThis = cbWrVld | capWrite;
    // Read port: the combine sweeps cbCi (>>1). Otherwise the host result word.
    final resRdAddr = mux(
      combineActive,
      cbCi.getRange(1, 1 + resAw),
      resultWordIdx,
    );
    final resEven = HarborBram(
      clk,
      width: 16,
      depth: resultWords,
      wrEn: resWrThis & ~resWrLsb,
      wrAddr: resWrAddr,
      wrData: resWrData,
      rdAddr: resRdAddr,
      name: 'res_even',
    );
    final resOdd = HarborBram(
      clk,
      width: 16,
      depth: resultWords,
      wrEn: resWrThis & resWrLsb,
      wrAddr: resWrAddr,
      wrData: resWrData,
      rdAddr: resRdAddr,
      name: 'res_odd',
    );
    resultCombOperand <= mux(cbCiLsbD, resOdd.rdData, resEven.rdData);
    // Host packed read: {row 2w+1, row 2w} = {resOdd[w], resEven[w]}.
    final resultReadData = [resOdd.rdData, resEven.rdData].swizzle();

    Sequential(clk, [
      If(
        reset,
        then: [
          colTilesReg < Const(0, width: 16),
          rowBlocksReg < Const(0, width: 16),
          weightBaseReg < Const(0, width: aw),
          scaleBaseReg < Const(0, width: aw),
          modeReg < Const(0, width: 3),
          actScaleReg < Const(0, width: 16),
          groupOffReg < Const(0, width: aw),
          saState < Const(_saIdle, width: 2),
          scaleAddr < Const(0, width: aw),
          sidx < Const(0, width: 16),
          actHiReg < Const(0, width: 16),
          actHiPending < Const(0),
          startPulse < Const(0),
          doneReg < Const(0),
          rowCtr < Const(0, width: 16),
          cbState < Const(_cbIdle, width: 2),
          cbCi < Const(0, width: 16),
          cbDrainCnt < Const(0, width: cbcw),
          cbCiLsbD < Const(0),
          gateSelReg < Const(0, width: 16),
          resultSelReg < Const(0, width: 16),
          resPend < Const(0),
          for (final p in cbIdxPipe) p < Const(0, width: 16),
          for (final v in cbVldPipe) v < Const(0),
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
        ],
        orElse: [
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
          startPulse < Const(0),
          actHiPending < Const(0),

          // Packed activation pair: latch high16, emit it next cycle.
          If(
            actPush2,
            then: [
              actHiReg < bus.dataIn.getRange(16, 32),
              actHiPending < Const(1),
            ],
          ),

          // Delay cbCi's parity one cycle to match the even/odd result BRAM read
          // latency, so resultCombOperand picks the right half for the operand
          // that lands this cycle. gate + result operands both track readLatency 1.
          cbCiLsbD < cbCi.getRange(0, 1),
          // Register the BRAM read outputs before SiLU/multiply (timing: keeps the
          // BRAM clock-to-out off the SiLU LUT path). This is the 2nd of the 3
          // _cbLatency stages (BRAM read -> these regs -> multiply flop).
          gateSelReg < gateBram.rdData,
          resultSelReg < resultCombOperand,

          // The matmul finishing sets STATUS.done for normal/capture_gate. In
          // fuse_up it instead kicks off the combine FSM (which sets done at its
          // own end). The matmul's own `done` must NOT set it early.
          If(
            flDone & modeReg.neq(Const(2, width: 3)),
            then: [doneReg < Const(1)],
          ),
          If(
            flDone &
                modeReg.eq(Const(2, width: 3)) &
                cbState.eq(Const(_cbIdle, width: 2)),
            then: [
              cbState < Const(_cbRun, width: 2),
              cbCi < Const(0, width: 16),
              cbDrainCnt < Const(0, width: cbcw),
              // Fresh sweep: clear the write pipeline so no stale valids fire.
              for (final p in cbIdxPipe) p < Const(0, width: 16),
              for (final v in cbVldPipe) v < Const(0),
            ],
          ),

          // SwiGLU combine FSM (fuse_up): sweep ci over the rows. The pipelined
          // product lands via the result BRAM write port (resEven/resOdd wrEn,
          // gated by cbWrVld) at the trailing index cbWr (= ci - _cbLatency).
          // Adds ~rows + _cbLatency cycles after the up matmul.
          Case(cbState, [
            CaseItem(Const(_cbIdle, width: 2), []),
            CaseItem(Const(_cbRun, width: 2), [
              // Feed a valid read into the write pipeline and shift it along.
              cbIdxPipe[0] < cbCi,
              cbVldPipe[0] < Const(1),
              for (var i = 1; i < _cbLatency; i++) ...[
                cbIdxPipe[i] < cbIdxPipe[i - 1],
                cbVldPipe[i] < cbVldPipe[i - 1],
              ],
              If(
                (cbCi + Const(1, width: 16)).gte(rowsTarget),
                then: [
                  // Last read presented. Drain the pipeline for _cbLatency cycles.
                  cbState < Const(_cbDrain, width: 2),
                  cbDrainCnt < Const(_cbLatency, width: cbcw),
                ],
                orElse: [cbCi < cbCi + Const(1, width: 16)],
              ),
            ]),
            CaseItem(Const(_cbDrain, width: 2), [
              // No new valid reads (stage-0 valid = 0); keep shifting so the last
              // _cbLatency products flush out (combineWrites fires on cbWrVld).
              cbIdxPipe[0] < cbCi,
              cbVldPipe[0] < Const(0),
              for (var i = 1; i < _cbLatency; i++) ...[
                cbIdxPipe[i] < cbIdxPipe[i - 1],
                cbVldPipe[i] < cbVldPipe[i - 1],
              ],
              If(
                cbDrainCnt.gt(Const(1, width: cbcw)),
                then: [cbDrainCnt < cbDrainCnt - Const(1, width: cbcw)],
                orElse: [cbState < Const(_cbDone, width: 2)],
              ),
            ]),
            CaseItem(Const(_cbDone, width: 2), [
              doneReg < Const(1),
              cbState < Const(_cbIdle, width: 2),
              for (final v in cbVldPipe) v < Const(0),
            ]),
          ]),

          // Scale-fetch FSM: read `rows` scales from flash, one word each, then
          // fire the inner linear. RUN waits for it to finish before going idle.
          If(
            scaleAck,
            then: [
              sidx < sidx + Const(1, width: 16),
              scaleAddr < scaleAddr + Const(4, width: aw),
              If(
                (sidx + Const(1, width: 16)).gte(rowsTarget),
                then: [
                  saState < Const(_saRun, width: 2),
                  startPulse < Const(1),
                ],
              ),
            ],
          ),
          If(
            saState.eq(Const(_saRun, width: 2)) & flDone,
            then: [saState < Const(_saIdle, width: 2)],
          ),

          // Capture each result row in order. The row lands in gateBram (mode 1)
          // or the even/odd resultBuf BRAM (normal / fuse_up up-matmul) via those
          // BRAMs' write-enables (keyed on flYValid + rowCtr above). Here we just
          // advance the write index.
          If(flYValid, then: [rowCtr < rowCtr + Const(1, width: 16)]),

          If(
            bus.stb & ~bus.ack & ~resPend,
            then: [
              // RESULT reads come from the even/odd result BRAM (a registered
              // read), so defer one cycle: data + ack land together next cycle
              // (resRdAddr already holds the word). CSR reads ack same-cycle.
              If(
                resultRange,
                then: [resPend < Const(1)],
                orElse: [
                  bus.ack < Const(1),
                  // Register file (offsets 0x000..0x0FF, region nibble 0).
                  If(
                    bus.addr.getRange(8, 12).eq(Const(0x0, width: 4)),
                    then: [
                      Case(bus.addr.getRange(0, 8), [
                        CaseItem(Const(_faVersion, width: 8), [
                          bus.dataOut < Const(versionMagic, width: 32),
                        ]),
                        CaseItem(Const(_faColTiles, width: 8), [
                          If(
                            bus.we,
                            then: [colTilesReg < bus.dataIn.getRange(0, 16)],
                            orElse: [bus.dataOut < colTilesReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faRowBlocks, width: 8), [
                          If(
                            bus.we,
                            then: [rowBlocksReg < bus.dataIn.getRange(0, 16)],
                            orElse: [bus.dataOut < rowBlocksReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faWeightBase, width: 8), [
                          If(
                            bus.we,
                            then: [weightBaseReg < bus.dataIn.getRange(0, aw)],
                            orElse: [
                              bus.dataOut < weightBaseReg.zeroExtend(32),
                            ],
                          ),
                        ]),
                        CaseItem(Const(_faScaleBase, width: 8), [
                          If(
                            bus.we,
                            then: [scaleBaseReg < bus.dataIn.getRange(0, aw)],
                            orElse: [bus.dataOut < scaleBaseReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faMode, width: 8), [
                          If(
                            bus.we,
                            then: [modeReg < bus.dataIn.getRange(0, 3)],
                            orElse: [bus.dataOut < modeReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faActScale, width: 8), [
                          If(
                            bus.we,
                            then: [actScaleReg < bus.dataIn.getRange(0, 16)],
                            orElse: [bus.dataOut < actScaleReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faGroupOff, width: 8), [
                          If(
                            bus.we,
                            then: [groupOffReg < bus.dataIn.getRange(0, aw)],
                            orElse: [bus.dataOut < groupOffReg.zeroExtend(32)],
                          ),
                        ]),
                        CaseItem(Const(_faControl, width: 8), [
                          If(
                            bus.we & bus.dataIn[0],
                            then: [
                              doneReg < Const(0),
                              rowCtr < Const(0, width: 16),
                              // Fresh matmul: also reset the combine FSM + pipeline.
                              cbState < Const(_cbIdle, width: 2),
                              cbCi < Const(0, width: 16),
                              cbDrainCnt < Const(0, width: cbcw),
                              for (final p in cbIdxPipe)
                                p < Const(0, width: 16),
                              for (final v in cbVldPipe) v < Const(0),
                              // Resident scales (SCALE_BASE != 0): fetch them from flash
                              // first, then fire. Else fire immediately (host pushed).
                              // The load base is SCALE_BASE + SCALE_GROUP_OFF: each
                              // col-block writes its own REG_SCALE_GROUP_OFF (0x030)
                              // before it starts, so block b loads GROUP b's own
                              // per-row scales instead of always re-reading group 0's.
                              // groupOffReg is aw-wide (same as scaleBaseReg) so this
                              // add does not truncate. The address space is small so
                              // the top bits are 0 and no carry that matters is dropped.
                              If(
                                scaleBaseReg.neq(Const(0, width: aw)),
                                then: [
                                  saState < Const(_saRead, width: 2),
                                  scaleAddr < scaleBaseReg + groupOffReg,
                                  sidx < Const(0, width: 16),
                                ],
                                orElse: [startPulse < Const(1)],
                              ),
                            ],
                          ),
                        ]),
                        CaseItem(Const(_faStatus, width: 8), [
                          // busy stays high while the combine FSM is still running.
                          bus.dataOut <
                              [
                                Const(0, width: 30),
                                doneReg,
                                flBusy | combineActive,
                              ].swizzle(),
                        ]),
                        // ACT_PUSH / ACT_PUSH2 / SCALE_PUSH are write-only side effects
                        // handled combinationally. Nothing to do in the register file.
                        CaseItem(Const(_faActPush, width: 8), []),
                        CaseItem(Const(_faActPush2, width: 8), []),
                        CaseItem(Const(_faScalePush, width: 8), []),
                      ]),
                    ],
                  ),
                ],
              ),
            ],
          ),
          // Deliver the deferred result read: the even/odd BRAM data is valid the
          // cycle after the address was presented.
          If(
            resPend,
            then: [
              bus.dataOut < resultReadData,
              bus.ack < Const(1),
              resPend < Const(0),
            ],
          ),
        ],
      ),
    ]);
  }

  @override
  HarborDeviceTreeNode get dtNode => HarborDeviceTreeNode(
    compatible: ['midstall,loom-fp-linear-accelerator'],
    reg: BusAddressRange(baseAddress, windowSize),
    properties: {'#address-cells': 1, '#size-cells': 1},
  );
}
