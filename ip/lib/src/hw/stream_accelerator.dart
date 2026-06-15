// LoomStreamAccelerator: a host-drivable peripheral wrapping LoomStreamMatmul.
//
// Two bus interfaces:
//   - 'bus'  : Wishbone SLAVE  - the host (runtime over UART/USB) writes the
//              config CSRs (dims, shift, memory base addresses), strobes start,
//              polls status, and reads back the per-row-block results.
//   - 'mem'  : Wishbone MASTER - the inner streaming matmul reads weights /
//              activations / mults from a memory (SRAM scratchpad or DDR).
//
// This is what lets the model-agnostic runtime run a memory-backed matmul that
// is NOT capped by on-chip register files (unlike LoomAccelerator's 8x8).
//
// CSR map (byte offsets, 12-bit slave bus, region nibble [11:8]):
//   0x000 VERSION     RO 0x4C4F4F4D
//   0x004 ROW_BLOCKS  RW number of peRows-row output blocks
//   0x008 COL_TILES   RW number of peCols-col inner tiles
//   0x00C SHIFT       RW requant shift
//   0x010 CONTROL     RW bit0 start (self-clears)
//   0x014 STATUS      RO bit0 busy, bit1 done (sticky until next start)
//   0x018 WEIGHT_BASE RW byte addr of tile-major weights in 'mem'
//   0x01C ACT_BASE    RW byte addr of activations in 'mem'
//   0x020 MULT_BASE   RW byte addr of per-row mults in 'mem'
//   0x100.. RESULT    RO per-block result word (peRows*outWidth) at 0x100+blk*4

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'stream_matmul.dart';

const _saVersion = 0x000;
const _saRowBlocks = 0x004;
const _saColTiles = 0x008;
const _saShift = 0x00C;
const _saControl = 0x010;
const _saStatus = 0x014;
const _saWeightBase = 0x018;
const _saActBase = 0x01C;
const _saMultBase = 0x020;

class LoomStreamAccelerator extends BridgeModule
    with HarborDeviceTreeNodeProvider {
  final LoomStreamMatmulConfig config;
  final int versionMagic;

  /// Base address of the CSR slave window in the SoC memory map.
  final int baseAddress;

  /// CSR slave window size (12-bit decode -> 0x1000, but the used regions and
  /// result buffer end well below 0x800).
  static const int windowSize = 0x800;

  late final BusSlavePort bus;
  late final WishboneInterface mem;

  LoomStreamAccelerator({
    this.baseAddress = 0x10000000,
    this.config = const LoomStreamMatmulConfig(),
    this.versionMagic = 0x4C4F4F4D,
    BusProtocol protocol = BusProtocol.wishbone,
    String? name,
  }) : super('LoomStreamAccelerator', name: name ?? 'loom_stream_accelerator') {
    config.validate();
    final cfg = config;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    // CSR slave address width matches the SoC fabric (the low bits decode the
    // registers. Only addr[11:0] are used by the Case logic below).
    bus = BusSlavePort.create(
      module: this,
      name: 'bus',
      protocol: protocol,
      addressWidth: cfg.addressWidth,
      dataWidth: 32,
    );

    final memRef = addInterface(
      WishboneInterface(
        WishboneConfig(addressWidth: cfg.addressWidth, dataWidth: 32),
      ),
      name: 'mem',
      role: PairRole.provider,
    );
    mem = memRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');

    // CSR registers.
    final rowBlocksReg = Logic(name: 'row_blocks_reg', width: 16);
    final colTilesReg = Logic(name: 'col_tiles_reg', width: 16);
    final shiftReg = Logic(name: 'shift_reg', width: cfg.shiftWidth);
    final weightBaseReg = Logic(
      name: 'weight_base_reg',
      width: cfg.addressWidth,
    );
    final actBaseReg = Logic(name: 'act_base_reg', width: cfg.addressWidth);
    final multBaseReg = Logic(name: 'mult_base_reg', width: cfg.addressWidth);
    final startPulse = Logic(name: 'start_pulse');
    final doneReg = Logic(name: 'done_reg');

    // Inner streaming matmul (the bus master).
    final mm = LoomStreamMatmul(config: cfg);
    mm.input('clk').srcConnection! <= clk;
    mm.input('reset').srcConnection! <= reset;
    mm.input('start').srcConnection! <= startPulse;
    mm.input('row_blocks').srcConnection! <= rowBlocksReg;
    mm.input('col_tiles').srcConnection! <= colTilesReg;
    mm.input('shift').srcConnection! <= shiftReg;
    mm.input('weight_base').srcConnection! <= weightBaseReg;
    mm.input('act_base').srcConnection! <= actBaseReg;
    mm.input('mult_base').srcConnection! <= multBaseReg;

    // Forward the inner master to the external 'mem' interface.
    mem.cyc <= mm.output('bus_CYC');
    mem.stb <= mm.output('bus_STB');
    mem.we <= mm.output('bus_WE');
    mem.adr <= mm.output('bus_ADR');
    mem.datMosi <= mm.output('bus_DAT_MOSI');
    mem.sel <= mm.output('bus_SEL');
    mm.input('bus_ACK').srcConnection! <= mem.ack;
    mm.input('bus_DAT_MISO').srcConnection! <= mem.datMiso;

    final mmBusy = mm.output('busy');
    final mmDone = mm.output('done');
    final mmResultValid = mm.output('result_valid');
    final mmResultBlock = mm.output('result_block');
    final mmResult = mm.output('result');

    // Result buffer: one word per row-block.
    final resultBuf = List.generate(
      cfg.maxRowBlocks,
      (i) => Logic(name: 'result_$i', width: 32),
    );
    final resultWrites = <Conditional>[
      for (var b = 0; b < cfg.maxRowBlocks; b++)
        If(
          mmResultBlock.eq(Const(b, width: 16)),
          then: [resultBuf[b] < mmResult.zeroExtend(32)],
        ),
    ];
    final resultReads = <Conditional>[
      for (var b = 0; b < cfg.maxRowBlocks; b++)
        If(
          bus.addr.getRange(2, 8).eq(Const(b, width: 6)),
          then: [bus.dataOut < resultBuf[b]],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          rowBlocksReg < Const(0, width: 16),
          colTilesReg < Const(0, width: 16),
          shiftReg < Const(0, width: cfg.shiftWidth),
          weightBaseReg < Const(0, width: cfg.addressWidth),
          actBaseReg < Const(0, width: cfg.addressWidth),
          multBaseReg < Const(0, width: cfg.addressWidth),
          startPulse < Const(0),
          doneReg < Const(0),
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
          for (final r in resultBuf) r < Const(0, width: 32),
        ],
        orElse: [
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
          startPulse < Const(0), // default: one-cycle pulse
          // Done is sticky: set when the matmul finishes, cleared on start.
          If(mmDone, then: [doneReg < Const(1)]),

          // Capture streamed results.
          If(mmResultValid, then: resultWrites),

          // Bus register interface.
          If(
            bus.stb & ~bus.ack,
            then: [
              bus.ack < Const(1),
              Case(bus.addr.getRange(8, 12), [
                CaseItem(Const(0x0, width: 4), [
                  Case(bus.addr.getRange(0, 8), [
                    CaseItem(Const(_saVersion, width: 8), [
                      bus.dataOut < Const(versionMagic, width: 32),
                    ]),
                    CaseItem(Const(_saRowBlocks, width: 8), [
                      If(
                        bus.we,
                        then: [rowBlocksReg < bus.dataIn.getRange(0, 16)],
                        orElse: [bus.dataOut < rowBlocksReg.zeroExtend(32)],
                      ),
                    ]),
                    CaseItem(Const(_saColTiles, width: 8), [
                      If(
                        bus.we,
                        then: [colTilesReg < bus.dataIn.getRange(0, 16)],
                        orElse: [bus.dataOut < colTilesReg.zeroExtend(32)],
                      ),
                    ]),
                    CaseItem(Const(_saShift, width: 8), [
                      If(
                        bus.we,
                        then: [
                          shiftReg < bus.dataIn.getRange(0, cfg.shiftWidth),
                        ],
                        orElse: [bus.dataOut < shiftReg.zeroExtend(32)],
                      ),
                    ]),
                    CaseItem(Const(_saControl, width: 8), [
                      If(
                        bus.we & bus.dataIn[0],
                        then: [startPulse < Const(1), doneReg < Const(0)],
                      ),
                    ]),
                    CaseItem(Const(_saStatus, width: 8), [
                      bus.dataOut <
                          [Const(0, width: 30), doneReg, mmBusy].swizzle(),
                    ]),
                    CaseItem(Const(_saWeightBase, width: 8), [
                      If(
                        bus.we,
                        then: [
                          weightBaseReg <
                              bus.dataIn.getRange(0, cfg.addressWidth),
                        ],
                        orElse: [bus.dataOut < weightBaseReg.zeroExtend(32)],
                      ),
                    ]),
                    CaseItem(Const(_saActBase, width: 8), [
                      If(
                        bus.we,
                        then: [
                          actBaseReg < bus.dataIn.getRange(0, cfg.addressWidth),
                        ],
                        orElse: [bus.dataOut < actBaseReg.zeroExtend(32)],
                      ),
                    ]),
                    CaseItem(Const(_saMultBase, width: 8), [
                      If(
                        bus.we,
                        then: [
                          multBaseReg <
                              bus.dataIn.getRange(0, cfg.addressWidth),
                        ],
                        orElse: [bus.dataOut < multBaseReg.zeroExtend(32)],
                      ),
                    ]),
                  ]),
                ]),
                // Result region (RO).
                CaseItem(Const(0x1, width: 4), [...resultReads]),
              ]),
            ],
          ),
        ],
      ),
    ]);
  }

  @override
  HarborDeviceTreeNode get dtNode => HarborDeviceTreeNode(
    compatible: ['midstall,loom-stream-accelerator'],
    reg: BusAddressRange(baseAddress, windowSize),
    properties: {
      'midstall,pe-rows': config.peRows,
      'midstall,pe-cols': config.peCols,
      '#address-cells': 1,
      '#size-cells': 1,
    },
  );
}
