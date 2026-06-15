import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

/// Configuration for [LoomController].
class LoomControllerConfig {
  /// Bus address width in bits.
  final int addressWidth;

  /// Bus data width in bits. Must be 32 (Wishbone constraint here).
  final int dataWidth;

  /// Number of 32-bit scratch registers (must be >= 1).
  final int numScratch;

  /// Version/magic constant returned from the VERSION register.
  /// Default is 0x4C4F4F4D ('LOOM' in ASCII).
  final int versionMagic;

  const LoomControllerConfig({
    this.addressWidth = 8,
    this.dataWidth = 32,
    this.numScratch = 1,
    this.versionMagic = 0x4C4F4F4D,
  });

  /// Validates the configuration. Throws [ArgumentError] on failure.
  void validate() {
    if (addressWidth < 3) {
      throw ArgumentError(
        'LoomControllerConfig.addressWidth must be >= 3, got $addressWidth',
      );
    }
    if (dataWidth != 32) {
      throw ArgumentError(
        'LoomControllerConfig.dataWidth must be 32, got $dataWidth',
      );
    }
    if (numScratch < 1) {
      throw ArgumentError(
        'LoomControllerConfig.numScratch must be >= 1, got $numScratch',
      );
    }
  }
}

/// Register map (word-addressed offsets, each word is 4 bytes).
///
/// Offset  Name     Access  Description
/// 0x00    VERSION  RO      Version/magic constant (default 0x4C4F4F4D 'LOOM')
/// 0x04    CONTROL  RW      Bit 0: start, Bit 1: reset (skeleton)
/// 0x08    STATUS   RO      Bit 0: busy, Bit 1: done (skeleton, wired to 0)
/// 0x0C    SCRATCH  RW      Read/write scratch register
const _loomControllerRegisterMap = HarborDeviceRegisterMap(
  name: 'loom_controller',
  fields: [
    HarborDeviceField(
      name: 'VERSION',
      width: 4,
      offset: 0x00,
      readOnly: true,
      resetValue: 0x4C4F4F4D,
    ),
    HarborDeviceField(name: 'CONTROL', width: 4, offset: 0x04),
    HarborDeviceField(name: 'STATUS', width: 4, offset: 0x08, readOnly: true),
    HarborDeviceField(name: 'SCRATCH', width: 4, offset: 0x0C),
  ],
);

// Word-address constants (byte offset >> 2).
const _regVersion = 0x00 >> 2; // 0
const _regControl = 0x04 >> 2; // 1
const _regStatus = 0x08 >> 2; // 2
const _regScratch = 0x0C >> 2; // 3

/// Loom accelerator host-facing controller.
///
/// A harbor [BridgeModule] exposing a small CSR block over a Wishbone slave.
/// This is the skeleton the host CPU uses to control and query the Loom
/// accelerator fabric.
///
/// Register map (see [_loomControllerRegisterMap]):
/// - 0x00 VERSION  RO  0x4C4F4F4D ('LOOM')
/// - 0x04 CONTROL  RW  bit 0 = start, bit 1 = soft-reset
/// - 0x08 STATUS   RO  bit 0 = busy, bit 1 = done (currently 0)
/// - 0x0C SCRATCH  RW  read/write scratch
class LoomController extends BridgeModule
    with HarborDeviceTreeNodeProvider, HarborSvdPeripheralProvider {
  /// Configuration.
  final LoomControllerConfig config;

  /// Base address in the SoC memory map.
  final int baseAddress;

  /// Wishbone slave port.
  late final BusSlavePort bus;

  LoomController({
    required this.baseAddress,
    this.config = const LoomControllerConfig(),
    BusProtocol protocol = BusProtocol.wishbone,
    String? name,
  }) : super('LoomController', name: name ?? 'loom_controller') {
    config.validate();

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    bus = BusSlavePort.create(
      module: this,
      name: 'bus',
      protocol: protocol,
      addressWidth: config.addressWidth,
      dataWidth: config.dataWidth,
    );

    final clk = input('clk');
    final reset = input('reset');

    // Internal registers.
    final controlReg = Logic(name: 'control_reg', width: 32);
    final scratchReg = Logic(name: 'scratch_reg', width: 32);

    Sequential(clk, [
      If(
        reset,
        then: [
          controlReg < Const(0, width: 32),
          scratchReg < Const(0, width: 32),
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
        ],
        orElse: [
          // Default: clear ack and data each cycle.
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),

          // Clear one-shot bits in CONTROL after they are processed.
          // bit 0 (start) self-clears after one cycle.
          If(
            controlReg[0],
            then: [controlReg < (controlReg & Const(0xFFFFFFFE, width: 32))],
          ),

          If(
            bus.stb & ~bus.ack,
            then: [
              bus.ack < Const(1),

              // The bus carries raw byte addresses; decode bits [5:2] as the
              // word index (same convention as harbor peripherals like efuse).
              Case(bus.addr.getRange(2, 6), [
                // VERSION (RO) - return magic constant.
                CaseItem(Const(_regVersion, width: 4), [
                  bus.dataOut <
                      Const(config.versionMagic, width: 32).zeroExtend(32),
                ]),

                // CONTROL (RW) - write updates the register; read returns it.
                CaseItem(Const(_regControl, width: 4), [
                  If(
                    bus.we,
                    then: [controlReg < bus.dataIn],
                    orElse: [bus.dataOut < controlReg],
                  ),
                ]),

                // STATUS (RO) - busy/done bits (both 0 in skeleton).
                CaseItem(Const(_regStatus, width: 4), [
                  bus.dataOut < Const(0, width: 32),
                ]),

                // SCRATCH (RW) - write then read-back.
                CaseItem(Const(_regScratch, width: 4), [
                  If(
                    bus.we,
                    then: [scratchReg < bus.dataIn],
                    orElse: [bus.dataOut < scratchReg],
                  ),
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
    compatible: ['midstall,loom-controller'],
    reg: BusAddressRange(baseAddress, 0x1000),
    properties: {
      'midstall,version-magic': config.versionMagic,
      'midstall,num-scratch': config.numScratch,
      '#address-cells': 1,
      '#size-cells': 1,
    },
  );

  @override
  HarborSvdPeripheral get svdPeripheral => HarborSvdPeripheral(
    name: 'LOOM_CONTROLLER',
    groupName: 'LOOM',
    description: 'Loom accelerator host-facing controller',
    baseAddress: baseAddress,
    size: 0x1000,
    registers: _loomControllerRegisterMap,
  );
}
