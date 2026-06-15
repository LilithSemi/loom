import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

/// Wishbone-master burst reader: the memory-access primitive the Loom
/// accelerator uses to stream weights/activations out of a memory (an on-chip
/// [HarborSram] scratchpad, or Harbor's DDR3 controller [HarborDdrController]).
///
/// On a `start` pulse it reads `count` consecutive 32-bit words beginning at
/// byte address `base` (word-aligned, +4 per word), emitting each on `word_out`
/// with a one-cycle `word_valid`, and pulses `done` after the last word. Single
/// outstanding transaction, matching Harbor's DDR/SRAM Wishbone slaves.
///
/// Ports (created with [createPort]/[addOutput], connected externally):
///   in : clk, reset, start, base[addressWidth], count[countWidth]
///   out: word_out[dataWidth], word_valid, done, busy
///   bus: Wishbone master interface (provider role)
class LoomWbReader extends BridgeModule {
  /// Wishbone master interface.
  late final WishboneInterface bus;

  LoomWbReader({
    int addressWidth = 32,
    int dataWidth = 32,
    int countWidth = 16,
    String? name,
  }) : super('LoomWbReader', name: name ?? 'loom_wb_reader') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('base', PortDirection.input, width: addressWidth);
    createPort('count', PortDirection.input, width: countWidth);

    final wordOut = addOutput('word_out', width: dataWidth);
    final wordValid = addOutput('word_valid');
    final done = addOutput('done');
    final busy = addOutput('busy');

    final busRef = addInterface(
      WishboneInterface(
        WishboneConfig(addressWidth: addressWidth, dataWidth: dataWidth),
      ),
      name: 'bus',
      role: PairRole.provider, // master
    );
    bus = busRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');
    final base = input('base');
    final count = input('count');

    // State.
    final idx = Logic(name: 'idx', width: countWidth);
    final latchedCount = Logic(name: 'latched_count', width: countWidth);
    final cyc = Logic(name: 'cyc');
    final stb = Logic(name: 'stb');
    final adr = Logic(name: 'adr', width: addressWidth);
    final wvalid = Logic(name: 'wvalid');
    final doneReg = Logic(name: 'done_reg');

    final active = cyc; // cyc high == a transaction is in flight
    final ackNow = cyc & stb & bus.ack;
    final lastWord = idx.eq(latchedCount - Const(1, width: countWidth));

    Sequential(clk, [
      If(
        reset,
        then: [
          idx < Const(0, width: countWidth),
          latchedCount < Const(0, width: countWidth),
          cyc < Const(0),
          stb < Const(0),
          adr < Const(0, width: addressWidth),
          wvalid < Const(0),
          doneReg < Const(0),
          wordOut < Const(0, width: dataWidth),
        ],
        orElse: [
          wvalid < Const(0),
          doneReg < Const(0),
          If(
            ~active,
            then: [
              If(
                start & count.gt(Const(0, width: countWidth)),
                then: [
                  idx < Const(0, width: countWidth),
                  latchedCount < count,
                  adr < base,
                  cyc < Const(1),
                  stb < Const(1),
                ],
              ),
            ],
            orElse: [
              If(
                ackNow,
                then: [
                  wordOut < bus.datMiso,
                  wvalid < Const(1),
                  If(
                    lastWord,
                    then: [cyc < Const(0), stb < Const(0), doneReg < Const(1)],
                    orElse: [
                      idx < (idx + Const(1, width: countWidth)),
                      adr < (adr + Const(4, width: addressWidth)),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    ]);

    bus.cyc <= cyc;
    bus.stb <= stb;
    bus.we <= Const(0);
    bus.adr <= adr;
    bus.datMosi <= Const(0, width: dataWidth);
    bus.sel <= Const((1 << (dataWidth ~/ 8)) - 1, width: dataWidth ~/ 8);
    wordValid <= wvalid;
    done <= doneReg;
    busy <= active;
  }
}
