// Sim sanity for the UART ECHO loopback (uartecho transport).
//
// Mirrors the hand-written LoomUartEchoTop glue in ROHD: a single LoomUart with
// its RX stream latched into a one-deep holding register and fed straight back
// into its TX stream. Feeds a byte in on rx and asserts it comes back on tx,
// proving the echo gateware logic is correct independent of the real board.

import 'dart:async';

import 'package:loom/src/hw/uart_bridge.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

// Fast UART config: 8 clocks/bit. Same bit-level logic as the 48 MHz/115200
// (divisor 417) build. Only the divisor differs.
const _simConfig = LoomUartConfig(
  busAddressWidth: 12,
  busDataWidth: 32,
  clockFrequency: 8,
  baudRate: 1, // divisor = 8
);

// ROHD model of LoomUartEchoTop's echo glue: LoomUart + one-deep RX holding
// register wired RX -> TX.
class _EchoTop extends BridgeModule {
  _EchoTop({required LoomUartConfig config})
    : super('_EchoTop', name: 'echo_top') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('rx', PortDirection.input);

    final clk = input('clk');
    final reset = input('reset');

    final uart = LoomUart(config: config, name: 'echo_uart');
    addSubModule(uart);
    uart.input('clk').srcConnection! <= clk;
    uart.input('reset').srcConnection! <= reset;
    uart.input('rx').srcConnection! <= input('rx');

    final rxHold = Logic(name: 'rx_hold', width: 8);
    final rxHoldFull = Logic(name: 'rx_hold_full');
    final txReady = uart.output('tx_ready');
    final txAccept = uart.output('tx_accept');

    Sequential(clk, [
      If(
        reset,
        then: [rxHold < Const(0, width: 8), rxHoldFull < Const(0)],
        orElse: [
          If(
            uart.output('rx_valid'),
            then: [rxHold < uart.output('rx_data'), rxHoldFull < Const(1)],
            orElse: [
              If(txAccept, then: [rxHoldFull < Const(0)]),
            ],
          ),
        ],
      ),
    ]);

    uart.input('tx_data').srcConnection! <= rxHold;
    uart.input('tx_valid').srcConnection! <= (rxHoldFull & txReady);

    addOutput('tx') <= uart.output('tx');
  }
}

Future<void> _sendByte(Logic rx, Logic clk, int byte, int divisor) async {
  Future<void> bit(int v) async {
    rx.put(v);
    for (var i = 0; i < divisor; i++) {
      await clk.nextPosedge;
    }
  }

  await bit(0); // start
  for (var i = 0; i < 8; i++) {
    await bit((byte >> i) & 1);
  }
  await bit(1); // stop
}

Future<int?> _recvByte(
  Logic tx,
  Logic clk,
  int divisor, {
  int guardCycles = 5000,
}) async {
  var guard = 0;
  while (tx.value.toInt() == 1 && guard < guardCycles) {
    guard++;
    await clk.nextPosedge;
  }
  if (guard >= guardCycles) return null;
  for (var i = 0; i < divisor ~/ 2; i++) {
    await clk.nextPosedge;
  }
  var b = 0;
  for (var i = 0; i < 8; i++) {
    for (var j = 0; j < divisor; j++) {
      await clk.nextPosedge;
    }
    b |= (tx.value.toInt() & 1) << i;
  }
  return b;
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('echo: a byte fed in on rx comes straight back out on tx', () async {
    final top = _EchoTop(config: _simConfig);

    final clk = SimpleClockGenerator(10).clk;
    top.port('clk').getsLogic(clk);
    await top.build();

    final reset = top.input('reset');
    final rx = top.input('rx');
    final tx = top.output('tx');
    rx.put(1);
    reset.put(1);

    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.put(0);

    final divisor = _simConfig.baudDivisor;

    for (final sent in [0x55, 0x41, 0x00, 0xFF, 0xA5]) {
      // idle gap
      rx.put(1);
      for (var i = 0; i < divisor; i++) {
        await clk.nextPosedge;
      }
      // Drive the byte in, and concurrently decode the echo on tx.
      final recvFut = _recvByte(tx, clk, divisor);
      await _sendByte(rx, clk, sent, divisor);
      final got = await recvFut;
      expect(
        got,
        equals(sent),
        reason: 'echo of 0x${sent.toRadixString(16)} should round-trip',
      );
    }

    await Simulator.endSimulation();
  });
}
