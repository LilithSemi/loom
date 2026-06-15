// Tests for LoomUart + LoomUartBridge (a UART command transport).
//
// LoomUartBridge drives LoomAccelerator over an 8N1 UART using the same
// command protocol as USB (header {opcode:u8, addr:u32 LE, len:u16 LE}; WRITE
// 0x01 + data, READ 0x02 returns len bytes), reusing LoomUsbCmdEngine. UART
// adds the line engine and byte-stream glue.
//
// Tests bit-bang the rx pad and decode the tx pad to exercise real framing
// (start / 8 data LSB-first / stop), then assert:
//   * READ VERSION @0x000 -> 0x4C4F4F4D ('LOOM'), LE on the wire
//   * WRITE ROWS=7 @0x004 then READ ROWS -> 7 (a RW round-trip)

import 'dart:async';

import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/hw/uart_bridge.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

// Fast UART config so sim runs in a reasonable number of cycles (8 clocks per
// bit). Bit-level logic matches the 48 MHz / 115200 (divisor 417) build.
const _simConfig = LoomUartConfig(
  busAddressWidth: 12,
  busDataWidth: 32,
  clockFrequency: 8,
  baudRate: 1, // divisor = round(8/1) = 8 clocks/bit
);

List<int> _le16(int v) => [v & 0xFF, (v >> 8) & 0xFF];
List<int> _le32(int v) => [
  v & 0xFF,
  (v >> 8) & 0xFF,
  (v >> 16) & 0xFF,
  (v >> 24) & 0xFF,
];

const _opWrite = 0x01;
const _opRead = 0x02;
const _opWriteStream = 0x03;

List<int> _readCmd(int addr, int len) => [
  _opRead,
  ..._le32(addr),
  ..._le16(len),
];
List<int> _writeCmd(int addr, List<int> data) => [
  _opWrite,
  ..._le32(addr),
  ..._le16(data.length),
  ...data,
];
List<int> _writeStreamCmd(int addr, List<int> data) => [
  _opWriteStream,
  ..._le32(addr),
  ..._le16(data.length),
  ...data,
];

// Top: bridge master + real accelerator slave.
class _BridgeAccelTop extends BridgeModule {
  final LoomUartBridge bridge;
  final LoomAccelerator accel;

  _BridgeAccelTop({required this.bridge, required this.accel})
    : super('_BridgeAccelTop', name: 'bridge_accel_top') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('rx', PortDirection.input);

    addSubModule(bridge);
    addSubModule(accel);

    connectPorts(port('clk'), bridge.port('clk'));
    connectPorts(port('reset'), bridge.port('reset'));
    connectPorts(port('rx'), bridge.port('rx'));
    connectPorts(port('clk'), accel.port('clk'));
    connectPorts(port('reset'), accel.port('reset'));

    connectInterfaces(bridge.interface('bus'), accel.interface('bus'));

    addOutput('tx') <= bridge.output('tx');
    addOutput('busy') <= bridge.output('busy');
  }
}

// Bit-bang one byte onto the rx line at the baud period (start, 8 data LSB
// first, stop). `divisor` clocks per bit.
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

Future<void> _sendBytes(
  Logic rx,
  Logic clk,
  List<int> bytes,
  int divisor,
) async {
  rx.put(1); // idle high
  // A short idle gap before the frame.
  for (var i = 0; i < divisor; i++) {
    await clk.nextPosedge;
  }
  for (final b in bytes) {
    await _sendByte(rx, clk, b, divisor);
  }
  rx.put(1);
}

// Watch the tx line and decode `n` bytes (8N1). Samples each bit at its center.
Future<List<int>> _recvBytes(
  Logic tx,
  Logic clk,
  int n,
  int divisor, {
  // Bounded so a stalled TX fails the test fast instead of spinning millions of
  // cycles. The command engine starts emitting within a few hundred cycles of
  // the READ frame draining, so a few thousand cycles of headroom is ample at
  // the sim divisor.
  int guardCycles = 50000,
}) async {
  final out = <int>[];
  var guard = 0;
  while (out.length < n && guard < guardCycles) {
    // Wait for a falling edge (start bit) on an idle-high line.
    while (tx.value.toInt() == 1 && guard < guardCycles) {
      guard++;
      await clk.nextPosedge;
    }
    if (guard >= guardCycles) break;
    // We are at (or just after) the falling edge. Advance to the center of the
    // start bit, then sample each subsequent bit at its center.
    for (var i = 0; i < divisor ~/ 2; i++) {
      await clk.nextPosedge;
    }
    // Now at center of start bit (should be 0). Step to each data bit center.
    var b = 0;
    for (var i = 0; i < 8; i++) {
      for (var j = 0; j < divisor; j++) {
        await clk.nextPosedge;
      }
      b |= (tx.value.toInt() & 1) << i;
    }
    out.add(b);
    // Advance to the MIDDLE of the stop bit, then keep stepping until the line
    // is high (idle) again, so the next iteration re-arms cleanly on the next
    // falling edge even when device frames are back-to-back. Bounded by guard.
    for (var j = 0; j < divisor; j++) {
      await clk.nextPosedge;
    }
    while (tx.value.toInt() == 0 && guard < guardCycles) {
      guard++;
      await clk.nextPosedge;
    }
  }
  return out;
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('LoomUartConfig', () {
    test('48 MHz / 115200 -> divisor 417', () {
      const c = LoomUartConfig();
      expect(c.baudDivisor, equals(417));
    });
    test('rejects a bad data width', () {
      expect(
        () => const LoomUartConfig(busDataWidth: 12).validate(),
        throwsA(isA<ArgumentError>()),
      );
    });
    test('rejects a divisor < 2', () {
      expect(
        () =>
            const LoomUartConfig(clockFrequency: 100, baudRate: 100).validate(),
        throwsA(isA<ArgumentError>()),
      );
    });
    test('accepts the sim config', () {
      expect(_simConfig.validate, returnsNormally);
      expect(_simConfig.baudDivisor, equals(8));
    });
  });

  group('LoomUart byte-stream loopback', () {
    test('a byte fed in on rx appears on rx_data/rx_valid', () async {
      final uart = LoomUart(config: _simConfig, name: 'uart_lb');
      final clk = SimpleClockGenerator(10).clk;
      uart.port('clk').getsLogic(clk);
      await uart.build();

      final reset = uart.input('reset');
      final rx = uart.input('rx');
      uart.input('tx_data').put(0);
      uart.input('tx_valid').put(0);
      rx.put(1);
      reset.put(1);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.put(0);
      await clk.nextPosedge;

      var got = -1;
      final divisor = _simConfig.baudDivisor;
      // Send 0xA5 and watch rx_valid.
      final sender = () async {
        await _sendByte(rx, clk, 0xA5, divisor);
      }();
      var guard = 0;
      while (got < 0 && guard < 100000) {
        guard++;
        if (uart.output('rx_valid').value.toInt() == 1) {
          got = uart.output('rx_data').value.toInt();
        }
        await clk.nextPosedge;
      }
      await sender;
      await Simulator.endSimulation();
      expect(got, equals(0xA5));
    });
  });

  group('LoomUartBridge -> real LoomAccelerator', () {
    test('READ VERSION @0x000 returns 0x4C4F4F4D ("LOOM") over UART', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final bridge = LoomUartBridge(config: _simConfig, name: 'bridge');
      final top = _BridgeAccelTop(bridge: bridge, accel: accel);

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
      await clk.nextPosedge;

      final divisor = _simConfig.baudDivisor;
      // Start the TX receiver first (concurrent with the send).
      final recv = _recvBytes(tx, clk, 4, divisor);
      await _sendBytes(rx, clk, _readCmd(0x000, 4), divisor);
      final resp = await recv;
      await Simulator.endSimulation();

      // VERSION = 0x4C4F4F4D, returned little-endian on the wire.
      expect(resp, equals([0x4D, 0x4F, 0x4F, 0x4C]));
    });

    test('WRITE ROWS=7 then READ ROWS round-trips over UART', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final bridge = LoomUartBridge(config: _simConfig, name: 'bridge');
      final top = _BridgeAccelTop(bridge: bridge, accel: accel);

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
      await clk.nextPosedge;

      final divisor = _simConfig.baudDivisor;
      // WRITE ROWS=7 (4-byte LE word: 07 00 00 00).
      await _sendBytes(
        rx,
        clk,
        _writeCmd(0x004, [0x07, 0x00, 0x00, 0x00]),
        divisor,
      );
      // Let the write settle (the Wishbone write completes within a few
      // cycles of the last byte draining).
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }
      // READ ROWS back.
      final recv = _recvBytes(tx, clk, 4, divisor);
      await _sendBytes(rx, clk, _readCmd(0x004, 4), divisor);
      final resp = await recv;
      await Simulator.endSimulation();

      expect(resp, equals([0x07, 0x00, 0x00, 0x00]));
    });

    test('WRITE_STREAM holds the address fixed (FIFO push)', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final bridge = LoomUartBridge(config: _simConfig, name: 'bridge');
      final top = _BridgeAccelTop(bridge: bridge, accel: accel);

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
      await clk.nextPosedge;

      final divisor = _simConfig.baudDivisor;
      // Stream two words (3 then 9) to ROWS @0x004. A streaming write holds the
      // address fixed, so both land on 0x004 and the LAST (9) wins. A normal
      // WRITE would increment (3->0x004, 9->0x008), leaving 0x004 == 3.
      await _sendBytes(
        rx,
        clk,
        _writeStreamCmd(0x004, [0x03, 0, 0, 0, 0x09, 0, 0, 0]),
        divisor,
      );
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }
      final recv = _recvBytes(tx, clk, 4, divisor);
      await _sendBytes(rx, clk, _readCmd(0x004, 4), divisor);
      final resp = await recv;
      await Simulator.endSimulation();

      expect(
        resp,
        equals([0x09, 0x00, 0x00, 0x00]),
      ); // last word, not the first
    });
  });
}
