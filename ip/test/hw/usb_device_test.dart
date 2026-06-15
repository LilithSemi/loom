// Tests for LoomUsbDevice + LoomUsbCmdEngine.
//
// LoomUsbDevice is a custom vendor-class full-speed USB device (bDeviceClass
// 0xFF) that exposes TWO bulk endpoints (EP1 OUT host->device, EP1 IN
// device->host) and a command protocol on top of them that performs Wishbone
// READS and WRITES: the command engine is a Wishbone MASTER, so a host can
// write the accelerator's buffers/CSRs, trigger compute, and read results
// back over USB.
//
// Command framing (host -> device, on the bulk OUT stream):
//   header = { opcode:u8, addr:u32 (LE), len:u16 (LE) }  (7 bytes)
//   WRITE (opcode 0x01): header then `len` data bytes. Each byte is written to
//                        Wishbone at addr, addr+1, ... (byte granular).
//   READ  (opcode 0x02): header only. The device fetches `len` bytes from
//                        Wishbone starting at addr and emits them on the bulk IN
//                        stream (resp_data / resp_valid, paced by resp_ready).
//
// The full raw dp/dm enumeration in sim is exercised by harbor's own engine
// tests, so this file tests the net-new parts at the COMMAND/byte-stream
// level: feed the bulk-OUT command bytes directly into the command engine,
// let it drive a REAL LoomAccelerator Wishbone slave, and observe the
// bulk-IN response bytes. Plus an SV-emission test on the full LoomUsbDevice
// that proves the harbor PHY/packet submodules are present.

import 'dart:async';
import 'dart:typed_data';

import 'package:harbor/harbor.dart';
import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/hw/usb_device.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

// Golden helpers (host-side, independent of the RTL).

List<int> _le16(int v) => [v & 0xFF, (v >> 8) & 0xFF];
List<int> _le32(int v) => [
  v & 0xFF,
  (v >> 8) & 0xFF,
  (v >> 16) & 0xFF,
  (v >> 24) & 0xFF,
];

const _opWrite = 0x01;
const _opRead = 0x02;

List<int> _writeCmd(int addr, List<int> data) => [
  _opWrite,
  ..._le32(addr),
  ..._le16(data.length),
  ...data,
];
List<int> _readCmd(int addr, int len) => [
  _opRead,
  ..._le32(addr),
  ..._le16(len),
];

// Golden helpers (copied from accelerator_test.dart to stay bit-exact).
int _roundShift(int prod, int shift) {
  if (shift == 0) return prod;
  final bias = 1 << (shift - 1);
  if (prod >= 0) return (prod + bias) >> shift;
  return -((-prod + bias) >> shift);
}

int requantRef(int acc, int mult, int shift) {
  final rounded = _roundShift(acc * mult, shift);
  if (rounded < -127) return -127;
  if (rounded > 127) return 127;
  return rounded;
}

Int32List matmulInt(Int8List w, Int8List x, int rows, int cols) {
  final acc = Int32List(rows);
  for (var r = 0; r < rows; r++) {
    var sum = 0;
    for (var c = 0; c < cols; c++) {
      sum += w[r * cols + c] * x[c];
    }
    acc[r] = sum;
  }
  return acc;
}

int int8FromBusWord(int word, int byteIndex) {
  final raw = (word >> (byteIndex * 8)) & 0xFF;
  return raw >= 0x80 ? raw - 256 : raw;
}

// Top wrapper: a LoomUsbCmdEngine (bulk command bridge + Wishbone master)
// connected to a REAL LoomAccelerator slave. The command-byte input stream
// and response-byte output stream are pulled to the top so the testbench can
// feed the bulk-OUT bytes and read the bulk-IN bytes. Single clock domain
// (the SoC keeps the USB-rate clock single).
class _CmdAccelTop extends BridgeModule {
  final LoomUsbCmdEngine engine;
  final LoomAccelerator accel;

  _CmdAccelTop({required this.engine, required this.accel})
    : super('_CmdAccelTop', name: 'cmd_accel_top') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    // Bulk-OUT command byte stream input.
    createPort('cmd_data', PortDirection.input, width: 8);
    createPort('cmd_valid', PortDirection.input);
    // Bulk-IN response byte stream consumer back-pressure.
    createPort('resp_ready', PortDirection.input);

    addSubModule(engine);
    addSubModule(accel);

    connectPorts(port('clk'), engine.port('clk'));
    connectPorts(port('reset'), engine.port('reset'));
    connectPorts(port('clk'), accel.port('clk'));
    connectPorts(port('reset'), accel.port('reset'));

    connectPorts(port('cmd_data'), engine.port('cmd_data'));
    connectPorts(port('cmd_valid'), engine.port('cmd_valid'));
    connectPorts(port('resp_ready'), engine.port('resp_ready'));

    connectInterfaces(engine.interface('bus'), accel.interface('bus'));

    addOutput('cmd_ready') <= engine.output('cmd_ready');
    addOutput('resp_data', width: 8) <= engine.output('resp_data');
    addOutput('resp_valid') <= engine.output('resp_valid');
    addOutput('busy') <= engine.output('busy');
  }
}

// Feed one command's bytes into the bulk-OUT stream, honoring cmd_ready.
Future<void> _feedCmd(
  _CmdAccelTop top,
  Logic clk,
  Logic cmdData,
  Logic cmdValid,
  List<int> bytes,
) async {
  for (final b in bytes) {
    var guard = 0;
    while (top.output('cmd_ready').value.toInt() == 0 && guard < 200000) {
      guard++;
      await clk.nextPosedge;
    }
    cmdData.put(b);
    cmdValid.put(1);
    await clk.nextPosedge;
    cmdValid.put(0);
    cmdData.put(0);
    // One idle cycle between bytes (the engine can NAK / process).
    await clk.nextPosedge;
  }
}

// Collect `n` response bytes from the bulk-IN stream with a per-byte valid/ready
// handshake. The command engine drives resp_data COMBINATIONALLY (byte[count])
// and holds each byte until it sees resp_ready, advancing the next cycle. So we
// must consume ONE byte per resp_ready pulse: on a cycle where resp_valid is
// high, latch resp_data and assert resp_ready for that cycle (the engine accepts
// and advances), then drop resp_ready and wait for the next byte. Holding
// resp_ready continuously would let the engine race ahead of our async sampling
// and we would miss/duplicate bytes. Pulsing per byte is the correct contract
// (the same one the UART transport and the harbor EP1-IN assembler use).
Future<List<int>> _collectResp(
  _CmdAccelTop top,
  Logic clk,
  Logic respReady,
  int n,
) async {
  final out = <int>[];
  var guard = 0;
  respReady.put(0);
  while (out.length < n && guard < 2000000) {
    guard++;
    final validNow =
        top.output('resp_valid').value.isValid &&
        top.output('resp_valid').value.toInt() == 1;
    if (validNow) {
      // Latch the currently-held byte and accept it this cycle (pulse ready so
      // the engine advances to the next byte on the coming edge).
      out.add(top.output('resp_data').value.toInt());
      respReady.put(1);
      await clk.nextPosedge;
      respReady.put(0);
      // Let the combinational resp_data settle to the next byte before sampling.
      await clk.nextPosedge;
    } else {
      await clk.nextPosedge;
    }
  }
  respReady.put(0);
  return out;
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // Config validation.
  group('LoomUsbDeviceConfig validation', () {
    test('rejects a non-[8,16,32,64] data width', () {
      expect(
        () => LoomUsbDeviceConfig(busDataWidth: 12).validate(),
        throwsA(isA<ArgumentError>()),
      );
    });
    test('rejects an out-of-range address width', () {
      expect(
        () => LoomUsbDeviceConfig(busAddressWidth: 0).validate(),
        throwsA(isA<ArgumentError>()),
      );
    });
    test('accepts a sane default', () {
      expect(() => const LoomUsbDeviceConfig().validate(), returnsNormally);
    });
  });

  // Descriptor ROM: vendor-class device descriptor + bulk-endpoint config.
  group('LoomUsbDescriptorRom', () {
    test(
      'device descriptor is vendor-class (0xFF) with the configured VID/PID',
      () {
        final dev = LoomUsbDescriptorRom.deviceDescriptor;
        expect(dev.length, equals(18));
        expect(dev[0], equals(18)); // bLength
        expect(dev[1], equals(0x01)); // DEVICE
        expect(dev[4], equals(0xFF)); // bDeviceClass = vendor-specific
        expect(dev[7], equals(64)); // bMaxPacketSize0
      },
    );

    test('config has one interface with two bulk endpoints (EP1 OUT + EP1 IN), '
        'maxPacketSize 64', () {
      final cfg = LoomUsbDescriptorRom.configDescriptorBulk;
      // wTotalLength == real byte count.
      final wTotal = cfg[2] | (cfg[3] << 8);
      expect(wTotal, equals(cfg.length));
      // config(9) + interface(9) + ep(7) + ep(7) = 32.
      expect(cfg.length, equals(32));
      // Interface descriptor at offset 9.
      expect(cfg[9 + 1], equals(0x04)); // INTERFACE
      expect(cfg[9 + 4], equals(2)); // bNumEndpoints = 2
      expect(cfg[9 + 5], equals(0xFF)); // bInterfaceClass = vendor
      // EP1 OUT descriptor at offset 18.
      final epOut = cfg.sublist(18, 25);
      expect(epOut[1], equals(0x05)); // ENDPOINT
      expect(epOut[2], equals(0x01)); // EP1 OUT (dir bit 7 = 0)
      expect(epOut[3], equals(0x02)); // bmAttributes = bulk
      expect(epOut[4] | (epOut[5] << 8), equals(64)); // wMaxPacketSize
      // EP1 IN descriptor at offset 25.
      final epIn = cfg.sublist(25, 32);
      expect(epIn[1], equals(0x05));
      expect(epIn[2], equals(0x81)); // EP1 IN (dir bit 7 = 1)
      expect(epIn[3], equals(0x02));
      expect(epIn[4] | (epIn[5] << 8), equals(64));
    });

    test('ROM presents device descriptor bytes at offset, present=1', () async {
      final rom = LoomUsbDescriptorRom(name: 'rom_t');
      await rom.build();
      rom.input('desc_type').put(0x01);
      rom.input('desc_index').put(0x00);
      final dev = LoomUsbDescriptorRom.deviceDescriptor;
      expect(rom.output('present').value.toInt(), equals(1));
      expect(rom.output('length').value.toInt(), equals(dev.length));
      for (var i = 0; i < dev.length; i++) {
        rom.input('offset').put(i);
        expect(
          rom.output('data').value.toInt(),
          equals(dev[i]),
          reason: 'device descriptor byte $i',
        );
      }
      // An absent descriptor reports present=0.
      rom.input('desc_type').put(0x09);
      expect(rom.output('present').value.toInt(), equals(0));
    });

    test('control-only (bulkEndpoints:false) config advertises 0 endpoints, '
        'wTotalLength 18', () {
      final cfg = LoomUsbDescriptorRom.configDescriptorNoBulk;
      // wTotalLength == real byte count == config(9) + interface(9) = 18.
      final wTotal = cfg[2] | (cfg[3] << 8);
      expect(wTotal, equals(cfg.length));
      expect(cfg.length, equals(18));
      expect(cfg[1], equals(0x02)); // CONFIGURATION
      // Interface descriptor at offset 9.
      expect(cfg[9 + 0], equals(9)); // bLength
      expect(cfg[9 + 1], equals(0x04)); // INTERFACE
      expect(cfg[9 + 4], equals(0)); // bNumEndpoints = 0
      expect(cfg[9 + 5], equals(0xFF)); // bInterfaceClass = vendor
    });

    test(
      'control-only ROM serves the 0-endpoint config byte-for-byte',
      () async {
        final rom = LoomUsbDescriptorRom(
          name: 'rom_nobulk',
          bulkEndpoints: false,
        );
        await rom.build();
        rom.input('desc_type').put(0x02);
        rom.input('desc_index').put(0x00);
        final cfg = LoomUsbDescriptorRom.configDescriptorNoBulk;
        expect(rom.output('present').value.toInt(), equals(1));
        expect(rom.output('length').value.toInt(), equals(cfg.length));
        for (var i = 0; i < cfg.length; i++) {
          rom.input('offset').put(i);
          expect(
            rom.output('data').value.toInt(),
            equals(cfg[i]),
            reason: 'nobulk config byte $i',
          );
        }
      },
    );

    test(
      'descriptorEntries(bulkEndpoints:false) injects the 0-endpoint config',
      () {
        final ents = LoomUsbDescriptorRom.descriptorEntries(
          bulkEndpoints: false,
        );
        final cfg = ents.firstWhere((e) => e.type == 0x02).bytes;
        expect(cfg.length, equals(18));
        expect(cfg[9 + 4], equals(0)); // bNumEndpoints = 0
        // device + 4 strings (langid + 3) still present.
        expect(ents.where((e) => e.type == 0x03).length, equals(4));
      },
    );
  });

  // WRITE command lands on the bus: stream a WRITE command that writes a few
  // bytes to the accelerator's ROWS/COLS CSRs, then read them back over the bus
  // with READ commands.
  group('LoomUsbCmdEngine WRITE -> real LoomAccelerator slave', () {
    test('WRITE to ROWS/COLS CSRs lands; READ reads them back', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final engine = LoomUsbCmdEngine(
        config: const LoomUsbDeviceConfig(
          busAddressWidth: 12,
          busDataWidth: 32,
        ),
        name: 'cmd_engine',
      );
      final top = _CmdAccelTop(engine: engine, accel: accel);

      final clk = SimpleClockGenerator(10).clk;
      top.port('clk').getsLogic(clk);
      await top.build();

      final reset = top.input('reset');
      final cmdData = top.input('cmd_data');
      final cmdValid = top.input('cmd_valid');
      final respReady = top.input('resp_ready');

      Simulator.setMaxSimTime(20000000);
      unawaited(Simulator.run());

      reset.inject(1);
      cmdData.inject(0);
      cmdValid.inject(0);
      respReady.inject(0);
      for (var i = 0; i < 5; i++) {
        await clk.nextPosedge;
      }
      reset.put(0);
      await clk.nextPosedge;

      // Write ROWS=0x00000003 (4 bytes) at 0x004, COLS=0x00000005 at 0x008.
      await _feedCmd(top, clk, cmdData, cmdValid, _writeCmd(0x004, _le32(3)));
      await _feedCmd(top, clk, cmdData, cmdValid, _writeCmd(0x008, _le32(5)));

      // Wait for the engine to go idle.
      var g = 0;
      while (top.output('busy').value.toInt() == 1 && g < 100000) {
        g++;
        await clk.nextPosedge;
      }

      // READ ROWS back (4 bytes from 0x004).
      await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x004, 4));
      final rows = await _collectResp(top, clk, respReady, 4);
      expect(rows, equals(_le32(3)), reason: 'ROWS read back == 3');

      // READ COLS back.
      await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x008, 4));
      final cols = await _collectResp(top, clk, respReady, 4);
      expect(cols, equals(_le32(5)), reason: 'COLS read back == 5');

      // READ VERSION (RO CSR) back -> the magic.
      await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x000, 4));
      final ver = await _collectResp(top, clk, respReady, 4);
      expect(ver, equals(_le32(0x4C4F4F4D)), reason: 'VERSION magic');

      await Simulator.endSimulation();
    });
  });

  // READ-back of a buffer region: write into the weight buffer then read it
  // back.
  group('LoomUsbCmdEngine buffer write + read-back', () {
    test(
      'WRITE 8 bytes into the weight buffer, READ them back byte-exact',
      () async {
        final accel = LoomAccelerator(
          config: const LoomAcceleratorConfig(baseAddress: 0x0),
          name: 'accel',
        );
        final engine = LoomUsbCmdEngine(
          config: const LoomUsbDeviceConfig(
            busAddressWidth: 12,
            busDataWidth: 32,
          ),
          name: 'cmd_engine',
        );
        final top = _CmdAccelTop(engine: engine, accel: accel);

        final clk = SimpleClockGenerator(10).clk;
        top.port('clk').getsLogic(clk);
        await top.build();

        final reset = top.input('reset');
        final cmdData = top.input('cmd_data');
        final cmdValid = top.input('cmd_valid');
        final respReady = top.input('resp_ready');

        Simulator.setMaxSimTime(20000000);
        unawaited(Simulator.run());

        reset.inject(1);
        cmdData.inject(0);
        cmdValid.inject(0);
        respReady.inject(0);
        for (var i = 0; i < 5; i++) {
          await clk.nextPosedge;
        }
        reset.put(0);
        await clk.nextPosedge;

        // The accelerator's weight/activation/rowMult buffers are WRITE-ONLY over
        // the bus and the result buffer is READ-ONLY, so the round-trip is proven
        // against the RW CSR region (ROWS 0x004 + COLS 0x008): write 8 bytes
        // spanning ROWS and COLS as two words, then READ all 8 bytes back
        // byte-exact.
        final bytes = <int>[0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00];
        await _feedCmd(top, clk, cmdData, cmdValid, _writeCmd(0x004, bytes));

        var g = 0;
        while (top.output('busy').value.toInt() == 1 && g < 100000) {
          g++;
          await clk.nextPosedge;
        }

        await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x004, 8));
        final got = await _collectResp(top, clk, respReady, 8);
        expect(got, equals(bytes), reason: 'CSR round-trip byte-exact');

        await Simulator.endSimulation();
      },
    );
  });

  // End-to-end: program a tiny matmul, trigger compute via CONTROL.start, poll
  // STATUS via READ until done, then READ the result buffer and check it
  // against the golden (matmulInt + requantRef).
  group('LoomUsbCmdEngine end-to-end matmul over USB commands', () {
    test(
      'write inputs, start, poll STATUS.done, read result == golden',
      () async {
        // Accelerator: 2x2 PE, maxRows=maxCols=4 (keeps the test small).
        final accel = LoomAccelerator(
          config: const LoomAcceleratorConfig(
            baseAddress: 0x0,
            peRows: 2,
            peCols: 2,
            maxRows: 4,
            maxCols: 4,
          ),
          name: 'accel',
        );
        final engine = LoomUsbCmdEngine(
          config: const LoomUsbDeviceConfig(
            busAddressWidth: 12,
            busDataWidth: 32,
          ),
          name: 'cmd_engine',
        );
        final top = _CmdAccelTop(engine: engine, accel: accel);

        final clk = SimpleClockGenerator(10).clk;
        top.port('clk').getsLogic(clk);
        await top.build();

        final reset = top.input('reset');
        final cmdData = top.input('cmd_data');
        final cmdValid = top.input('cmd_valid');
        final respReady = top.input('resp_ready');

        Simulator.setMaxSimTime(40000000);
        unawaited(Simulator.run());

        reset.inject(1);
        cmdData.inject(0);
        cmdValid.inject(0);
        respReady.inject(0);
        for (var i = 0; i < 5; i++) {
          await clk.nextPosedge;
        }
        reset.put(0);
        await clk.nextPosedge;

        // A 2x2 problem. maxCols=4, so the accelerator stores W[r,c] at byte
        // r*maxCols + c (the strided weight layout, see accelerator_test).
        const rows = 2;
        const cols = 2;
        const maxCols = 4;
        const shift = 16;
        final w = Int8List.fromList([
          2, -1, // row 0
          3, 4, // row 1
        ]);
        final x = Int8List.fromList([5, 6]);
        final rowMults = <int>[32768, 16384]; // unsigned 16-bit

        final accRef = matmulInt(w, x, rows, cols);
        final golden = List.generate(
          rows,
          (r) => requantRef(accRef[r], rowMults[r], shift),
        );

        int u8(int v) => v & 0xFF;

        // Helper: write one 32-bit word at a CSR/buffer address.
        Future<void> writeWord(int addr, int word) =>
            _feedCmd(top, clk, cmdData, cmdValid, _writeCmd(addr, _le32(word)));

        // ROWS / COLS / SHIFT.
        await writeWord(0x004, rows);
        await writeWord(0x008, cols);
        await writeWord(0x00C, shift);

        // Weight buffer 0x100, strided by maxCols. Build a flat maxCols-strided
        // byte array, pad to words, and write word by word.
        final wFlat = List.filled(maxCols * rows, 0);
        for (var r = 0; r < rows; r++) {
          for (var c = 0; c < cols; c++) {
            wFlat[r * maxCols + c] = w[r * cols + c];
          }
        }
        while (wFlat.length % 4 != 0) {
          wFlat.add(0);
        }
        for (var i = 0; i < wFlat.length; i += 4) {
          final word =
              u8(wFlat[i]) |
              (u8(wFlat[i + 1]) << 8) |
              (u8(wFlat[i + 2]) << 16) |
              (u8(wFlat[i + 3]) << 24);
          await writeWord(0x100 + i, word);
        }

        // Activation buffer 0x200: int8 packed 4/word, x[c].
        final xWord = u8(x[0]) | (u8(x[1]) << 8);
        await writeWord(0x200, xWord);

        // rowMult buffer 0x300: uint16 packed 2/word, mult[r].
        final mWord = (rowMults[0] & 0xFFFF) | ((rowMults[1] & 0xFFFF) << 16);
        await writeWord(0x300, mWord);

        // CONTROL.start (bit 0) at 0x010.
        await writeWord(0x010, 0x1);

        // Poll STATUS (0x014) via READ until done (bit 1) is set.
        var done = false;
        for (var poll = 0; poll < 200 && !done; poll++) {
          await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x014, 4));
          final st = await _collectResp(top, clk, respReady, 4);
          final status = st[0] | (st[1] << 8) | (st[2] << 16) | (st[3] << 24);
          if ((status & 0x2) != 0) done = true;
        }
        expect(done, isTrue, reason: 'STATUS.done set after compute');

        // READ the result buffer (0x400): int8 packed 4/word, y[r]. Read 1 word.
        await _feedCmd(top, clk, cmdData, cmdValid, _readCmd(0x400, 4));
        final res = await _collectResp(top, clk, respReady, 4);
        final word = res[0] | (res[1] << 8) | (res[2] << 16) | (res[3] << 24);
        expect(
          int8FromBusWord(word, 0),
          equals(golden[0]),
          reason: 'y[0] == golden (acc=${accRef[0]})',
        );
        expect(
          int8FromBusWord(word, 1),
          equals(golden[1]),
          reason: 'y[1] == golden (acc=${accRef[1]})',
        );

        await Simulator.endSimulation();
      },
    );
  });

  // Data-toggle handling sanity: the bulk OUT endpoint must ACK a DATA0 then a
  // DATA1, ignore a duplicate (wrong-toggle) packet, and the engine's toggle
  // state advances. We exercise the toggle logic on the bulk OUT FSM directly.
  group('LoomUsbCmdEngine data-toggle on bulk OUT', () {
    test('out_toggle starts at DATA0 and advances when a command packet is '
        'accepted', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final engine = LoomUsbCmdEngine(
        config: const LoomUsbDeviceConfig(
          busAddressWidth: 12,
          busDataWidth: 32,
        ),
        name: 'toggle_engine',
      );
      final top = _CmdAccelTop(engine: engine, accel: accel);
      final clk = SimpleClockGenerator(10).clk;
      top.port('clk').getsLogic(clk);
      await top.build();

      final reset = top.input('reset');
      final cmdData = top.input('cmd_data');
      final cmdValid = top.input('cmd_valid');
      final respReady = top.input('resp_ready');

      Simulator.setMaxSimTime(10000000);
      unawaited(Simulator.run());

      reset.inject(1);
      cmdData.inject(0);
      cmdValid.inject(0);
      respReady.inject(0);
      for (var i = 0; i < 5; i++) {
        await clk.nextPosedge;
      }
      reset.put(0);
      await clk.nextPosedge;

      // After reset, the expected OUT toggle is DATA0 (0).
      expect(
        engine.output('out_toggle').value.toInt(),
        equals(0),
        reason: 'first expected OUT packet is DATA0',
      );

      // Send one full command (a 0-length WRITE: header only). The header is a
      // complete OUT DATA packet, so once dispatched the toggle must flip to 1.
      await _feedCmd(top, clk, cmdData, cmdValid, _writeCmd(0x004, []));
      var g = 0;
      while (top.output('busy').value.toInt() == 1 && g < 1000) {
        g++;
        await clk.nextPosedge;
      }
      expect(
        engine.output('out_toggle').value.toInt(),
        equals(1),
        reason: 'OUT toggle advanced to DATA1 after the first packet',
      );

      await Simulator.endSimulation();
    });
  });

  // SV emission: full LoomUsbDevice names itself + the harbor PHY/packet
  // submodules (proving the real USB transport is wired, not stubbed).
  group('LoomUsbDevice SV emission', () {
    test('emits non-empty SV naming LoomUsbDevice + the harbor PHY/packet '
        'submodules + the command engine', () async {
      final dev = LoomUsbDevice(
        config: const LoomUsbDeviceConfig(
          busAddressWidth: 12,
          busDataWidth: 32,
        ),
        name: 'loom_usb_device',
      );
      await dev.build();
      final sv = dev.generateSynth();
      expect(sv, isNotEmpty);
      expect(sv, contains('LoomUsbDevice'));
      expect(sv, contains('HarborUsbFsPhyRx'));
      expect(sv, contains('HarborUsbFsPhyTx'));
      expect(sv, contains('UsbPacketRx'));
      expect(sv, contains('UsbPacketTx'));
      expect(sv, contains('LoomUsbCmdEngine'));
      // UsbEp0Engine is the EP0+bulk front-end.
      expect(sv, contains('UsbEp0Engine'));
    });
  });

  // Raw dp/dm enumeration sanity: drive a host USB packet layer (the same
  // harbor UsbPacketTx/PhyTx/PhyRx/UsbPacketRx the device uses) into harbor's
  // UsbEp0Engine (loom vendor descriptors + bulk), run GET_DESCRIPTOR(device),
  // and assert the vendor descriptor (class 0xFF, configured VID/PID) reaches
  // the wire, not just the ROM.
  group('UsbEp0Engine (loom vendor) raw dp/dm GET_DESCRIPTOR', () {
    const pidSetup = 0x2D;
    const pidIn = 0x69;
    const pidOut = 0xE1;
    const pidData0 = 0xC3;
    const pidData1 = 0x4B;
    const pidAck = 0xD2;

    test(
      'returns the vendor device descriptor byte-exact over the line',
      () async {
        const vid = 0x1209;
        const pid = 0x10C0;

        final eng = UsbEp0Engine(
          name: 'enum_eng',
          descriptors: LoomUsbDescriptorRom.descriptorEntries(
            idVendor: vid,
            idProduct: pid,
          ),
          bulkEndpoints: true,
        );

        // Host-side packet/PHY layer (reused harbor blocks).
        final htx = UsbPacketTx(name: 'host_ptx');
        final hphyTx = HarborUsbFsPhyTx(name: 'host_phytx');
        final hphyRx = HarborUsbFsPhyRx(name: 'host_phyrx');
        final hrx = UsbPacketRx(name: 'host_prx', bufBytes: 80);

        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        final hSend = Logic(name: 'h_send');
        final hIsData = Logic(name: 'h_is_data');
        final hPid = Logic(name: 'h_pid', width: 8);
        final hPayLen = Logic(name: 'h_payload_len', width: 8);
        final hPayByte = Logic(name: 'h_payload_byte', width: 8);
        final hRdIndex = Logic(name: 'h_rd_index', width: 8);

        htx.input('clk').srcConnection! <= clk;
        htx.input('reset').srcConnection! <= reset;
        htx.input('send').srcConnection! <= hSend;
        htx.input('is_data').srcConnection! <= hIsData;
        htx.input('pid').srcConnection! <= hPid;
        htx.input('payload_len').srcConnection! <= hPayLen;
        htx.input('payload_byte').srcConnection! <= hPayByte;

        hphyTx.input('clk').srcConnection! <= clk;
        hphyTx.input('reset').srcConnection! <= reset;
        hphyTx.input('data').srcConnection! <= htx.output('tx_data');
        hphyTx.input('data_valid').srcConnection! <=
            htx.output('tx_data_valid');
        hphyTx.input('eop_req').srcConnection! <= htx.output('tx_eop_req');
        htx.input('tx_ready').srcConnection! <= hphyTx.output('ready');
        htx.input('tx_oe').srcConnection! <= hphyTx.output('oe');

        // Engine under test: host line -> engine pads.
        eng.input('clk').srcConnection! <= clk;
        eng.input('reset').srcConnection! <= reset;
        eng.input('dp').srcConnection! <= hphyTx.output('dp_out');
        eng.input('dm').srcConnection! <= hphyTx.output('dm_out');
        // The bulk command-engine ports are not exercised here: tie inputs off.
        eng.input('cmd_ready').srcConnection! <= Const(1);
        eng.input('resp_data').srcConnection! <= Const(0, width: 8);
        eng.input('resp_valid').srcConnection! <= Const(0);

        // Engine line -> host PhyRx -> host UsbPacketRx.
        hphyRx.input('clk').srcConnection! <= clk;
        hphyRx.input('reset').srcConnection! <= reset;
        hphyRx.input('dp').srcConnection! <= eng.output('dp_out');
        hphyRx.input('dm').srcConnection! <= eng.output('dm_out');
        hrx.input('clk').srcConnection! <= clk;
        hrx.input('reset').srcConnection! <= reset;
        hrx.input('rx_data').srcConnection! <= hphyRx.output('data');
        hrx.input('rx_valid').srcConnection! <= hphyRx.output('valid');
        hrx.input('rx_sop').srcConnection! <= hphyRx.output('sop');
        hrx.input('rx_eop').srcConnection! <= hphyRx.output('eop');
        hrx.input('rd_index').srcConnection! <= hRdIndex;

        await eng.build();
        await htx.build();
        await hphyTx.build();
        await hphyRx.build();
        await hrx.build();

        reset.inject(1);
        hSend.inject(0);
        hIsData.inject(0);
        hPid.inject(0);
        hPayLen.inject(0);
        hPayByte.inject(0);
        hRdIndex.inject(0);
        Simulator.setMaxSimTime(160000000);
        unawaited(Simulator.run());

        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        List<int> curPayload = const [];
        void serve() {
          if (curPayload.isEmpty) return;
          final i = htx.output('payload_index').value;
          final idx = i.isValid ? i.toInt() : 0;
          hPayByte.inject(idx < curPayload.length ? curPayload[idx] : 0);
        }

        Future<void> hostSend({
          required int pidByte,
          required bool isData,
          List<int> payload = const [],
        }) async {
          curPayload = payload;
          hIsData.inject(isData ? 1 : 0);
          hPid.inject(pidByte);
          hPayLen.inject(payload.length);
          serve();
          hSend.inject(1);
          await clk.nextPosedge;
          hSend.inject(0);
          serve();
          var guard = 0;
          while (htx.output('busy').value.toInt() == 0 && guard < 50) {
            guard++;
            serve();
            await clk.nextPosedge;
          }
          guard = 0;
          while (htx.output('done').value.toInt() == 0 && guard < 4000) {
            guard++;
            serve();
            await clk.nextPosedge;
          }
          for (var i = 0; i < 30; i++) {
            await clk.nextPosedge;
          }
        }

        Future<void> hostToken(int p) => hostSend(pidByte: p, isData: false);

        Future<Map<String, dynamic>> hostExpectData({
          int timeout = 8000,
        }) async {
          var guard = 0;
          while (guard < timeout) {
            guard++;
            await clk.nextPosedge;
            if (hrx.output('pkt_done').value.toInt() == 1) {
              final p = hrx.output('pid').value.toInt();
              final count = hrx.output('byte_count').value.toInt();
              final bytes = <int>[];
              for (var i = 0; i < count; i++) {
                hRdIndex.inject(i);
                await clk.nextPosedge;
                bytes.add(hrx.output('rd_byte').value.toInt());
              }
              hRdIndex.inject(0);
              final payload = count >= 2
                  ? bytes.sublist(0, count - 2)
                  : <int>[];
              return {'pid': p, 'bytes': payload};
            }
          }
          return {'pid': -1, 'bytes': <int>[]};
        }

        // GET_DESCRIPTOR(device, wLength 18): bmRequestType=0x80, bRequest=6,
        // wValue=0x0100 (DEVICE, index 0), wIndex=0, wLength=18.
        final setupDev = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x12, 0x00];
        await hostToken(pidSetup);
        await hostSend(pidByte: pidData0, isData: true, payload: setupDev);
        final ack = await hostExpectData();
        expect(
          ack['pid'],
          equals(pidAck),
          reason: 'device ACKs the GET_DESCRIPTOR(device) SETUP',
        );

        // IN token -> device sends the 18-byte DATA1 device descriptor.
        await hostToken(pidIn);
        final dev = await hostExpectData();
        expect(
          dev['pid'],
          equals(pidData1),
          reason: 'first IN-data packet is DATA1',
        );
        final devBytes = dev['bytes'] as List<int>;

        // Expected = LoomUsbDescriptorRom.deviceDescriptor with VID/PID patched.
        final expected = List<int>.from(LoomUsbDescriptorRom.deviceDescriptor);
        expected[8] = vid & 0xFF;
        expected[9] = (vid >> 8) & 0xFF;
        expected[10] = pid & 0xFF;
        expected[11] = (pid >> 8) & 0xFF;

        expect(
          devBytes.length,
          equals(18),
          reason: 'device descriptor is 18 bytes',
        );
        for (var i = 0; i < 18; i++) {
          expect(
            devBytes[i],
            equals(expected[i]),
            reason: 'vendor device descriptor byte[$i]',
          );
        }
        // Sanity on the vendor-defining fields.
        expect(
          devBytes[4],
          equals(0xFF),
          reason: 'bDeviceClass vendor-specific',
        );
        expect(
          devBytes[8] | (devBytes[9] << 8),
          equals(vid),
          reason: 'idVendor',
        );
        expect(
          devBytes[10] | (devBytes[11] << 8),
          equals(pid),
          reason: 'idProduct',
        );

        // Host ACKs the data, then the OUT status stage (OUT token + ZLP DATA1).
        await hostSend(pidByte: pidAck, isData: false);
        await hostToken(pidOut);
        await hostSend(pidByte: pidData1, isData: true, payload: const []);
        final st = await hostExpectData();
        expect(
          st['pid'],
          equals(pidAck),
          reason: 'device ACKs the OUT status stage',
        );

        await Simulator.endSimulation();
      },
    );

    // Full enumeration sequence over raw dp/dm: GET_DESCRIPTOR, then
    // SET_ADDRESS, then a request AT THE NEW ADDRESS, then SET_CONFIGURATION.
    //
    // Regression coverage for stale capIdx: a single-transfer test can't catch
    // a buffer-walk index left over from a prior transfer, so this exercises
    // GET_DESCRIPTOR (which walks capIdx to 7) immediately followed by
    // SET_ADDRESS and a transfer at the new address. dev_addr must flip only
    // after the status stage.
    test('SET_ADDRESS after GET_DESCRIPTOR: addr applies after status, then a '
        'control transfer at the new address still completes', () async {
      const vid = 0x1209;
      const pid = 0x10C0;
      const newAddr = 7;

      final eng = UsbEp0Engine(
        name: 'enum_seq_eng',
        descriptors: LoomUsbDescriptorRom.descriptorEntries(
          idVendor: vid,
          idProduct: pid,
        ),
        bulkEndpoints: true,
      );

      final htx = UsbPacketTx(name: 'hs_ptx');
      final hphyTx = HarborUsbFsPhyTx(name: 'hs_phytx');
      final hphyRx = HarborUsbFsPhyRx(name: 'hs_phyrx');
      final hrx = UsbPacketRx(name: 'hs_prx', bufBytes: 80);

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final hSend = Logic(name: 'h_send');
      final hIsData = Logic(name: 'h_is_data');
      final hPid = Logic(name: 'h_pid', width: 8);
      final hPayLen = Logic(name: 'h_payload_len', width: 8);
      final hPayByte = Logic(name: 'h_payload_byte', width: 8);
      final hRdIndex = Logic(name: 'h_rd_index', width: 8);

      htx.input('clk').srcConnection! <= clk;
      htx.input('reset').srcConnection! <= reset;
      htx.input('send').srcConnection! <= hSend;
      htx.input('is_data').srcConnection! <= hIsData;
      htx.input('pid').srcConnection! <= hPid;
      htx.input('payload_len').srcConnection! <= hPayLen;
      htx.input('payload_byte').srcConnection! <= hPayByte;

      hphyTx.input('clk').srcConnection! <= clk;
      hphyTx.input('reset').srcConnection! <= reset;
      hphyTx.input('data').srcConnection! <= htx.output('tx_data');
      hphyTx.input('data_valid').srcConnection! <= htx.output('tx_data_valid');
      hphyTx.input('eop_req').srcConnection! <= htx.output('tx_eop_req');
      htx.input('tx_ready').srcConnection! <= hphyTx.output('ready');
      htx.input('tx_oe').srcConnection! <= hphyTx.output('oe');

      eng.input('clk').srcConnection! <= clk;
      eng.input('reset').srcConnection! <= reset;
      eng.input('dp').srcConnection! <= hphyTx.output('dp_out');
      eng.input('dm').srcConnection! <= hphyTx.output('dm_out');
      eng.input('cmd_ready').srcConnection! <= Const(1);
      eng.input('resp_data').srcConnection! <= Const(0, width: 8);
      eng.input('resp_valid').srcConnection! <= Const(0);

      hphyRx.input('clk').srcConnection! <= clk;
      hphyRx.input('reset').srcConnection! <= reset;
      hphyRx.input('dp').srcConnection! <= eng.output('dp_out');
      hphyRx.input('dm').srcConnection! <= eng.output('dm_out');
      hrx.input('clk').srcConnection! <= clk;
      hrx.input('reset').srcConnection! <= reset;
      hrx.input('rx_data').srcConnection! <= hphyRx.output('data');
      hrx.input('rx_valid').srcConnection! <= hphyRx.output('valid');
      hrx.input('rx_sop').srcConnection! <= hphyRx.output('sop');
      hrx.input('rx_eop').srcConnection! <= hphyRx.output('eop');
      hrx.input('rd_index').srcConnection! <= hRdIndex;

      await eng.build();
      await htx.build();
      await hphyTx.build();
      await hphyRx.build();
      await hrx.build();

      reset.inject(1);
      hSend.inject(0);
      hIsData.inject(0);
      hPid.inject(0);
      hPayLen.inject(0);
      hPayByte.inject(0);
      hRdIndex.inject(0);
      Simulator.setMaxSimTime(400000000);
      unawaited(Simulator.run());

      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }

      List<int> curPayload = const [];
      void serve() {
        if (curPayload.isEmpty) return;
        final i = htx.output('payload_index').value;
        final idx = i.isValid ? i.toInt() : 0;
        hPayByte.inject(idx < curPayload.length ? curPayload[idx] : 0);
      }

      Future<void> hostSend({
        required int pidByte,
        required bool isData,
        List<int> payload = const [],
      }) async {
        curPayload = payload;
        hIsData.inject(isData ? 1 : 0);
        hPid.inject(pidByte);
        hPayLen.inject(payload.length);
        serve();
        hSend.inject(1);
        await clk.nextPosedge;
        hSend.inject(0);
        serve();
        var guard = 0;
        while (htx.output('busy').value.toInt() == 0 && guard < 50) {
          guard++;
          serve();
          await clk.nextPosedge;
        }
        guard = 0;
        while (htx.output('done').value.toInt() == 0 && guard < 4000) {
          guard++;
          serve();
          await clk.nextPosedge;
        }
        for (var i = 0; i < 30; i++) {
          await clk.nextPosedge;
        }
      }

      Future<void> hostToken(int p) => hostSend(pidByte: p, isData: false);

      Future<Map<String, dynamic>> hostExpectData({int timeout = 8000}) async {
        var guard = 0;
        while (guard < timeout) {
          guard++;
          await clk.nextPosedge;
          if (hrx.output('pkt_done').value.toInt() == 1) {
            final p = hrx.output('pid').value.toInt();
            final count = hrx.output('byte_count').value.toInt();
            final bytes = <int>[];
            for (var i = 0; i < count; i++) {
              hRdIndex.inject(i);
              await clk.nextPosedge;
              bytes.add(hrx.output('rd_byte').value.toInt());
            }
            hRdIndex.inject(0);
            final payload = count >= 2 ? bytes.sublist(0, count - 2) : <int>[];
            return {'pid': p, 'bytes': payload};
          }
        }
        return {'pid': -1, 'bytes': <int>[]};
      }

      int devAddrNow() => eng.output('dev_addr').value.toInt();
      int configuredNow() => eng.output('configured').value.toInt();

      // A full GET_DESCRIPTOR(device) control transfer (SETUP + IN data +
      // OUT status). Returns the descriptor payload bytes.
      Future<List<int>> getDeviceDescriptor() async {
        final setupDev = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x12, 0x00];
        await hostToken(pidSetup);
        await hostSend(pidByte: pidData0, isData: true, payload: setupDev);
        final ack = await hostExpectData();
        expect(ack['pid'], equals(pidAck), reason: 'device ACKs the SETUP');
        await hostToken(pidIn);
        final dev = await hostExpectData();
        expect(dev['pid'], equals(pidData1), reason: 'first IN-data is DATA1');
        // Host ACKs the data, then the OUT status stage (OUT + ZLP DATA1).
        await hostSend(pidByte: pidAck, isData: false);
        await hostToken(pidOut);
        await hostSend(pidByte: pidData1, isData: true, payload: const []);
        final st = await hostExpectData();
        expect(
          st['pid'],
          equals(pidAck),
          reason: 'device ACKs the OUT status stage',
        );
        return dev['bytes'] as List<int>;
      }

      final expected = List<int>.from(LoomUsbDescriptorRom.deviceDescriptor);
      expected[8] = vid & 0xFF;
      expected[9] = (vid >> 8) & 0xFF;
      expected[10] = pid & 0xFF;
      expected[11] = (pid >> 8) & 0xFF;

      final dev1 = await getDeviceDescriptor();
      expect(dev1.length, equals(18), reason: 'transfer 1 descriptor is 18 B');
      for (var i = 0; i < 18; i++) {
        expect(
          dev1[i],
          equals(expected[i]),
          reason: 'transfer 1 device descriptor byte[$i]',
        );
      }
      expect(
        devAddrNow(),
        equals(0),
        reason: 'device still at address 0 before SET_ADDRESS',
      );

      final setupAddr = [0x00, 0x05, newAddr, 0x00, 0x00, 0x00, 0x00, 0x00];
      await hostToken(pidSetup);
      await hostSend(pidByte: pidData0, isData: true, payload: setupAddr);
      final addrSetupAck = await hostExpectData();
      expect(
        addrSetupAck['pid'],
        equals(pidAck),
        reason: 'device ACKs the SET_ADDRESS SETUP',
      );
      // SPEC: the address must NOT change at the SETUP stage. It stays 0 until
      // the status stage completes.
      expect(
        devAddrNow(),
        equals(0),
        reason: 'address must NOT change at the SET_ADDRESS SETUP stage',
      );

      // IN status: device sends a zero-length DATA1. Host ACKs it.
      await hostToken(pidIn);
      final addrStatus = await hostExpectData();
      expect(
        addrStatus['pid'],
        equals(pidData1),
        reason: 'SET_ADDRESS IN status is a zero-length DATA1',
      );
      expect(
        (addrStatus['bytes'] as List<int>).isEmpty,
        isTrue,
        reason: 'SET_ADDRESS status data is zero-length',
      );
      // SPEC: the device still owns address 0 for the status stage. It applies
      // the new address only AFTER it sees the host ACK of the status ZLP.
      expect(
        devAddrNow(),
        equals(0),
        reason: 'address still 0 until the host ACKs the status ZLP',
      );
      await hostSend(pidByte: pidAck, isData: false);
      // Now the address must have switched.
      for (var i = 0; i < 10; i++) {
        await clk.nextPosedge;
      }
      expect(
        devAddrNow(),
        equals(newAddr),
        reason: 'device applies the new address AFTER the status stage ACK',
      );

      // A third transfer, exercising the capIdx-walked state from transfer 1.
      final dev3 = await getDeviceDescriptor();
      expect(
        dev3.length,
        equals(18),
        reason: 'transfer 3 (post-SET_ADDRESS) descriptor is 18 B',
      );
      for (var i = 0; i < 18; i++) {
        expect(
          dev3[i],
          equals(expected[i]),
          reason: 'post-SET_ADDRESS device descriptor byte[$i]',
        );
      }

      expect(configuredNow(), equals(0), reason: 'not configured yet');
      final setupCfg = [0x00, 0x09, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
      await hostToken(pidSetup);
      await hostSend(pidByte: pidData0, isData: true, payload: setupCfg);
      final cfgSetupAck = await hostExpectData();
      expect(
        cfgSetupAck['pid'],
        equals(pidAck),
        reason: 'device ACKs the SET_CONFIGURATION SETUP',
      );
      await hostToken(pidIn);
      final cfgStatus = await hostExpectData();
      expect(
        cfgStatus['pid'],
        equals(pidData1),
        reason: 'SET_CONFIGURATION IN status is a zero-length DATA1',
      );
      await hostSend(pidByte: pidAck, isData: false);
      for (var i = 0; i < 10; i++) {
        await clk.nextPosedge;
      }
      expect(
        configuredNow(),
        equals(1),
        reason: 'device is configured after SET_CONFIGURATION status ACK',
      );
      // The address must be unchanged by the configuration transfer.
      expect(
        devAddrNow(),
        equals(newAddr),
        reason: 'address unchanged across SET_CONFIGURATION',
      );

      await Simulator.endSimulation();
    });

    // The IDLE handler decodes the token endpoint from tokEndp0 = rxByte[7]
    // (payload byte 0 bit 7), read at pktRx.buf[capIdx]. capIdx must be held
    // at 0 in IDLE so a GET_DESCRIPTOR beforehand (which walks capIdx to 7)
    // can't leave a stale index for the next token: deliver a real EP1 IN
    // token (payload byte 0 bit 7 set, byte 7 == 0) and the device must read
    // byte 0 and NAK it, not silently ignore it by reading the stale byte 7.
    //
    // A token is delivered as a PID + 2 payload bytes packet (byte 0 =
    // addr|endp<<7) so pktRx.buf[0] carries the real token endpoint byte.
    test(
      'EP1 IN token in IDLE after a GET_DESCRIPTOR is recognized (NAKed), not '
      'lost to a stale buffer-walk index',
      () async {
        const pidNak = 0x5A;

        final eng = UsbEp0Engine(
          name: 'ep1_idle_eng',
          descriptors: LoomUsbDescriptorRom.descriptorEntries(),
          bulkEndpoints: true,
        );

        final htx = UsbPacketTx(name: 'e1_ptx');
        final hphyTx = HarborUsbFsPhyTx(name: 'e1_phytx');
        final hphyRx = HarborUsbFsPhyRx(name: 'e1_phyrx');
        final hrx = UsbPacketRx(name: 'e1_prx', bufBytes: 80);

        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        final hSend = Logic(name: 'h_send');
        final hIsData = Logic(name: 'h_is_data');
        final hPid = Logic(name: 'h_pid', width: 8);
        final hPayLen = Logic(name: 'h_payload_len', width: 8);
        final hPayByte = Logic(name: 'h_payload_byte', width: 8);
        final hRdIndex = Logic(name: 'h_rd_index', width: 8);

        htx.input('clk').srcConnection! <= clk;
        htx.input('reset').srcConnection! <= reset;
        htx.input('send').srcConnection! <= hSend;
        htx.input('is_data').srcConnection! <= hIsData;
        htx.input('pid').srcConnection! <= hPid;
        htx.input('payload_len').srcConnection! <= hPayLen;
        htx.input('payload_byte').srcConnection! <= hPayByte;

        hphyTx.input('clk').srcConnection! <= clk;
        hphyTx.input('reset').srcConnection! <= reset;
        hphyTx.input('data').srcConnection! <= htx.output('tx_data');
        hphyTx.input('data_valid').srcConnection! <=
            htx.output('tx_data_valid');
        hphyTx.input('eop_req').srcConnection! <= htx.output('tx_eop_req');
        htx.input('tx_ready').srcConnection! <= hphyTx.output('ready');
        htx.input('tx_oe').srcConnection! <= hphyTx.output('oe');

        eng.input('clk').srcConnection! <= clk;
        eng.input('reset').srcConnection! <= reset;
        eng.input('dp').srcConnection! <= hphyTx.output('dp_out');
        eng.input('dm').srcConnection! <= hphyTx.output('dm_out');
        // The command engine offers NO response byte, so a recognized EP1 IN
        // must be NAKed.
        eng.input('cmd_ready').srcConnection! <= Const(1);
        eng.input('resp_data').srcConnection! <= Const(0, width: 8);
        eng.input('resp_valid').srcConnection! <= Const(0);

        hphyRx.input('clk').srcConnection! <= clk;
        hphyRx.input('reset').srcConnection! <= reset;
        hphyRx.input('dp').srcConnection! <= eng.output('dp_out');
        hphyRx.input('dm').srcConnection! <= eng.output('dm_out');
        hrx.input('clk').srcConnection! <= clk;
        hrx.input('reset').srcConnection! <= reset;
        hrx.input('rx_data').srcConnection! <= hphyRx.output('data');
        hrx.input('rx_valid').srcConnection! <= hphyRx.output('valid');
        hrx.input('rx_sop').srcConnection! <= hphyRx.output('sop');
        hrx.input('rx_eop').srcConnection! <= hphyRx.output('eop');
        hrx.input('rd_index').srcConnection! <= hRdIndex;

        await eng.build();
        await htx.build();
        await hphyTx.build();
        await hphyRx.build();
        await hrx.build();

        reset.inject(1);
        hSend.inject(0);
        hIsData.inject(0);
        hPid.inject(0);
        hPayLen.inject(0);
        hPayByte.inject(0);
        hRdIndex.inject(0);
        Simulator.setMaxSimTime(200000000);
        unawaited(Simulator.run());

        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        List<int> curPayload = const [];
        void serve() {
          if (curPayload.isEmpty) return;
          final i = htx.output('payload_index').value;
          final idx = i.isValid ? i.toInt() : 0;
          hPayByte.inject(idx < curPayload.length ? curPayload[idx] : 0);
        }

        Future<void> hostSend({
          required int pidByte,
          required bool isData,
          List<int> payload = const [],
        }) async {
          curPayload = payload;
          hIsData.inject(isData ? 1 : 0);
          hPid.inject(pidByte);
          hPayLen.inject(payload.length);
          serve();
          hSend.inject(1);
          await clk.nextPosedge;
          hSend.inject(0);
          serve();
          var guard = 0;
          while (htx.output('busy').value.toInt() == 0 && guard < 50) {
            guard++;
            serve();
            await clk.nextPosedge;
          }
          guard = 0;
          while (htx.output('done').value.toInt() == 0 && guard < 4000) {
            guard++;
            serve();
            await clk.nextPosedge;
          }
          for (var i = 0; i < 30; i++) {
            await clk.nextPosedge;
          }
        }

        Future<void> hostToken(int p) => hostSend(pidByte: p, isData: false);

        Future<int> hostExpectPid({int timeout = 8000}) async {
          var guard = 0;
          while (guard < timeout) {
            guard++;
            await clk.nextPosedge;
            if (hrx.output('pkt_done').value.toInt() == 1) {
              return hrx.output('pid').value.toInt();
            }
          }
          return -1;
        }

        // Run a GET_DESCRIPTOR(device) control transfer first; this walks
        // capIdx to 7.
        final setupDev = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x12, 0x00];
        await hostToken(pidSetup);
        await hostSend(pidByte: pidData0, isData: true, payload: setupDev);
        expect(
          await hostExpectPid(),
          equals(pidAck),
          reason: 'device ACKs the SETUP',
        );
        await hostToken(pidIn);
        expect(
          await hostExpectPid(),
          equals(pidData1),
          reason: 'device returns the descriptor DATA1',
        );
        await hostSend(pidByte: pidAck, isData: false);
        await hostToken(pidOut);
        await hostSend(pidByte: pidData1, isData: true, payload: const []);
        expect(
          await hostExpectPid(),
          equals(pidAck),
          reason:
              'device ACKs the OUT status (now back in IDLE; capIdx==7 in '
              'the buggy engine)',
        );

        // Deliver a real EP1 IN token: PID = IN, payload byte 0 = endp 1 (bit 7
        // set). The device must recognize EP1 and, having no response byte, NAK.
        final ep1InToken = [0x80, 0x00]; // byte0 = addr0|endp[0]=1<<7
        await hostSend(pidByte: pidIn, isData: true, payload: ep1InToken);
        expect(
          await hostExpectPid(timeout: 8000),
          equals(pidNak),
          reason:
              'device NAKs a recognized EP1 IN with no response byte; on '
              'the stale-capIdx bug it reads buf[7]==0, never sees the EP1 '
              'token, and stays silent (no NAK) - the -71 symptom',
        );

        await Simulator.endSimulation();
      },
    );
  });
}
