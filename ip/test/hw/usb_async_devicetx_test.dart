// ASYNC device-TX decode: prove HarborUsbFsPhyTx (the device's transmit PHY)
// produces a line that a real asynchronous host decodes correctly at every
// bit-clock lock phase, not just the sim's own clock alignment.
//
// Drives HarborUsbFsPhyTx with a known framed packet (SYNC + PID + payload +
// CRC16), captures its dp_out/dm_out line at one fixed off-edge sample per
// device cycle (4 samples/FS bit) via Simulator.registerAction, then decodes
// that waveform with a true 4x DLL at all four lock phases. Every phase must
// recover the identical, CRC-valid packet.

import 'dart:async';

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

const _pidData1 = 0x4B;

int crc16(List<int> bytes) {
  var crc = 0xFFFF;
  for (final b in bytes) {
    for (var i = 0; i < 8; i++) {
      final bit = (b >> i) & 1;
      final xorIn = (crc & 1) ^ bit;
      crc >>= 1;
      if (xorIn != 0) crc ^= 0xA001;
    }
  }
  return (~crc) & 0xFFFF;
}

// Decode 4-sample/bit device-cycle line samples with a 4x DLL at [lockPhase].
Map<String, dynamic> decodeDll(List<int> samples, int lockPhase) {
  if (samples.length < 8) return {'pid': -1, 'bytes': <int>[]};
  final idle0 = samples[0];
  var sop = 0;
  while (sop < samples.length && samples[sop] == idle0) {
    sop++;
  }
  if (sop >= samples.length) return {'pid': -1, 'bytes': <int>[]};
  final win = samples.sublist(sop);
  final bitSyms = <int>[];
  var phase = 0;
  var last = idle0;
  for (var i = 0; i < win.length; i++) {
    final cur = win[i];
    if (cur != last) phase = 0;
    if (phase == lockPhase) bitSyms.add(cur);
    phase = (phase + 1) & 3;
    last = cur;
  }
  final bits = <int>[];
  var prev = 2;
  var ones = 0;
  for (final s in bitSyms) {
    if (s == 0 || s == 3) break;
    final decoded = s == prev ? 1 : 0;
    prev = s;
    if (ones == 6) {
      ones = 0;
      continue;
    }
    bits.add(decoded);
    if (decoded == 1) {
      ones++;
    } else {
      ones = 0;
    }
  }
  if (bits.length < 16) return {'pid': -1, 'bytes': <int>[]};
  final syncOk = List.generate(
    8,
    (k) => bits[k] == (k == 7 ? 1 : 0),
  ).every((e) => e);
  if (!syncOk) return {'pid': -2, 'bytes': <int>[]};
  final body = bits.sublist(8);
  final bytes = <int>[];
  for (var i = 0; i + 8 <= body.length; i += 8) {
    var b = 0;
    for (var j = 0; j < 8; j++) {
      b |= body[i + j] << j;
    }
    bytes.add(b);
  }
  if (bytes.isEmpty) return {'pid': -3, 'bytes': <int>[]};
  final pid = bytes.first;
  final payload = bytes.length >= 3
      ? bytes.sublist(1, bytes.length - 2)
      : <int>[];
  final rxCrc = bytes.length >= 3
      ? bytes[bytes.length - 2] | (bytes[bytes.length - 1] << 8)
      : -1;
  if (rxCrc != crc16(payload))
    return {'pid': -4, 'bytes': payload, 'crc': rxCrc};
  return {'pid': pid, 'bytes': payload};
}

void main() {
  tearDown(() async => Simulator.reset());

  test(
    'HarborUsbFsPhyTx line decodes cleanly at every async DLL lock phase',
    () async {
      const devPeriod = 8; // 4 samples/bit, sample at +2 (quarter cycle)
      final phyTx = HarborUsbFsPhyTx(name: 'devtx');
      final clk = SimpleClockGenerator(devPeriod).clk;
      final reset = Logic(name: 'reset');
      final txData = Logic(name: 'tx_data');
      final txValid = Logic(name: 'tx_valid');
      final txEop = Logic(name: 'tx_eop');
      phyTx.input('clk').srcConnection! <= clk;
      phyTx.input('reset').srcConnection! <= reset;
      phyTx.input('data').srcConnection! <= txData;
      phyTx.input('data_valid').srcConnection! <= txValid;
      phyTx.input('eop_req').srcConnection! <= txEop;
      await phyTx.build();

      txData.inject(0);
      txValid.inject(0);
      txEop.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(1 << 28);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }

      // The DATA1 device descriptor packet, exactly as the device EP0 engine emits
      // it: SYNC(0x80) + PID(0x4B) + 18 descriptor bytes + CRC16. The host frames
      // SYNC+PID+body and the PhyTx owns NRZI/bit-stuff/EOP at the wire level.
      final desc = [
        18, 0x01, 0x00, 0x02, 0xFF, 0x00, 0x00, 64, //
        0x09, 0x12, 0xC0, 0x10, 0x00, 0x01, 1, 2, 0, 1,
      ];
      final c = crc16(desc);
      final bodyBytes = [0x80, _pidData1, ...desc, c & 0xFF, (c >> 8) & 0xFF];
      final bits = <int>[];
      for (final b in bodyBytes) {
        for (var i = 0; i < 8; i++) {
          bits.add((b >> i) & 1);
        }
      }

      // Fine-sample the device TX line at +2 each device cycle via registerAction.
      final samples = <int>[];
      final off = devPeriod ~/ 4;
      var sawDrive = false;
      var idleTail = 0;
      var done = false;
      void schedule(int t) {
        Simulator.registerAction(t, () {
          final oe = phyTx.output('oe').value.toInt();
          final dpv = phyTx.output('dp_out').value.toInt();
          final dmv = phyTx.output('dm_out').value.toInt();
          if (oe == 1) {
            samples.add((dpv << 1) | dmv);
            sawDrive = true;
            idleTail = 0;
          } else {
            samples.add(2);
            if (sawDrive) {
              idleTail++;
              if (idleTail >= 8) done = true;
            }
          }
          if (!done && samples.length < 6000) schedule(t + devPeriod);
        });
      }

      schedule(Simulator.time + off);

      // Drive the PhyTx handshake: present a bit, advance the pointer only on a
      // real accept edge (ready & oe), exactly as harbor's UsbPacketTx does.
      var idx = 0;
      txEop.inject(0);
      txData.inject(bits[0]);
      txValid.inject(1);
      var g = 0;
      while (idx < bits.length && g < 40000) {
        g++;
        final accept =
            phyTx.output('ready').value.toInt() == 1 &&
            phyTx.output('oe').value.toInt() == 1;
        if (accept) {
          idx++;
          if (idx < bits.length) {
            txData.inject(bits[idx]);
            txValid.inject(1);
          } else {
            txValid.inject(0);
            txEop.inject(1);
          }
        }
        await clk.nextPosedge;
      }
      txValid.inject(0);
      txEop.inject(1);
      g = 0;
      while (g++ < 4000) {
        await clk.nextPosedge;
        if (phyTx.output('oe').value.toInt() == 0 &&
            phyTx.output('busy').value.toInt() == 0) {
          break;
        }
      }
      txEop.inject(0);
      // Let the sampler drain its idle tail.
      g = 0;
      while (!done && g++ < 200) {
        await clk.nextPosedge;
      }

      final results = <int, Map<String, dynamic>>{};
      for (var lock = 0; lock < 4; lock++) {
        results[lock] = decodeDll(samples, lock);
      }

      for (var lock = 0; lock < 4; lock++) {
        final r = results[lock]!;
        expect(
          r['pid'],
          equals(_pidData1),
          reason:
              'lock=$lock: device TX must decode to DATA1 at this async '
              'lock phase (got pid ${r['pid']}; -4=CRC16 fail, -2=SYNC '
              'mis-decode). A garble here at any phase is the hardware -32 cause',
        );
        expect(
          r['bytes'],
          equals(desc),
          reason:
              'lock=$lock: descriptor bytes match exactly across the async '
              'boundary',
        );
      }

      await Simulator.endSimulation();
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
