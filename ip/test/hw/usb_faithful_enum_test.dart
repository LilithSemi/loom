// FAITHFUL raw-dp/dm host model for the consolidated USB front-end (harbor's
// UsbEp0Engine + loom vendor descriptors + bulk).
//
// Unlike usb_harbor_engine_test.dart, this host serializes real USB token
// packets (SYNC + PID + addr[6:0] + endp[3:0] + CRC5 + EOP, driven directly
// on the host PhyTx bit stream with a software CRC5), drives a real SE0 bus
// reset before enumerating, and addresses the device by its assigned address
// after SET_ADDRESS. DATA/handshake packets still reuse harbor's proven
// UsbPacketTx (SYNC+PID+payload+CRC16+EOP), the same framing the device's own
// TX produces, so only the token layer and the bus reset are new here.
//
// Spec grounding (USB 2.0, per reference-driven-rohd):
//   - 8.3.5 token packet: PID then 7-bit ADDR, 4-bit ENDP, 5-bit CRC5. The
//     11-bit field is fed to CRC5 LSB-first. ADDR is bits [6:0], ENDP [10:7].
//   - 8.3.5.1 CRC5: generator x^5 + x^2 + 1 (0x05), init all-ones (0x1F),
//     the residual is one's-complemented before transmission, sent MSB-first
//     of the residual into the high bits of the last byte. On the wire the
//     11-bit field plus the 5-bit CRC pack LSB-first into 2 bytes after the PID:
//       byte0 = ADDR[6:0] | (ENDP[0] << 7)
//       byte1 = ENDP[3:1] | (CRC5[4:0] << 3)
//   - 7.1.7.5 bus reset: SE0 (both lines low) for >= 2.5 us; the PHY's
//     HarborUsbLineRx declares a bus reset after resetTicks (120) SE0 cycles.
//   - CRC16 for data packets: reflected poly 0xA001, init 0xFFFF, residual
//     inverted (matches harbor UsbPacketTx exactly).

import 'dart:async';

import 'package:harbor/harbor.dart';
import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/hw/usb_device.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

// PID bytes (USB 2.0).
const _pidSetup = 0x2D;
const _pidIn = 0x69;
const _pidOut = 0xE1;
const _pidData0 = 0xC3;
const _pidData1 = 0x4B;
const _pidAck = 0xD2;

// CRC5 over the 11-bit token field (USB 2.0 8.3.5.1).
//   poly x^5+x^2+1 (0x05), init 0x1F, residual one's-complemented.
// The 11-bit value is the {endp[3:0], addr[6:0]} field fed LSB-first.
int crc5Token(int addr, int endp) {
  final field = (addr & 0x7F) | ((endp & 0xF) << 7); // 11-bit, addr in LSBs
  var crc = 0x1F;
  for (var i = 0; i < 11; i++) {
    final bit = (field >> i) & 1;
    final xorIn = bit ^ (crc & 1);
    crc >>= 1;
    if (xorIn != 0) {
      crc ^= 0x14; // reflected x^5+x^2+1 over 5 bits
    }
  }
  return (~crc) & 0x1F;
}

// Independent re-derivation of an expected STRING descriptor from the field
// table, so a transcription error in the module is caught (not imported).
List<int> _expectedString(String s) {
  final out = <int>[2 + 2 * s.length, 0x03];
  for (final u in s.codeUnits) {
    out.add(u & 0xFF);
    out.add((u >> 8) & 0xFF);
  }
  return out;
}

/// Top: harbor UsbEp0Engine (loom vendor descriptors + bulk) + the real
/// LoomUsbCmdEngine + LoomAccelerator, single clock domain.
class _DeviceTop extends BridgeModule {
  _DeviceTop({String? name}) : super('_DeviceTop', name: name ?? 'dev_top') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('dp', PortDirection.input);
    createPort('dm', PortDirection.input);
    addOutput('dp_out');
    addOutput('dm_out');
    addOutput('oe');
    addOutput('dev_addr', width: 7);
    addOutput('configured');

    final clk = input('clk');
    final reset = input('reset');

    final ep0 = UsbEp0Engine(
      name: 'dev_ep0',
      descriptors: LoomUsbDescriptorRom.descriptorEntries(),
      bulkEndpoints: true,
    );
    addSubModule(ep0);
    ep0.input('clk').srcConnection! <= clk;
    ep0.input('reset').srcConnection! <= reset;
    ep0.input('dp').srcConnection! <= input('dp');
    ep0.input('dm').srcConnection! <= input('dm');
    output('dp_out') <= ep0.output('dp_out');
    output('dm_out') <= ep0.output('dm_out');
    output('oe') <= ep0.output('oe');
    output('dev_addr') <= ep0.output('dev_addr');
    output('configured') <= ep0.output('configured');

    final engine = LoomUsbCmdEngine(
      config: const LoomUsbDeviceConfig(busAddressWidth: 12, busDataWidth: 32),
      name: 'dev_cmd_engine',
    );
    addSubModule(engine);
    engine.input('clk').srcConnection! <= clk;
    engine.input('reset').srcConnection! <= reset;
    engine.input('cmd_data').srcConnection! <= ep0.output('cmd_data');
    engine.input('cmd_valid').srcConnection! <= ep0.output('cmd_valid');
    engine.input('resp_ready').srcConnection! <= ep0.output('resp_ready');
    ep0.input('cmd_ready').srcConnection! <= engine.output('cmd_ready');
    ep0.input('resp_data').srcConnection! <= engine.output('resp_data');
    ep0.input('resp_valid').srcConnection! <= engine.output('resp_valid');
    ep0.input('resp_last').srcConnection! <= engine.output('resp_last');

    final accel = LoomAccelerator(
      config: const LoomAcceleratorConfig(baseAddress: 0x0),
      name: 'dev_accel',
    );
    addSubModule(accel);
    accel.input('clk').srcConnection! <= clk;
    accel.input('reset').srcConnection! <= reset;
    connectInterfaces(engine.interface('bus'), accel.interface('bus'));
  }
}

/// A FAITHFUL host-side driver.
///
/// DATA/handshake packets reuse harbor's UsbPacketTx (SYNC+PID+payload+CRC16).
/// TOKEN packets are serialized as REAL token packets (SYNC+PID+11-bit
/// addr/endp+CRC5+EOP) by driving the host PhyTx bit stream directly. A BUS
/// RESET drives SE0 onto the device pads for longer than the PHY reset timer.
class _Host {
  final Logic clk;
  final Logic reset;
  // UsbPacketTx control (for DATA/handshake packets).
  final Logic hSend;
  final Logic hIsData;
  final Logic hPid;
  final Logic hPayLen;
  final Logic hPayByte;
  final Logic hRdIndex;
  final UsbPacketTx htx;
  final UsbPacketRx hrx;
  // Direct PhyTx bit-stream control (for TOKEN packets) and the line-source
  // mux select so the host can either let its PHY drive the line or force SE0.
  final Logic hTxData; // raw bit into PhyTx
  final Logic hTxDataValid;
  final Logic hTxEopReq;
  final Logic hForceSe0; // when 1, the device pads are forced to SE0
  final HarborUsbFsPhyTx hphyTx;
  final Logic devOe; // the device's transmit output-enable
  List<int> _curPayload = const [];

  _Host._(
    this.clk,
    this.reset,
    this.hSend,
    this.hIsData,
    this.hPid,
    this.hPayLen,
    this.hPayByte,
    this.hRdIndex,
    this.htx,
    this.hrx,
    this.hTxData,
    this.hTxDataValid,
    this.hTxEopReq,
    this.hForceSe0,
    this.hphyTx,
    this.devOe,
  );

  // Bus turnaround: a real host never starts driving the line until the device
  // has released it (its tristate drivers off, oe low). The device PHY squelches
  // its receiver while it drives (real-PHY TX/RX isolation), so a host that began
  // a token while the device's oe were still asserted would have its SYNC
  // blanked. Wait for the device to release the bus, then a short idle gap.
  Future<void> _awaitBusIdle() async {
    // Debounce oe low for several cycles, since it can glitch low mid-EOP.
    var guard = 0;
    var lowRun = 0;
    while (lowRun < 16 && guard < 8000) {
      guard++;
      lowRun = devOe.value.toInt() == 0 ? lowRun + 1 : 0;
      await clk.nextPosedge;
    }
  }

  void _serve() {
    if (_curPayload.isEmpty) return;
    final i = htx.output('payload_index').value;
    final idx = i.isValid ? i.toInt() : 0;
    hPayByte.inject(idx < _curPayload.length ? _curPayload[idx] : 0);
  }

  // Send a DATA or handshake packet via the proven UsbPacketTx framing.
  Future<void> send({
    required int pid,
    required bool isData,
    List<int> payload = const [],
  }) async {
    await _awaitBusIdle();
    _curPayload = payload;
    hIsData.inject(isData ? 1 : 0);
    hPid.inject(pid);
    hPayLen.inject(payload.length);
    _serve();
    hSend.inject(1);
    await clk.nextPosedge;
    hSend.inject(0);
    _serve();
    var guard = 0;
    while (htx.output('busy').value.toInt() == 0 && guard < 50) {
      guard++;
      _serve();
      await clk.nextPosedge;
    }
    guard = 0;
    while (htx.output('done').value.toInt() == 0 && guard < 4000) {
      guard++;
      _serve();
      await clk.nextPosedge;
    }
    for (var i = 0; i < 30; i++) {
      await clk.nextPosedge;
    }
  }

  // Send a REAL token packet: SYNC + PID + addr[6:0] + endp[3:0] + CRC5 + EOP.
  // The bit stream presented to the host PhyTx is, LSB-first per byte:
  //   SYNC byte 0x80, PID byte, then byte0 = addr|endp0<<7, byte1 =
  //   endp[3:1]|crc5<<3. PhyTx owns NRZI/bit-stuff/EOP at the wire level.
  Future<void> token(int pid, {int addr = 0, int endp = 0}) async {
    await _awaitBusIdle();
    final crc = crc5Token(addr, endp);
    final byte0 = (addr & 0x7F) | ((endp & 1) << 7);
    final byte1 = ((endp >> 1) & 0x7) | ((crc & 0x1F) << 3);
    final bytes = <int>[0x80, pid & 0xFF, byte0, byte1];
    // Flatten to a LSB-first bit stream.
    final bits = <int>[];
    for (final b in bytes) {
      for (var i = 0; i < 8; i++) {
        bits.add((b >> i) & 1);
      }
    }
    // Drive the PhyTx handshake exactly like harbor's UsbPacketTx: present a
    // bit and only advance the pointer on a real accept edge (ready & oe).
    // PhyTx pulses ready once per bit time (phase==3), so advancing on any
    // other edge would double-count a bit and misalign the packet.
    var bitIdx = 0;
    hTxEopReq.inject(0);
    hTxData.inject(bits[0]);
    hTxDataValid.inject(1);
    var guard = 0;
    while (bitIdx < bits.length && guard < 20000) {
      guard++;
      final accept =
          (hphyTx.output('ready').value.toInt() == 1) &&
          (hphyTx.output('oe').value.toInt() == 1);
      if (accept) {
        bitIdx++;
        if (bitIdx < bits.length) {
          hTxData.inject(bits[bitIdx]);
          hTxDataValid.inject(1);
        } else {
          hTxDataValid.inject(0);
          hTxEopReq.inject(1);
        }
      }
      await clk.nextPosedge;
    }
    // All data bits consumed. Hold eop_req until the PhyTx runs the EOP and
    // returns to idle (oe drops).
    hTxDataValid.inject(0);
    hTxEopReq.inject(1);
    guard = 0;
    // Wait for the EOP to start (oe may still be high) then complete (oe low).
    while (guard < 4000) {
      guard++;
      await clk.nextPosedge;
      if (hphyTx.output('oe').value.toInt() == 0 &&
          hphyTx.output('busy').value.toInt() == 0) {
        break;
      }
    }
    hTxEopReq.inject(0);
    // Settle gap so the device framer returns to idle before the next packet.
    for (var i = 0; i < 30; i++) {
      await clk.nextPosedge;
    }
  }

  // A bare handshake (ACK/NAK): PID only, via UsbPacketTx.
  Future<void> handshake(int pid) => send(pid: pid, isData: false);

  // Drive a REAL bus reset: hold SE0 on the device pads for `cycles` clocks
  // (must exceed the PHY resetTicks=120). This forces both pads low regardless
  // of the host PHY, exactly like a host pulling the bus to SE0.
  Future<void> busReset({int cycles = 200}) async {
    await _awaitBusIdle();
    hTxDataValid.inject(0);
    hTxEopReq.inject(0);
    hForceSe0.inject(1);
    for (var i = 0; i < cycles; i++) {
      await clk.nextPosedge;
    }
    hForceSe0.inject(0);
    // Return to idle J and let the line settle.
    for (var i = 0; i < 60; i++) {
      await clk.nextPosedge;
    }
  }

  // Wait for and capture ONE device packet. Returns {pid, bytes} where bytes
  // excludes the 2 trailing CRC16 bytes (for DATA packets). On timeout returns
  // pid -1 (no response).
  Future<Map<String, dynamic>> expectPkt({int timeout = 8000}) async {
    var guard = 0;
    while (guard < timeout) {
      guard++;
      await clk.nextPosedge;
      if (hrx.output('pkt_done').value.toInt() == 1) {
        final pid = hrx.output('pid').value.toInt();
        final count = hrx.output('byte_count').value.toInt();
        final bytes = <int>[];
        for (var i = 0; i < count; i++) {
          hRdIndex.inject(i);
          await clk.nextPosedge;
          bytes.add(hrx.output('rd_byte').value.toInt());
        }
        hRdIndex.inject(0);
        final payload = count >= 2 ? bytes.sublist(0, count - 2) : <int>[];
        return {'pid': pid, 'bytes': payload};
      }
    }
    return {'pid': -1, 'bytes': <int>[]};
  }

  // SETUP stage: SETUP token (addressed) + DATA0(8 bytes), capture the ACK.
  Future<Map<String, dynamic>> setupStage(
    List<int> bytes, {
    int addr = 0,
  }) async {
    await token(_pidSetup, addr: addr, endp: 0);
    await send(pid: _pidData0, isData: true, payload: bytes);
    return expectPkt();
  }

  // GET_DESCRIPTOR(type, index, wLength) addressed at `addr`. Returns the
  // collected descriptor bytes.
  Future<List<int>> getDescriptor(
    int type,
    int index,
    int wLength, {
    int addr = 0,
  }) async {
    final setup = [
      0x80,
      0x06,
      index & 0xFF,
      type & 0xFF,
      0x00,
      0x00,
      wLength & 0xFF,
      (wLength >> 8) & 0xFF,
    ];
    final ack = await setupStage(setup, addr: addr);
    expect(
      ack['pid'],
      equals(_pidAck),
      reason:
          'device ACKs GET_DESCRIPTOR(type=$type,index=$index) SETUP '
          'at addr=$addr (got pid 0x${(ack['pid'] as int).toRadixString(16)})',
    );
    final collected = <int>[];
    var toggleData1 = true;
    while (true) {
      await token(_pidIn, addr: addr, endp: 0);
      final d = await expectPkt();
      expect(
        d['pid'],
        equals(toggleData1 ? _pidData1 : _pidData0),
        reason: 'IN-data chunk toggles DATA1/DATA0 at addr=$addr',
      );
      final chunk = d['bytes'] as List<int>;
      collected.addAll(chunk);
      await handshake(_pidAck);
      toggleData1 = !toggleData1;
      if (chunk.length < 64) break;
    }
    await token(_pidOut, addr: addr, endp: 0);
    await send(pid: _pidData1, isData: true, payload: const []);
    final st = await expectPkt();
    expect(
      st['pid'],
      equals(_pidAck),
      reason: 'device ACKs the OUT status at addr=$addr',
    );
    return collected;
  }

  // SET_ADDRESS(addr): SETUP(addressed at 0) then IN status, then the device
  // applies the new address after the status stage.
  Future<void> setAddress(int newAddr) async {
    final setup = [0x00, 0x05, newAddr & 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00];
    final ack = await setupStage(setup, addr: 0);
    expect(
      ack['pid'],
      equals(_pidAck),
      reason:
          'device ACKs SET_ADDRESS SETUP (got '
          '0x${(ack['pid'] as int).toRadixString(16)})',
    );
    // IN status: IN token (still addr 0) -> device ZLP DATA1 -> host ACK.
    await token(_pidIn, addr: 0, endp: 0);
    final st = await expectPkt();
    expect(
      st['pid'],
      equals(_pidData1),
      reason: 'SET_ADDRESS IN status is a zero-length DATA1',
    );
    await handshake(_pidAck);
    // Let the device apply the address.
    for (var i = 0; i < 20; i++) {
      await clk.nextPosedge;
    }
  }
}

Future<_Host> _buildHost(_DeviceTop top, Logic clk, Logic reset) async {
  final htx = UsbPacketTx(name: 'host_ptx');
  final hphyTx = HarborUsbFsPhyTx(name: 'host_phytx');
  final hphyRx = HarborUsbFsPhyRx(name: 'host_phyrx');
  final hrx = UsbPacketRx(name: 'host_prx', bufBytes: 80);

  final hSend = Logic(name: 'h_send');
  final hIsData = Logic(name: 'h_is_data');
  final hPid = Logic(name: 'h_pid', width: 8);
  final hPayLen = Logic(name: 'h_payload_len', width: 8);
  final hPayByte = Logic(name: 'h_payload_byte', width: 8);
  final hRdIndex = Logic(name: 'h_rd_index', width: 8);

  // Direct PhyTx bit-stream + SE0 force.
  final hTxData = Logic(name: 'h_tx_data');
  final hTxDataValid = Logic(name: 'h_tx_data_valid');
  final hTxEopReq = Logic(name: 'h_tx_eop_req');
  final hForceSe0 = Logic(name: 'h_force_se0');
  final hUseToken = Logic(name: 'h_use_token');

  // UsbPacketTx feeds the PhyTx for DATA/handshake packets.
  htx.input('clk').srcConnection! <= clk;
  htx.input('reset').srcConnection! <= reset;
  htx.input('send').srcConnection! <= hSend;
  htx.input('is_data').srcConnection! <= hIsData;
  htx.input('pid').srcConnection! <= hPid;
  htx.input('payload_len').srcConnection! <= hPayLen;
  htx.input('payload_byte').srcConnection! <= hPayByte;

  // The host PhyTx accepts EITHER the UsbPacketTx stream (DATA/handshake) OR
  // the direct token bit stream, selected by hUseToken. hUseToken is high
  // whenever the token driver is presenting bits or requesting an EOP.
  hUseToken <= (hTxDataValid | hTxEopReq);
  hphyTx.input('clk').srcConnection! <= clk;
  hphyTx.input('reset').srcConnection! <= reset;
  hphyTx.input('data').srcConnection! <=
      mux(hUseToken, hTxData, htx.output('tx_data'));
  hphyTx.input('data_valid').srcConnection! <=
      mux(hUseToken, hTxDataValid, htx.output('tx_data_valid'));
  hphyTx.input('eop_req').srcConnection! <=
      mux(hUseToken, hTxEopReq, htx.output('tx_eop_req'));
  // The UsbPacketTx pacing always tracks the PhyTx (it only advances when it is
  // actually driving, gated on its own busy/oe, so feeding it the PhyTx
  // handshake while a token is in flight is inert for it).
  htx.input('tx_ready').srcConnection! <= hphyTx.output('ready');
  htx.input('tx_oe').srcConnection! <= hphyTx.output('oe');

  // Host line -> device pads, with an SE0 force override for the bus reset.
  // When hForceSe0 is high, both pads are pulled low (SE0) regardless of the
  // PhyTx. Otherwise the device sees the host PhyTx line.
  final hostDp = mux(hForceSe0, Const(0), hphyTx.output('dp_out'));
  final hostDm = mux(hForceSe0, Const(0), hphyTx.output('dm_out'));
  top.input('dp').srcConnection! <= hostDp;
  top.input('dm').srcConnection! <= hostDm;

  // Device line -> host PhyRx -> host UsbPacketRx.
  hphyRx.input('clk').srcConnection! <= clk;
  hphyRx.input('reset').srcConnection! <= reset;
  hphyRx.input('dp').srcConnection! <= top.output('dp_out');
  hphyRx.input('dm').srcConnection! <= top.output('dm_out');
  hrx.input('clk').srcConnection! <= clk;
  hrx.input('reset').srcConnection! <= reset;
  hrx.input('rx_data').srcConnection! <= hphyRx.output('data');
  hrx.input('rx_valid').srcConnection! <= hphyRx.output('valid');
  hrx.input('rx_sop').srcConnection! <= hphyRx.output('sop');
  hrx.input('rx_eop').srcConnection! <= hphyRx.output('eop');
  hrx.input('rd_index').srcConnection! <= hRdIndex;

  await top.build();
  await htx.build();
  await hphyTx.build();
  await hphyRx.build();
  await hrx.build();

  hSend.inject(0);
  hIsData.inject(0);
  hPid.inject(0);
  hPayLen.inject(0);
  hPayByte.inject(0);
  hRdIndex.inject(0);
  hTxData.inject(0);
  hTxDataValid.inject(0);
  hTxEopReq.inject(0);
  hForceSe0.inject(0);

  return _Host._(
    clk,
    reset,
    hSend,
    hIsData,
    hPid,
    hPayLen,
    hPayByte,
    hRdIndex,
    htx,
    hrx,
    hTxData,
    hTxDataValid,
    hTxEopReq,
    hForceSe0,
    hphyTx,
    top.output('oe'),
  );
}

void main() {
  // Self-check the CRC5 against USB 2.0 spec worked examples (8.3.5.1):
  //   addr=0, endp=0      -> 0x02
  //   addr=0x47, endp=0xA -> 0x17
  group('CRC5 token check', () {
    test('matches USB 2.0 worked examples', () {
      expect(crc5Token(0, 0), equals(0x02));
      // Spec 8.3.5.1 example: 11-bit value 0x547 -> CRC5 0x17. 0x547 has
      // addr = 0x47 & 0x7F = 0x47, endp = 0x547>>7 = 0xA. Verify via the field.
      expect(crc5Token(0x47, 0xA), equals(0x17));
    });
  });

  tearDown(() async {
    await Simulator.reset();
  });

  group('FAITHFUL USB token driver isolation (no bus reset)', () {
    test(
      'DIAGNOSTIC: real token GET_DESCRIPTOR(DEVICE,64) at addr 0 works '
      'WITHOUT a bus reset (isolates the token driver from the bus-reset bug)',
      () async {
        final top = _DeviceTop(name: 'tokonly_top');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        reset.inject(1);
        Simulator.setMaxSimTime(400000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        // NO bus reset: just a real token GET_DESCRIPTOR.
        final dev = await host.getDescriptor(0x01, 0, 64, addr: 0);
        expect(
          dev.length,
          equals(18),
          reason: 'real-token GET_DESCRIPTOR works without bus reset',
        );
        expect(dev[8] | (dev[9] << 8), equals(0x1209));

        await Simulator.endSimulation();
      },
    );
  });

  group(
    'FAITHFUL USB enumeration (real tokens + CRC5 + bus reset + address)',
    () {
      test('full enumeration after a real bus reset, addressed at 7', () async {
        final top = _DeviceTop(name: 'faithful_top');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        reset.inject(1);
        Simulator.setMaxSimTime(800000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        // A real host always bus-resets before enumerating.
        await host.busReset();

        // GET_DESCRIPTOR(DEVICE, 64) at address 0 (the first transaction a real
        // host issues).
        final dev = await host.getDescriptor(0x01, 0, 64, addr: 0);
        expect(dev.length, equals(18), reason: 'device descriptor 18 bytes');
        expect(dev[4], equals(0xFF), reason: 'bDeviceClass vendor-specific');
        expect(dev[8] | (dev[9] << 8), equals(0x1209), reason: 'idVendor');
        expect(dev[10] | (dev[11] << 8), equals(0x10C0), reason: 'idProduct');

        // SET_ADDRESS(7).
        await host.setAddress(7);
        expect(
          top.output('dev_addr').value.toInt(),
          equals(7),
          reason: 'device applied address 7',
        );

        // GET_DESCRIPTOR(DEVICE) full, now ADDRESSED at 7.
        final dev2 = await host.getDescriptor(0x01, 0, 18, addr: 7);
        expect(
          dev2,
          equals(dev),
          reason: 'device descriptor identical at addr 7',
        );

        // GET_DESCRIPTOR(CONFIG) at 7.
        final cfg = await host.getDescriptor(0x02, 0, 32, addr: 7);
        expect(cfg.length, equals(32), reason: 'config tree 32 bytes');
        expect(cfg[18 + 2], equals(0x01), reason: 'EP1 OUT address 0x01');
        expect(cfg[25 + 2], equals(0x81), reason: 'EP1 IN address 0x81');

        // SET_CONFIGURATION(1).
        final scSetup = [0x00, 0x09, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        final scAck = await host.setupStage(scSetup, addr: 7);
        expect(scAck['pid'], equals(_pidAck), reason: 'SET_CONFIGURATION ACK');
        await host.token(_pidIn, addr: 7, endp: 0);
        final scStatus = await host.expectPkt();
        expect(
          scStatus['pid'],
          equals(_pidData1),
          reason: 'SET_CONFIGURATION IN status ZLP',
        );
        await host.handshake(_pidAck);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }
        expect(
          top.output('configured').value.toInt(),
          equals(1),
          reason: 'device configured',
        );

        // String descriptors at 7.
        final s0 = await host.getDescriptor(0x03, 0, 4, addr: 7);
        expect(s0, equals([4, 0x03, 0x09, 0x04]), reason: 'LANGID en-US');
        final s1 = await host.getDescriptor(0x03, 1, 64, addr: 7);
        expect(
          s1,
          equals(_expectedString('Midstall')),
          reason: 'iManufacturer',
        );
        final s2 = await host.getDescriptor(0x03, 2, 64, addr: 7);
        expect(s2, equals(_expectedString('Loom')), reason: 'iProduct');
        final s3 = await host.getDescriptor(0x03, 3, 64, addr: 7);
        expect(
          s3,
          equals(_expectedString('Loom Command Interface')),
          reason: 'iInterface',
        );

        await Simulator.endSimulation();
      });
    },
  );

  // USB 2.0 9.1.1.6 requires the device to return to the Default state
  // (address 0, not configured, control FSM IDLE, data toggles cleared) on a
  // bus reset, including mid-transfer and on retries.
  //
  // Scenario:
  //   1. bus reset, enumerate, SET_ADDRESS(7) -> dev_addr == 7.
  //   2. SECOND bus reset mid-stream -> dev_addr MUST snap back to 0.
  //   3. The device must answer GET_DESCRIPTOR at address 0 again (proving the
  //      whole control path reset, not just the address register).
  //   4. A bus reset DURING a half-finished control transfer cleanly resets the
  //      engine: a fresh GET_DESCRIPTOR at address 0 still works afterwards.
  group('FAITHFUL USB bus-reset recovery (the hardware-bug catcher)', () {
    test(
      'a bus reset returns the device to the Default state (addr 0)',
      () async {
        final top = _DeviceTop(name: 'busreset_top');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        reset.inject(1);
        Simulator.setMaxSimTime(1200000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        // 1. Host bus-resets, enumerates, addresses the device at 7.
        await host.busReset();
        final dev = await host.getDescriptor(0x01, 0, 64, addr: 0);
        expect(dev.length, equals(18), reason: 'device descriptor at addr 0');
        await host.setAddress(7);
        expect(
          top.output('dev_addr').value.toInt(),
          equals(7),
          reason: 'device applied address 7 before the second bus reset',
        );

        // 2. SECOND bus reset (the host's reset+retry cycle). The device MUST
        //    return to the Default state: address back to 0.
        await host.busReset();
        expect(
          top.output('dev_addr').value.toInt(),
          equals(0),
          reason:
              'BUS RESET must return the device to address 0 (USB 9.1.1.6); '
              'before the fix the engine kept the stale address 7 across the '
              "host's reset+retry, which is the -32/-71 hardware failure",
        );
        expect(
          top.output('configured').value.toInt(),
          equals(0),
          reason: 'bus reset must clear the configured flag',
        );

        // 3. The device must answer at address 0 again (the FIRST transaction a
        //    real host issues after the bus reset is GET_DESCRIPTOR(DEVICE) at 0).
        final dev2 = await host.getDescriptor(0x01, 0, 18, addr: 0);
        expect(
          dev2,
          equals(dev),
          reason: 'device answers GET_DESCRIPTOR at addr 0 after a bus reset',
        );

        await Simulator.endSimulation();
      },
    );

    test('a bus reset MID control transfer cleanly resets the engine', () async {
      final top = _DeviceTop(name: 'midreset_top');
      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      top.port('clk').getsLogic(clk);
      top.input('reset').srcConnection! <= reset;
      final host = await _buildHost(top, clk, reset);

      reset.inject(1);
      Simulator.setMaxSimTime(1200000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }

      // Enumerate + address at 7 first, so the engine is in a non-default state.
      await host.busReset();
      await host.getDescriptor(0x01, 0, 64, addr: 0);
      await host.setAddress(7);
      expect(top.output('dev_addr').value.toInt(), equals(7));

      // Begin a NEW control transfer but ABANDON it with a bus reset partway:
      // send only the SETUP token (addressed at 7) then drive a bus reset
      // before the DATA0/IN stages. This is the host-aborts-mid-transfer case.
      await host.token(_pidSetup, addr: 7, endp: 0);
      await host.busReset();

      // The engine must be fully back in the Default state.
      expect(
        top.output('dev_addr').value.toInt(),
        equals(0),
        reason: 'mid-transfer bus reset returns the device to address 0',
      );

      // A completely fresh enumeration at address 0 must succeed: proves the
      // control FSM, capture walk and data toggles all reset (not just addr).
      final dev = await host.getDescriptor(0x01, 0, 64, addr: 0);
      expect(
        dev.length,
        equals(18),
        reason:
            'fresh GET_DESCRIPTOR at addr 0 works after a mid-transfer '
            'bus reset',
      );
      expect(dev[8] | (dev[9] << 8), equals(0x1209), reason: 'idVendor intact');

      await Simulator.endSimulation();
    });
  });
}
