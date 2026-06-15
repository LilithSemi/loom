// BIDIRECTIONAL-PAD USB device model.
//
// Other sim hosts (usb_faithful_enum_test.dart, usb_harbor_engine_test) wire
// the device's dp/dm input straight from the host line and dp_out/dm_out
// straight to the host RX: a unidirectional pad model where the device never
// reads back its own transmitted line.
//
// Real OrangeCrab hardware is bidirectional. The LoomTop shim in
// `ip/bin/loom_genip.dart` folds the split dp(in)/dp_out/oe pads onto ONE
// inout ball each:
//   assign usb_dp = oe ? dp_out : 1'bz;   // device drives only when oe
//   assign soc_dp_in = usb_dp;            // device ALWAYS reads the resolved pad
// So when the device transmits (oe high) its own RX sees its own dp_out, and
// when oe is low it sees the host. The host's RX always sees the resolved pad.
//
// This harness models that resolution:
//   line_dp = oe ? dev_dp_out : host_dp
//   line_dm = oe ? dev_dm_out : host_dm
//   device.dp = line_dp ; device.dm = line_dm   (reads its own TX when oe)
//   host_rx.dp = line_dp ; host_rx.dm = line_dm
//
// A VCD of a failing IN-DATA transaction is dumped to build/usb_bidir.vcd
// when USBVCD=1.

import 'dart:async';
import 'dart:io';

import 'package:harbor/harbor.dart';
import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/hw/usb_device.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

const _pidSetup = 0x2D;
const _pidIn = 0x69;
const _pidData0 = 0xC3;
const _pidData1 = 0x4B;
const _pidAck = 0xD2;

bool get _vcd => Platform.environment['USBVCD'] == '1';

int crc5Token(int addr, int endp) {
  final field = (addr & 0x7F) | ((endp & 0xF) << 7);
  var crc = 0x1F;
  for (var i = 0; i < 11; i++) {
    final bit = (field >> i) & 1;
    final xorIn = bit ^ (crc & 1);
    crc >>= 1;
    if (xorIn != 0) crc ^= 0x14;
  }
  return (~crc) & 0x1F;
}

/// Device top with a TRUE BIDIRECTIONAL pad. The device's dp/dm INPUT is the
/// resolved line (its own dp_out when oe, else the host line), exactly like the
/// LoomTop inout shim. Host drives host_dp/host_dm. Host RX reads the resolved
/// line via the line_dp/line_dm outputs.
class _BidirTop extends BridgeModule {
  _BidirTop({String? name}) : super('_BidirTop', name: name ?? 'bidir_top') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('host_dp', PortDirection.input);
    createPort('host_dm', PortDirection.input);
    addOutput('line_dp');
    addOutput('line_dm');
    addOutput('oe');
    addOutput('dev_addr', width: 7);
    addOutput('configured');

    final clk = input('clk');
    final reset = input('reset');

    final ep0 = UsbEp0Engine(
      name: 'dev_ep0',
      descriptors: LoomUsbDescriptorRom.descriptorEntries(),
      bulkEndpoints: true,
      filterByAddress: true,
    );
    addSubModule(ep0);
    ep0.input('clk').srcConnection! <= clk;
    ep0.input('reset').srcConnection! <= reset;

    final oe = ep0.output('oe');
    final devDp = ep0.output('dp_out');
    final devDm = ep0.output('dm_out');

    // The inout pad resolution: when the device drives (oe), the line carries
    // the device output. Otherwise it carries the host drive.
    final lineDp = mux(oe, devDp, input('host_dp'));
    final lineDm = mux(oe, devDm, input('host_dm'));

    // The device ALWAYS reads back the resolved pad, modeling the real
    // bidirectional pad.
    ep0.input('dp').srcConnection! <= lineDp;
    ep0.input('dm').srcConnection! <= lineDm;

    output('line_dp') <= lineDp;
    output('line_dm') <= lineDm;
    output('oe') <= oe;
    output('dev_addr') <= ep0.output('dev_addr');
    output('configured') <= ep0.output('configured');

    final engine = LoomUsbCmdEngine(
      config: const LoomUsbDeviceConfig(busAddressWidth: 12, busDataWidth: 32),
      name: 'dev_cmd_engine',
    );
    addSubModule(engine);
    engine.input('clk').srcConnection! <= clk;
    // Mirror LoomUsbDevice: reset the cmd engine on FPGA reset OR USB bus reset.
    engine.input('reset').srcConnection! <= (reset | ep0.output('bus_reset'));
    engine.input('cmd_data').srcConnection! <= ep0.output('cmd_data');
    engine.input('cmd_valid').srcConnection! <= ep0.output('cmd_valid');
    engine.input('resp_ready').srcConnection! <= ep0.output('resp_ready');
    engine.input('cmd_start').srcConnection! <= ep0.output('cmd_start');
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

class _Host {
  final Logic clk;
  final Logic reset;
  final Logic hSend;
  final Logic hIsData;
  final Logic hPid;
  final Logic hPayLen;
  final Logic hPayByte;
  final Logic hRdIndex;
  final UsbPacketTx htx;
  final UsbPacketRx hrx;
  final Logic hTxData;
  final Logic hTxDataValid;
  final Logic hTxEopReq;
  final Logic hForceSe0;
  final HarborUsbFsPhyTx hphyTx;
  final Logic devOe;
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
  // has fully released it (its tristate drivers off). On the real bidirectional
  // pad the device's drive WINS while oe is high, so a host that starts its
  // token too early would have its SYNC eaten by the device's EOP tail. Wait
  // for the device to release the bus, then a short inter-packet gap, before we
  // drive.
  Future<void> _awaitBusIdle() async {
    // Debounce oe low for several cycles, since it can glitch low mid-EOP.
    var guard = 0;
    var lowRun = 0;
    while (lowRun < 16 && guard < 4000) {
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

  /// When true, token() uses the TIGHT real-host turnaround (a short
  /// inter-packet gap after the device drops oe) instead of the lenient
  /// _awaitBusIdle (16 idle cycles). This stresses the device's OUT-ACK ->
  /// drain -> read -> IN-emit handoff and the capIdx re-zero on return to IDLE
  /// the way a real host's back-to-back polling does.
  bool tightTokens = false;

  Future<void> token(int pid, {int addr = 0, int endp = 0}) async {
    if (tightTokens) {
      await _awaitTightTurnaround();
    } else {
      await _awaitBusIdle();
    }
    final crc = crc5Token(addr, endp);
    final byte0 = (addr & 0x7F) | ((endp & 1) << 7);
    final byte1 = ((endp >> 1) & 0x7) | ((crc & 0x1F) << 3);
    final bytes = <int>[0x80, pid & 0xFF, byte0, byte1];
    final bits = <int>[];
    for (final b in bytes) {
      for (var i = 0; i < 8; i++) {
        bits.add((b >> i) & 1);
      }
    }
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
    hTxDataValid.inject(0);
    hTxEopReq.inject(1);
    guard = 0;
    while (guard < 4000) {
      guard++;
      await clk.nextPosedge;
      if (hphyTx.output('oe').value.toInt() == 0 &&
          hphyTx.output('busy').value.toInt() == 0) {
        break;
      }
    }
    hTxEopReq.inject(0);
    for (var i = 0; i < 30; i++) {
      await clk.nextPosedge;
    }
  }

  Future<void> handshake(int pid) => send(pid: pid, isData: false);

  /// FAITHFUL bus turnaround: real host ACKs an IN-DATA after the minimum USB
  /// inter-packet gap (a couple bit-times), not after long idle like
  /// _awaitBusIdle (16 cycles) does. Forces the device to squelch-recover its
  /// RX from its own IN-DATA TX in time to decode a tightly-following ACK.
  Future<void> _awaitTightTurnaround() async {
    // Debounce oe low, briefly (not the 16-cycle idle _awaitBusIdle uses).
    var guard = 0;
    var lowRun = 0;
    while (lowRun < 2 && guard < 4000) {
      guard++;
      lowRun = devOe.value.toInt() == 0 ? lowRun + 1 : 0;
      await clk.nextPosedge;
    }
    // Minimum inter-packet gap: ~2 USB bit-times (8 oversample cycles). A real
    // host turns the bus around this fast. The device's squelch-recovery must
    // have re-locked the RX by now.
    for (var i = 0; i < _turnaroundGap; i++) {
      await clk.nextPosedge;
    }
  }

  /// Inter-packet turnaround gap in oversample cycles after the device drops oe.
  /// Configurable so a test can squeeze the host ACK right up against the
  /// device's EOP (the worst-case real-host turnaround).
  final int _turnaroundGap = 8;

  /// Send a handshake with the TIGHT (faithful) turnaround instead of the
  /// lenient _awaitBusIdle. Mirrors send() but skips the long idle wait.
  Future<void> handshakeTight(int pid) async {
    await _awaitTightTurnaround();
    _curPayload = const [];
    hIsData.inject(0);
    hPid.inject(pid);
    hPayLen.inject(0);
    hSend.inject(1);
    await clk.nextPosedge;
    hSend.inject(0);
    var guard = 0;
    while (htx.output('busy').value.toInt() == 0 && guard < 50) {
      guard++;
      await clk.nextPosedge;
    }
    guard = 0;
    while (htx.output('done').value.toInt() == 0 && guard < 4000) {
      guard++;
      await clk.nextPosedge;
    }
    for (var i = 0; i < 30; i++) {
      await clk.nextPosedge;
    }
  }

  /// FAITHFUL EP1 bulk IN: send the IN token, read the device DATA (or NAK),
  /// then if DATA, ACK it with the TIGHT real-host turnaround (the device must
  /// squelch-recover from its own IN-DATA TX to decode this ACK).
  Future<Map<String, dynamic>> bulkInFaithful({int addr = 0}) async {
    await token(_pidIn, addr: addr, endp: 1);
    final d = await expectPkt();
    final pid = d['pid'] as int;
    if (pid == _pidData0 || pid == _pidData1) {
      await handshakeTight(_pidAck);
    }
    return d;
  }

  Future<void> busReset({int cycles = 200}) async {
    hTxDataValid.inject(0);
    hTxEopReq.inject(0);
    hForceSe0.inject(1);
    for (var i = 0; i < cycles; i++) {
      await clk.nextPosedge;
    }
    hForceSe0.inject(0);
    for (var i = 0; i < 60; i++) {
      await clk.nextPosedge;
    }
  }

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

  Future<Map<String, dynamic>> setupStage(
    List<int> bytes, {
    int addr = 0,
  }) async {
    await token(_pidSetup, addr: addr, endp: 0);
    await send(pid: _pidData0, isData: true, payload: bytes);
    return expectPkt();
  }

  /// EP1 bulk OUT: send a token(OUT, endp=1) then a DATA0/DATA1 packet with the
  /// command bytes, and read the device's handshake (expect ACK).
  Future<Map<String, dynamic>> bulkOut(
    List<int> bytes, {
    required bool data1,
    int addr = 0,
  }) async {
    await token(_pidOut, addr: addr, endp: 1);
    await send(
      pid: data1 ? _pidData1 : _pidData0,
      isData: true,
      payload: bytes,
    );
    return expectPkt();
  }

  /// EP1 bulk IN: send a token(IN, endp=1), read the device's DATA packet (or
  /// NAK), then if it was DATA, ACK it.
  Future<Map<String, dynamic>> bulkIn({int addr = 0}) async {
    await token(_pidIn, addr: addr, endp: 1);
    final d = await expectPkt();
    final pid = d['pid'] as int;
    if (pid == _pidData0 || pid == _pidData1) {
      await handshake(_pidAck);
    }
    return d;
  }
}

const _pidOut = 0xE1;

Future<_Host> _buildHost(_BidirTop top, Logic clk, Logic reset) async {
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

  final hTxData = Logic(name: 'h_tx_data');
  final hTxDataValid = Logic(name: 'h_tx_data_valid');
  final hTxEopReq = Logic(name: 'h_tx_eop_req');
  final hForceSe0 = Logic(name: 'h_force_se0');
  final hUseToken = Logic(name: 'h_use_token');

  htx.input('clk').srcConnection! <= clk;
  htx.input('reset').srcConnection! <= reset;
  htx.input('send').srcConnection! <= hSend;
  htx.input('is_data').srcConnection! <= hIsData;
  htx.input('pid').srcConnection! <= hPid;
  htx.input('payload_len').srcConnection! <= hPayLen;
  htx.input('payload_byte').srcConnection! <= hPayByte;

  hUseToken <= (hTxDataValid | hTxEopReq);
  hphyTx.input('clk').srcConnection! <= clk;
  hphyTx.input('reset').srcConnection! <= reset;
  hphyTx.input('data').srcConnection! <=
      mux(hUseToken, hTxData, htx.output('tx_data'));
  hphyTx.input('data_valid').srcConnection! <=
      mux(hUseToken, hTxDataValid, htx.output('tx_data_valid'));
  hphyTx.input('eop_req').srcConnection! <=
      mux(hUseToken, hTxEopReq, htx.output('tx_eop_req'));
  htx.input('tx_ready').srcConnection! <= hphyTx.output('ready');
  htx.input('tx_oe').srcConnection! <= hphyTx.output('oe');

  // Host line -> device pads (with SE0 force for bus reset). The host only
  // drives the line when the DEVICE is not driving (the device's oe wins on
  // the real inout, which the _BidirTop mux models). Here we present the host
  // drive; _BidirTop resolves oe.
  final hostDp = mux(hForceSe0, Const(0), hphyTx.output('dp_out'));
  final hostDm = mux(hForceSe0, Const(0), hphyTx.output('dm_out'));
  top.input('host_dp').srcConnection! <= hostDp;
  top.input('host_dm').srcConnection! <= hostDm;

  // Host RX reads the RESOLVED line (the real inout pad), not the device's raw
  // dp_out. This is what a real host scope sees.
  hphyRx.input('clk').srcConnection! <= clk;
  hphyRx.input('reset').srcConnection! <= reset;
  hphyRx.input('dp').srcConnection! <= top.output('line_dp');
  hphyRx.input('dm').srcConnection! <= top.output('line_dm');
  // The host receiver is a plain (non-squelchable) PhyRx: it never transmits
  // while reading the device, so it needs no TX squelch.
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
  tearDown(() async {
    await Simulator.reset();
  });

  group('BIDIRECTIONAL pad: device reads back its own TX', () {
    test(
      'GET_DESCRIPTOR(DEVICE,64) at addr 0 IN-DATA is delivered to the host',
      () async {
        final top = _BidirTop(name: 'bidir_dev');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        if (_vcd) {
          Directory('build').createSync(recursive: true);
          WaveDumper(top, outputPath: 'build/usb_bidir.vcd');
        }

        reset.inject(1);
        Simulator.setMaxSimTime(400000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        await host.busReset();

        // SETUP stage: this part the prior agent saw SUCCEED even on hardware.
        final setup = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x40, 0x00];
        final ack = await host.setupStage(setup, addr: 0);
        expect(
          ack['pid'],
          equals(_pidAck),
          reason:
              'device ACKs the GET_DESCRIPTOR SETUP (SETUP+ACK succeed even '
              'on hardware)',
        );

        // IN-DATA stage: this is where hardware returns -32. With a bidirectional
        // pad the device reads back its own TX while transmitting. If that
        // corrupts the SYNC/PID the host never decodes the DATA1 packet. A real
        // host retries the IN a few times before giving up. Mirror that.
        Map<String, dynamic> d = {'pid': -1, 'bytes': <int>[]};
        for (var attempt = 0; attempt < 4; attempt++) {
          await host.token(_pidIn, addr: 0, endp: 0);
          d = await host.expectPkt();
          if (d['pid'] != -1) break;
        }

        expect(
          d['pid'],
          equals(_pidData1),
          reason:
              'BIDIR: the device IN-DATA (DATA1) packet must be delivered '
              'to the host. pid -1 == not delivered == hardware -32; a wrong '
              'pid == corrupted SYNC/PID == hardware -32 too. Got '
              'pid=0x${(d['pid'] as int).toRadixString(16)}',
        );
        final chunk = d['bytes'] as List<int>;
        expect(
          chunk.length,
          equals(18),
          reason: 'device descriptor is 18 bytes',
        );
        expect(chunk[8] | (chunk[9] << 8), equals(0x1209), reason: 'idVendor');
        expect(
          chunk[10] | (chunk[11] << 8),
          equals(0x10C0),
          reason: 'idProduct',
        );

        await Simulator.endSimulation();
      },
    );

    test(
      'EP1 BULK: IN-only probe (no cmd) must NAK, never STALL/garble',
      () async {
        final top = _BidirTop(name: 'bidir_innak');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        if (_vcd) {
          Directory('build').createSync(recursive: true);
          WaveDumper(top, outputPath: 'build/usb_bidir_innak.vcd');
        }

        reset.inject(1);
        Simulator.setMaxSimTime(400000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }
        await host.busReset();

        // Exactly the host's first action: IN token to EP1 with nothing queued.
        await host.token(_pidIn, addr: 0, endp: 1);
        final d = await host.expectPkt();
        // Device should NAK (pid 0x5A). pid -1 = nothing on the wire = host sees
        // no response = will eventually mark the pipe errored.
        expect(
          d['pid'],
          equals(0x5A),
          reason:
              'IN with no queued response must NAK. Got '
              'pid=0x${(d['pid'] as int).toRadixString(16)}',
        );

        await Simulator.endSimulation();
      },
    );

    test(
      'EP1 BULK: READ VERSION returns 0x4C4F4F4D over bulk OUT+IN',
      () async {
        final top = _BidirTop(name: 'bidir_bulk');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        if (_vcd) {
          Directory('build').createSync(recursive: true);
          WaveDumper(top, outputPath: 'build/usb_bidir_bulk.vcd');
        }

        reset.inject(1);
        Simulator.setMaxSimTime(800000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        await host.busReset();

        // Bulk OUT: READ command for VERSION @ 0x000, len=4.
        // header = opcode(0x02) addr(0x00000000 LE) len(0x0004 LE)
        final readCmd = [0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00];
        final outAck = await host.bulkOut(readCmd, data1: false, addr: 0);
        expect(
          outAck['pid'],
          equals(_pidAck),
          reason:
              'device must ACK the bulk OUT command DATA packet. Got '
              'pid=0x${(outAck['pid'] as int).toRadixString(16)}',
        );

        // Bulk IN: read 4 response bytes (the device sends 1 byte per IN DATA
        // packet). Retry NAKs (device is doing a Wishbone read).
        final resp = <int>[];
        var inAttempts = 0;
        while (resp.length < 4 && inAttempts < 60) {
          inAttempts++;
          final d = await host.bulkIn(addr: 0);
          final pid = d['pid'] as int;
          if (pid == _pidData0 || pid == _pidData1) {
            resp.addAll(d['bytes'] as List<int>);
          }
        }

        expect(
          resp.length,
          greaterThanOrEqualTo(4),
          reason:
              'device must return 4 VERSION bytes over bulk IN. Got '
              '${resp.length}: $resp after $inAttempts IN tokens',
        );
        final v = resp[0] | (resp[1] << 8) | (resp[2] << 16) | (resp[3] << 24);
        expect(
          v,
          equals(0x4C4F4F4D),
          reason:
              'VERSION over bulk must be 0x4C4F4F4D (LOOM). Got '
              '0x${v.toRadixString(16)}',
        );

        await Simulator.endSimulation();
      },
    );

    test('EP1 BULK: a single IN DATA packet carries the FULL 4-byte VERSION '
        '(not a 1-byte short packet)', () async {
      // A single bulk IN must return the full response in ONE DATA packet (up
      // to wMaxPacketSize=64): one IN token must yield a DATA packet whose
      // payload is the complete 4-byte VERSION 0x4D 0x4F 0x4F 0x4C.
      final top = _BidirTop(name: 'bidir_onepkt');
      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      top.port('clk').getsLogic(clk);
      top.input('reset').srcConnection! <= reset;
      final host = await _buildHost(top, clk, reset);

      if (_vcd) {
        Directory('build').createSync(recursive: true);
        WaveDumper(top, outputPath: 'build/usb_bidir_onepkt.vcd');
      }

      reset.inject(1);
      Simulator.setMaxSimTime(800000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }

      await host.busReset();

      // Bulk OUT: READ VERSION @ 0x000, len=4.
      final readCmd = [0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00];
      final outAck = await host.bulkOut(readCmd, data1: false, addr: 0);
      expect(
        outAck['pid'],
        equals(_pidAck),
        reason: 'device must ACK the bulk OUT command DATA packet',
      );

      // Bulk IN: a single IN token must return the WHOLE response in one DATA
      // packet. The device may NAK a few times while it does the Wishbone read;
      // the FIRST non-NAK DATA packet must already carry all 4 bytes.
      Map<String, dynamic> data = {'pid': -1, 'bytes': <int>[]};
      var inAttempts = 0;
      while (inAttempts < 60) {
        inAttempts++;
        final d = await host.bulkIn(addr: 0);
        final pid = d['pid'] as int;
        if (pid == _pidData0 || pid == _pidData1) {
          data = d;
          break;
        }
      }

      expect(
        data['pid'],
        anyOf(equals(_pidData0), equals(_pidData1)),
        reason:
            'device must return a DATA packet (not NAK forever). Got '
            'pid=0x${(data['pid'] as int).toRadixString(16)} after '
            '$inAttempts IN tokens',
      );
      final chunk = data['bytes'] as List<int>;
      expect(
        chunk.length,
        equals(4),
        reason:
            'THE FIX: a single IN DATA packet must carry the full 4-byte '
            'response, not 1 byte. Got ${chunk.length} byte(s): $chunk',
      );
      final v =
          chunk[0] | (chunk[1] << 8) | (chunk[2] << 16) | (chunk[3] << 24);
      expect(
        v,
        equals(0x4C4F4F4D),
        reason:
            'the single IN packet payload must be VERSION 0x4C4F4F4D '
            '(0x4D 0x4F 0x4F 0x4C). Got 0x${v.toRadixString(16)} from $chunk',
      );

      await Simulator.endSimulation();
    });

    test('FAITHFUL FIRST-READ: a single bulk OUT READ then TIGHT back-to-back IN '
        'polling (no lenient idle gap) must still return 0x4C4F4F4D', () async {
      // Mirrors tools/loom_usb_test.py: polls IN back-to-back with the minimum
      // inter-packet turnaround (not the 16-idle-cycle gap _awaitBusIdle uses),
      // stressing the OUT-ACK -> drain -> Wishbone-read -> IN-emit handoff and
      // the capIdx re-zero under tight timing.
      final top = _BidirTop(name: 'bidir_firstread');
      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      top.port('clk').getsLogic(clk);
      top.input('reset').srcConnection! <= reset;
      final host = await _buildHost(top, clk, reset);
      host.tightTokens = true;

      if (_vcd) {
        Directory('build').createSync(recursive: true);
        WaveDumper(top, outputPath: 'build/usb_bidir_firstread.vcd');
      }

      reset.inject(1);
      Simulator.setMaxSimTime(1600000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      for (var i = 0; i < 20; i++) {
        await clk.nextPosedge;
      }
      await host.busReset();

      final readCmd = [0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00];
      final outAck = await host.bulkOut(readCmd, data1: false, addr: 0);
      expect(
        outAck['pid'],
        equals(_pidAck),
        reason: 'device must ACK the bulk OUT command DATA packet',
      );

      final resp = <int>[];
      var inAttempts = 0;
      while (resp.length < 4 && inAttempts < 40) {
        inAttempts++;
        final d = await host.bulkInFaithful(addr: 0);
        final pid = d['pid'] as int;
        if (pid == _pidData0 || pid == _pidData1) {
          resp.addAll(d['bytes'] as List<int>);
        }
      }
      expect(
        resp.length,
        greaterThanOrEqualTo(4),
        reason:
            'FAITHFUL FIRST-READ: device must return 4 VERSION bytes with '
            'tight back-to-back IN polling. Got ${resp.length}: $resp after '
            '$inAttempts IN tokens (an IN that NAKs forever here == the '
            'real-hardware errno 110 first-read wedge)',
      );
      final v = resp[0] | (resp[1] << 8) | (resp[2] << 16) | (resp[3] << 24);
      expect(
        v,
        equals(0x4C4F4F4D),
        reason:
            'FIRST-READ VERSION must be 0x4C4F4F4D. Got '
            '0x${v.toRadixString(16)}',
      );

      await Simulator.endSimulation();
    });

    test(
      'FAITHFUL WEDGE: READ VERSION over bulk OUT+IN with TIGHT real-host ACK '
      'turnaround returns 0x4C4F4F4D and a following SETUP still decodes',
      () async {
        // Differs from the READ VERSION test above in ACK turnaround: here the
        // host ACKs the device's EP1 IN-DATA after the minimum USB inter-packet
        // gap (bulkInFaithful -> handshakeTight), as a real xHCI host does,
        // instead of waiting many idle cycles. The device must squelch-recover
        // its RX from its own IN-DATA TX in time to decode that
        // tightly-following ACK before flipping its toggle and emitting the
        // next byte.
        final top = _BidirTop(name: 'bidir_faithful');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        if (_vcd) {
          Directory('build').createSync(recursive: true);
          WaveDumper(top, outputPath: 'build/usb_bidir_faithful.vcd');
        }

        reset.inject(1);
        Simulator.setMaxSimTime(1600000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        await host.busReset();

        // Mirrors tools/loom_usb_test.py TEST 1: interleaves a partial read with
        // a command re-send while the device still has a multi-byte response in
        // flight, with tight real-host turnarounds and a tight OUT->IN gap (the
        // device NAKs IN while it does the Wishbone read).
        //   B. OUT-only probe : send the READ-VERSION command (4 bytes queued).
        //   C. short read     : read just ONE byte of the 4-byte response, STOP.
        //   D. re-send OUT    : send the READ-VERSION command AGAIN.
        //   D. full read      : read 4 bytes.
        final readCmd = [0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00];

        final outAck1 = await host.bulkOut(readCmd, data1: false, addr: 0);
        expect(
          outAck1['pid'],
          equals(_pidAck),
          reason: 'device must ACK the first bulk OUT command DATA packet',
        );

        var shortBytes = 0;
        var shortAttempts = 0;
        while (shortBytes < 1 && shortAttempts < 20) {
          shortAttempts++;
          final d = await host.bulkInFaithful(addr: 0);
          final pid = d['pid'] as int;
          if (pid == _pidData0 || pid == _pidData1) {
            shortBytes += (d['bytes'] as List<int>).length;
          }
        }
        expect(
          shortBytes,
          greaterThanOrEqualTo(1),
          reason:
              'FAITHFUL step C: the short 1-byte IN read must return a byte. '
              'Got $shortBytes after $shortAttempts IN tokens. A wedge here == '
              'the first IN-DATA-after-OUT corrupted (the errno 32/110 the user '
              'reports).',
        );

        // Step D: re-send OUT while step C's response bytes are still in flight.
        final outAck2 = await host.bulkOut(readCmd, data1: true, addr: 0);
        expect(
          outAck2['pid'],
          equals(_pidAck),
          reason:
              'FAITHFUL step D: device must ACK the RE-SENT bulk OUT command '
              'even though it still had response bytes in flight. A non-ACK here '
              '== the OUT endpoint wedged (cmd engine stuck mid-read; the drain '
              'never completes). Got '
              'pid=0x${(outAck2['pid'] as int).toRadixString(16)}',
        );

        final resp = <int>[];
        var inAttempts = 0;
        while (resp.length < 4 && inAttempts < 30) {
          inAttempts++;
          final d = await host.bulkInFaithful(addr: 0);
          final pid = d['pid'] as int;
          if (pid == _pidData0 || pid == _pidData1) {
            resp.addAll(d['bytes'] as List<int>);
          }
        }

        expect(
          resp.length,
          greaterThanOrEqualTo(4),
          reason:
              'FAITHFUL step D: device must return 4 VERSION bytes over the '
              're-sent READ. Got ${resp.length}: $resp after $inAttempts IN '
              'tokens (a short count == the engine wedged on the interleaved '
              'OUT-after-partial-read == the errno 32/110 wedge)',
        );
        final v = resp[0] | (resp[1] << 8) | (resp[2] << 16) | (resp[3] << 24);
        expect(
          v,
          equals(0x4C4F4F4D),
          reason:
              'FAITHFUL VERSION must be 0x4C4F4F4D (LOOM). Got '
              '0x${v.toRadixString(16)} from bytes $resp',
        );

        // No wedge: a control SETUP right after must still decode + ACK.
        final setup = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x40, 0x00];
        final ack = await host.setupStage(setup, addr: 0);
        expect(
          ack['pid'],
          equals(_pidAck),
          reason:
              'FAITHFUL: device must still decode + ACK a SETUP after the '
              'bulk IN sequence (RX re-locked). Got '
              'pid=0x${(ack['pid'] as int).toRadixString(16)}',
        );

        await Simulator.endSimulation();
      },
    );

    test(
      'WEDGE REPRO: an EP1 IN DATA response must NOT desync the device RX - a '
      'following control SETUP still decodes (no PID misdecode / stuck FSM)',
      () async {
        // After the device transmits an EP1 IN response, its RX must still
        // decode a following control SETUP correctly (no PID misdecode or
        // stuck FSM from the TX->RX turnaround).
        final top = _BidirTop(name: 'bidir_wedge');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        if (_vcd) {
          Directory('build').createSync(recursive: true);
          WaveDumper(top, outputPath: 'build/usb_bidir_wedge.vcd');
        }

        reset.inject(1);
        Simulator.setMaxSimTime(800000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        await host.busReset();

        // 1) Drive a real EP1 IN DATA response: bulk OUT a READ then bulk IN it.
        final readCmd = [0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00];
        final outAck = await host.bulkOut(readCmd, data1: false, addr: 0);
        expect(
          outAck['pid'],
          equals(_pidAck),
          reason: 'device must ACK the bulk OUT command DATA packet',
        );

        var gotInData = false;
        var inAttempts = 0;
        while (!gotInData && inAttempts < 60) {
          inAttempts++;
          final d = await host.bulkIn(addr: 0);
          final pid = d['pid'] as int;
          if (pid == _pidData0 || pid == _pidData1) {
            gotInData = true; // a real device-initiated IN DATA hit the wire
          }
        }
        expect(
          gotInData,
          isTrue,
          reason:
              'precondition: the device must have transmitted at least one '
              'EP1 IN DATA response so the turnaround is actually exercised',
        );

        // 2) Immediately after the EP1 IN response, run a CONTROL transfer. If the
        // EP1 IN-response turnaround desynced the device RX, the device would
        // misdecode this SETUP PID and never ACK (the wedge). It must ACK and
        // deliver the device descriptor.
        final setup = [0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x40, 0x00];
        final ack = await host.setupStage(setup, addr: 0);
        expect(
          ack['pid'],
          equals(_pidAck),
          reason:
              'WEDGE: device must still decode + ACK a SETUP AFTER an EP1 IN '
              'response. A non-ACK here == the RX desynced on the IN-response '
              'turnaround (the original bug). Got '
              'pid=0x${(ack['pid'] as int).toRadixString(16)}',
        );

        Map<String, dynamic> d = {'pid': -1, 'bytes': <int>[]};
        for (var attempt = 0; attempt < 4; attempt++) {
          await host.token(_pidIn, addr: 0, endp: 0);
          d = await host.expectPkt();
          if (d['pid'] != -1) break;
        }
        expect(
          d['pid'],
          equals(_pidData1),
          reason:
              'WEDGE: the control IN-DATA following an EP1 IN response must '
              'be delivered (RX re-locked). Got '
              'pid=0x${(d['pid'] as int).toRadixString(16)}',
        );
        final chunk = d['bytes'] as List<int>;
        expect(
          chunk.length,
          equals(18),
          reason: 'device descriptor is 18 bytes',
        );
        expect(chunk[8] | (chunk[9] << 8), equals(0x1209), reason: 'idVendor');

        await Simulator.endSimulation();
      },
    );
  });
}
