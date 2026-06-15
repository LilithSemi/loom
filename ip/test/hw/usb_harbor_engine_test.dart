// Full raw dp/dm sim coverage for the CONSOLIDATED USB front-end: harbor's
// proven UsbEp0Engine, configured with loom's VENDOR descriptors (injected) and
// bulkEndpoints enabled, wired to the real LoomUsbCmdEngine + LoomAccelerator.
//
// Drives a host USB packet layer (the SAME harbor UsbPacketTx/PhyTx/PhyRx/
// UsbPacketRx the device uses) across the crossed D+/D- line into the engine
// and asserts:
//   (a) FULL ENUMERATION including ALL THREE string descriptors, byte-exact
//       (harbor's FSM resets its IN-data state per transfer).
//   (b) a FULL bulk OUT transaction (OUT token + DATA0 + device ACK) delivering
//       command bytes into the command stream.
//   (c) a FULL bulk IN transaction (IN token -> device DATA -> host ACK. And a
//       NAK when the command engine has no response byte ready).
//   (d) a command-engine ROUND-TRIP over the wire: bulk OUT a READ-VERSION
//       command, then bulk IN the 4 response bytes == 0x4C4F4F4D ("LOOM").
//
// Kept in its own SMALL file: dart test parallelizes per file, and a single
// isolate accumulates per-wire sim listeners across tests, so a focused file
// keeps the per-tick fan-out (and wall time) bounded.

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
const _pidNak = 0x5A;

// Command opcodes (LoomUsbCmdEngine).
const _opRead = 0x02;

List<int> _le16(int v) => [v & 0xFF, (v >> 8) & 0xFF];
List<int> _le32(int v) => [
  v & 0xFF,
  (v >> 8) & 0xFF,
  (v >> 16) & 0xFF,
  (v >> 24) & 0xFF,
];

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

// A token packet's payload byte 0 = addr[6:0] | endp[0]<<7. The engine only
// decodes endp[0] (bit 7), so byte 0 = endp<<7 selects EP0 (0x00) or EP1
// (0x80). Byte 1's crc5 is not checked by the (trusted-framing) engine.
List<int> _tokenPayload(int endp) => [(endp & 1) << 7, 0x00];

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

/// A host-side driver: the proven harbor packet/PHY model, crossed onto the
/// device's dp/dm, with hostSend / hostExpectData helpers.
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
  );

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

  // Send a TOKEN packet (PID + a 2-byte addr/endp payload, no CRC framing -
  // the engine ignores addr/CRC5 and only reads endp[0] from byte 0).
  Future<void> token(int pid, {int endp = 0}) =>
      send(pid: pid, isData: true, payload: _tokenPayload(endp));

  // A bare handshake (ACK/NAK): PID only.
  Future<void> handshake(int pid) => send(pid: pid, isData: false);

  // Wait for and capture ONE device packet. Returns {pid, bytes} where bytes
  // excludes the 2 trailing CRC16 bytes (for DATA packets).
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

  // SETUP stage: SETUP token + DATA0(8 bytes), capture the device ACK.
  Future<Map<String, dynamic>> setupStage(List<int> bytes) async {
    await token(_pidSetup);
    await send(pid: _pidData0, isData: true, payload: bytes);
    return expectPkt();
  }

  // GET_DESCRIPTOR(type, index, wLength) -> the returned descriptor bytes.
  Future<List<int>> getDescriptor(int type, int index, int wLength) async {
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
    final ack = await setupStage(setup);
    expect(
      ack['pid'],
      equals(_pidAck),
      reason: 'device ACKs GET_DESCRIPTOR(type=$type,index=$index) SETUP',
    );
    final collected = <int>[];
    // Pull IN-data chunks until a short/zero chunk ends the data stage.
    var toggleData1 = true;
    while (true) {
      await token(_pidIn);
      final d = await expectPkt();
      expect(
        d['pid'],
        equals(toggleData1 ? _pidData1 : _pidData0),
        reason: 'IN-data chunk toggles DATA1/DATA0',
      );
      final chunk = d['bytes'] as List<int>;
      collected.addAll(chunk);
      await handshake(_pidAck);
      toggleData1 = !toggleData1;
      if (chunk.length < 64) break; // short chunk ends the data stage
    }
    // OUT status: OUT token + zero-length DATA1. Device ACKs.
    await token(_pidOut);
    await send(pid: _pidData1, isData: true, payload: const []);
    final st = await expectPkt();
    expect(st['pid'], equals(_pidAck), reason: 'device ACKs the OUT status');
    return collected;
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

  // Host line -> device pads.
  top.input('dp').srcConnection! <= hphyTx.output('dp_out');
  top.input('dm').srcConnection! <= hphyTx.output('dm_out');

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
  );
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('Consolidated USB (harbor engine + loom vendor + bulk) raw dp/dm', () {
    test(
      'FULL enumeration: device + config + ALL THREE string descriptors '
      'come back byte-exact (regression for the alternating empty-string bug)',
      () async {
        final top = _DeviceTop(name: 'enum_top');
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

        // DEVICE descriptor (18 bytes), vendor class 0xFF.
        final dev = await host.getDescriptor(0x01, 0, 18);
        expect(dev.length, equals(18), reason: 'device descriptor 18 bytes');
        expect(dev[4], equals(0xFF), reason: 'bDeviceClass vendor-specific');
        expect(dev[8] | (dev[9] << 8), equals(0x1209), reason: 'idVendor');
        expect(dev[10] | (dev[11] << 8), equals(0x10C0), reason: 'idProduct');
        // iManufacturer=1, iProduct=2, iInterface lives in the config tree.
        expect(dev[14], equals(1), reason: 'iManufacturer index 1');
        expect(dev[15], equals(2), reason: 'iProduct index 2');

        // CONFIGURATION tree (32 bytes): vendor interface + 2 bulk endpoints.
        final cfg = await host.getDescriptor(0x02, 0, 32);
        expect(cfg.length, equals(32), reason: 'config tree 32 bytes');
        expect(
          cfg[0 + 9 + 5],
          equals(0xFF),
          reason: 'bInterfaceClass vendor-specific',
        );
        // EP1 OUT at offset 18, EP1 IN at offset 25.
        expect(cfg[18 + 2], equals(0x01), reason: 'EP1 OUT address 0x01');
        expect(cfg[18 + 3], equals(0x02), reason: 'EP1 OUT bulk');
        expect(cfg[25 + 2], equals(0x81), reason: 'EP1 IN address 0x81');
        expect(cfg[25 + 3], equals(0x02), reason: 'EP1 IN bulk');
        final iInterface = cfg[9 + 8]; // interface descriptor iInterface field
        expect(iInterface, equals(3), reason: 'iInterface index 3');

        // STRING 0 (LANGID).
        final s0 = await host.getDescriptor(0x03, 0, 4);
        expect(s0, equals([4, 0x03, 0x09, 0x04]), reason: 'LANGID en-US');

        // The three meaningful strings (iManufacturer, iProduct, iInterface).
        final s1 = await host.getDescriptor(0x03, 1, 64);
        expect(
          s1,
          equals(_expectedString('Midstall')),
          reason: 'iManufacturer string index 1 == "Midstall" (not empty!)',
        );

        final s2 = await host.getDescriptor(0x03, 2, 64);
        expect(
          s2,
          equals(_expectedString('Loom')),
          reason: 'iProduct string index 2 == "Loom"',
        );

        final s3 = await host.getDescriptor(0x03, 3, 64);
        expect(
          s3,
          equals(_expectedString('Loom Command Interface')),
          reason:
              'iInterface string index 3 == "Loom Command Interface" '
              '(not empty!)',
        );

        await Simulator.endSimulation();
      },
    );

    test('FULL bulk OUT + bulk IN round-trip: bulk OUT a READ-VERSION command, '
        'then bulk IN returns 0x4C4F4F4D ("LOOM") over the wire', () async {
      final top = _DeviceTop(name: 'rt_top');
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

      // the READ-VERSION command (opcode 0x02, addr 0x000, len 4). The device
      // streams the 7 bytes into the command engine and ACKs the DATA packet.
      final readCmd = [_opRead, ..._le32(0x000), ..._le16(4)];
      expect(readCmd.length, equals(7));
      await host.token(_pidOut, endp: 1);
      await host.send(pid: _pidData0, isData: true, payload: readCmd);
      final outAck = await host.expectPkt();
      expect(
        outAck['pid'],
        equals(_pidAck),
        reason: 'device ACKs the bulk OUT DATA packet',
      );

      // VERSION word (0x4C4F4F4D). A bulk IN must return ALL available response
      // bytes in ONE DATA packet (up to wMaxPacketSize=64); the device may NAK
      // a few times while it does the Wishbone read. Retry those. Expect LE
      // 0x4D,0x4F,0x4F,0x4C in one packet.
      final resp = <int>[];
      var guard = 0;
      while (resp.length < 4 && guard < 200) {
        guard++;
        await host.token(_pidIn, endp: 1);
        final d = await host.expectPkt();
        if (d['pid'] == _pidNak) {
          // No response byte ready yet (engine still issuing the bus read):
          // a NAK is the correct bulk-IN answer. Retry.
          continue;
        }
        expect(
          d['pid'],
          anyOf(equals(_pidData0), equals(_pidData1)),
          reason: 'bulk IN DATA packet (toggling)',
        );
        final chunk = d['bytes'] as List<int>;
        // The whole response is packed into one DATA packet: the first non-NAK
        // packet carries all 4 VERSION bytes.
        expect(
          chunk.length,
          equals(4),
          reason:
              'the bulk IN DATA packet must carry the FULL 4-byte response '
              'in one packet (not a 1-byte short packet). Got '
              '${chunk.length}: $chunk',
        );
        resp.addAll(chunk);
        await host.handshake(_pidAck);
      }
      expect(
        resp,
        equals(_le32(0x4C4F4F4D)),
        reason:
            'bulk IN returns the VERSION magic 0x4C4F4F4D byte-exact, '
            'proving the OUT->cmd-engine->Wishbone-read->IN round-trip works '
            'over raw dp/dm',
      );

      await Simulator.endSimulation();
    });

    test(
      'bulk IN with NO command in flight NAKs (correct bulk-IN behavior)',
      () async {
        final top = _DeviceTop(name: 'nak_top');
        final clk = SimpleClockGenerator(10).clk;
        final reset = Logic(name: 'reset');
        top.port('clk').getsLogic(clk);
        top.input('reset').srcConnection! <= reset;
        final host = await _buildHost(top, clk, reset);

        reset.inject(1);
        Simulator.setMaxSimTime(200000000);
        unawaited(Simulator.run());
        await clk.nextPosedge;
        await clk.nextPosedge;
        reset.inject(0);
        for (var i = 0; i < 20; i++) {
          await clk.nextPosedge;
        }

        // No command issued, so the command engine has no response byte: an EP1
        // IN token must be NAKed, not answered with data and not dropped.
        await host.token(_pidIn, endp: 1);
        final d = await host.expectPkt();
        expect(
          d['pid'],
          equals(_pidNak),
          reason: 'device NAKs a bulk IN when no response byte is ready',
        );

        await Simulator.endSimulation();
      },
    );
  });
}
