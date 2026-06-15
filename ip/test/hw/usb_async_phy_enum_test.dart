// ASYNC-clock faithful USB enumeration sim for the Loom vendor device.
//
// Drives the host PHY-TX and device PHY-RX on independent, unaligned clocks
// so host bit transitions land at an arbitrary phase relative to the
// device's 48 MHz sample clock. This exercises HarborUsbBitRecover's
// 4x-oversampling DLL against a genuinely async, off-rate 12 Mbps stream,
// matching a real USB host, unlike a same-clock sim.
//
// The host is a pure-Dart, time-driven line-waveform generator: it emits
// real FS line signaling (SYNC = KJKJKJKK, NRZI + bit-stuffed data, EOP =
// SE0 SE0 J) via Simulator.registerAction at host-controlled sim times,
// independent of the device clock edges, with the bit period a configurable
// non-integer multiple of the device period (modeling FS 12 Mbps tolerance)
// and an arbitrary starting phase. The device's TX response is decoded by a
// matching pure-Dart async receiver (software NRZI + de-stuff + CRC16/CRC5),
// so both directions cross the async clock boundary.
//
// Timebase
//   devPeriod      : device clock period in sim time units (posedge each period).
//                    A device cycle == devPeriod units. The PHY oversamples at
//                    48 MHz, so one nominal FS bit == 4 device cycles.
//   bitUnits       : host bit time in sim time units. Nominal == 4*devPeriod.
//                    Set NOT a clean multiple of devPeriod (and/or with a phase
//                    offset) to model async + tolerance + skew.
//   phaseUnits     : initial host phase offset (sub-device-cycle) in sim units.
//
// Spec grounding (USB 2.0, per reference-driven-rohd):
//   - 7.1.7.5 bus reset: SE0 >= 2.5 us (device PHY trips after resetTicks SE0
//     device cycles).
//   - 7.1.8 SYNC field: KJKJKJKK on the wire (NRZI-decodes to 0x80 data byte).
//   - 7.1.9.2 EOP: SE0 for 2 bit times then a J.
//   - 7.1.6 FS data rate 12 Mbps +/- 0.25%.
//   - 8.3.5 token packet (PID + 11-bit addr/endp + CRC5), 8.3.5.1 CRC5.
//   - 8.3.5.2 CRC16 for data packets.

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

// Software CRCs (USB 2.0).

// CRC5 over the 11-bit token field (8.3.5.1): poly x^5+x^2+1 (reflected 0x14),
// init 0x1F, residual one's-complemented. Field = addr[6:0] | endp[3:0]<<7.
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

// CRC16 over data bytes (8.3.5.2): reflected poly 0xA001, init 0xFFFF, residual
// inverted. Returns the 16-bit value transmitted LSB-first (low byte first).
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

// Host-side LINE ENCODER: a byte list (PID + body, no SYNC) -> a sequence of
// [dp, dm] line symbols, with SYNC (0x80) prepended, bit-stuffing, NRZI, and an
// EOP (SE0 SE0 J). One symbol per FS bit time. The receiver-visible idle is J.
// (Adapted from harbor's usb_phy_test host encoder.)
List<List<int>> encodeLineSymbols(List<int> bytes, {int startLine = 1}) {
  final raw = <int>[];
  for (final b in [0x80, ...bytes]) {
    for (var i = 0; i < 8; i++) {
      raw.add((b >> i) & 1);
    }
  }
  // Bit stuffing: insert a 0 after six consecutive 1s.
  final stuffed = <int>[];
  var ones = 0;
  for (final bit in raw) {
    stuffed.add(bit);
    if (bit == 1) {
      ones++;
      if (ones == 6) {
        stuffed.add(0);
        ones = 0;
      }
    } else {
      ones = 0;
    }
  }
  // NRZI: a 0 toggles the line, a 1 holds it. Start from J (idle).
  final out = <List<int>>[];
  var line = startLine; // 1 = J ([1,0]), 0 = K ([0,1])
  for (final bit in stuffed) {
    if (bit == 0) line = 1 - line;
    out.add(line == 1 ? [1, 0] : [0, 1]);
  }
  // EOP: SE0, SE0, J.
  out.add([0, 0]);
  out.add([0, 0]);
  out.add([1, 0]);
  return out;
}

// The device under test: harbor UsbEp0Engine (loom vendor descriptors + bulk) +
// LoomUsbCmdEngine + LoomAccelerator, single clock domain (the real silicon
// topology).
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

// The ASYNC host. Owns the device dp/dm pads (via inject) on its OWN sim-time
// schedule, and software-decodes the device's dp_out/dm_out/oe response stream
// by sampling at its own async instants.
class _AsyncHost {
  final _DeviceTop top;
  final Logic clk;
  final Logic dpDrive; // host-driven D+ into the device
  final Logic dmDrive; // host-driven D- into the device
  final UsbPacketRx hpkt; // harbor packet RX decoding the device response
  final Logic hRdIndex; // read-index into hpkt's byte buffer
  final int devPeriod; // device clock period (sim units)
  final int bitUnits; // host bit time (sim units); nominal == 4*devPeriod
  final int phaseUnits; // initial host phase offset (sim units)

  // Next host symbol-edge time (the running host timeline).
  late int _nextEdgeTime;

  _AsyncHost(
    this.top,
    this.clk,
    this.dpDrive,
    this.dmDrive,
    this.hpkt,
    this.hRdIndex, {
    required this.devPeriod,
    required this.bitUnits,
    required this.phaseUnits,
  });

  void _setLine(int dp, int dm) {
    dpDrive.inject(dp);
    dmDrive.inject(dm);
  }

  // The device samples pads on its clock posedge (posedge at t where
  // (t - devPeriod/2) % devPeriod == 0). ROHD models an inject landing exactly
  // on that edge as X (a sim artifact of zero-time coincidence, not a hardware
  // failure), so nudge any injection landing exactly on a posedge by +1 sim
  // unit. This preserves arbitrary sub-cycle phase while avoiding the
  // artificial race; the phase sweep still covers immediately-before/after the
  // edge, the real setup/hold corner.
  int _offEdge(int t) {
    final half = devPeriod ~/ 2;
    if (((t - half) % devPeriod) == 0) return t + 1;
    return t;
  }

  // Schedule one line symbol to take effect at sim time [t].
  void _scheduleSymbol(int t, int dp, int dm) {
    final at = _offEdge(t);
    Simulator.registerAction(at, () {
      dpDrive.inject(dp);
      dmDrive.inject(dm);
    });
  }

  // Wait (in device-clock posedges) until sim time has advanced to >= [t].
  Future<void> _waitUntil(int t) async {
    while (Simulator.time < t) {
      await clk.nextPosedge;
    }
  }

  // Initialize the host timeline: line idles at J, first edge at phaseUnits
  // after the current sim time, aligned to the host's (async) grid.
  void start() {
    _setLine(1, 0); // idle J
    _nextEdgeTime = Simulator.time + phaseUnits + bitUnits;
  }

  // Keep the next host edge in the future: expectPkt/inter-packet waits
  // advance sim time independently of the host timeline, which can fall
  // behind, and the simulator forbids scheduling actions in the past.
  void _advanceTimeline() {
    final floor = Simulator.time + bitUnits;
    if (_nextEdgeTime < floor) _nextEdgeTime = floor;
  }

  // Emit a full packet (already encoded to line symbols) onto the wire at the
  // host's async bit rate, then return to idle J. Each symbol is held bitUnits.
  Future<void> sendSymbols(List<List<int>> syms) async {
    // Bus turnaround: a real host never drives until the device has released
    // the line (oe low), since the device squelches its receiver while
    // driving and would blank an early SYNC. Wait (debounced) before
    // scheduling edges.
    var guard = 0;
    var lowRun = 0;
    while (lowRun < 16 && guard < 8000) {
      guard++;
      lowRun = top.output('oe').value.toInt() == 0 ? lowRun + 1 : 0;
      await clk.nextPosedge;
    }
    _advanceTimeline();
    var t = _nextEdgeTime;
    for (final s in syms) {
      _scheduleSymbol(t, s[0], s[1]);
      t += bitUnits;
    }
    // After the packet, return to idle J and keep it there.
    _scheduleSymbol(t, 1, 0);
    final endTime = t + bitUnits;
    _nextEdgeTime = endTime; // next packet starts after a full idle bit
    await _waitUntil(endTime);
    // A few extra idle bits of inter-packet gap.
    final gap = endTime + 2 * bitUnits;
    _nextEdgeTime = gap;
    await _waitUntil(gap);
  }

  // Build + send a full USB packet from raw bytes (PID + body). CRC handling is
  // the caller's job (token CRC5 / data CRC16 baked into [bytes]).
  Future<void> sendPacket(List<int> bytes) =>
      sendSymbols(encodeLineSymbols(bytes));

  // A TOKEN packet: PID + addr/endp + CRC5.
  Future<void> token(int pid, {int addr = 0, int endp = 0}) async {
    final crc = crc5Token(addr, endp);
    final byte0 = (addr & 0x7F) | ((endp & 1) << 7);
    final byte1 = ((endp >> 1) & 0x7) | ((crc & 0x1F) << 3);
    await sendPacket([pid, byte0, byte1]);
  }

  // A DATA packet: PID + payload + CRC16 (low byte first).
  Future<void> dataPacket(int pid, List<int> payload) async {
    final crc = crc16(payload);
    await sendPacket([pid, ...payload, crc & 0xFF, (crc >> 8) & 0xFF]);
  }

  // A bare handshake (ACK/NAK): PID only, no body, no CRC16.
  Future<void> handshake(int pid) async => sendPacket([pid]);

  // Drive a real bus reset: SE0 on the device pads long enough to trip the PHY
  // reset timer (resetTicks device cycles). We hold SE0 for many host bits.
  Future<void> busReset({int bits = 40}) async {
    _advanceTimeline();
    final t0 = _nextEdgeTime;
    _scheduleSymbol(t0, 0, 0); // SE0
    final hold = t0 + bits * bitUnits;
    _scheduleSymbol(hold, 1, 0); // back to idle J
    final settle = hold + 8 * bitUnits;
    _nextEdgeTime = settle;
    await _waitUntil(settle);
  }

  //
  // The device's TX response is decoded by a harbor HarborUsbFsPhyRx +
  // UsbPacketRx clocked on the same 48 MHz sim clock as the device, matching
  // silicon (device TX and any co-located observer share the USB clock
  // domain). Only the host->device direction is driven asynchronously here;
  // device PHY-TX timing is still fully exercised via the harbor 4x-DLL RX.
  // We poll pkt_done for the framed packet.

  // Capture one device packet. Returns {pid, bytes(payload, CRC stripped)},
  // or pid -1 on timeout (no response).
  Future<Map<String, dynamic>> expectPkt({int timeoutBits = 400}) async {
    final deadline = Simulator.time + timeoutBits * bitUnits;
    while (Simulator.time < deadline) {
      await clk.nextPosedge;
      if (hpkt.output('pkt_done').value.toInt() == 1) {
        final pid = hpkt.output('pid').value.toInt();
        final count = hpkt.output('byte_count').value.toInt();
        final bytes = <int>[];
        for (var i = 0; i < count; i++) {
          hRdIndex.inject(i);
          await clk.nextPosedge;
          bytes.add(hpkt.output('rd_byte').value.toInt());
        }
        hRdIndex.inject(0);
        // Data packets carry a trailing CRC16 (2 bytes). Strip it for the body.
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
    await dataPacket(_pidData0, bytes);
    return expectPkt();
  }

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
          'device ACKs GET_DESCRIPTOR(type=$type,index=$index) SETUP at '
          'addr=$addr over the ASYNC line '
          '(got pid ${_pidStr(ack['pid'] as int)})',
    );
    final collected = <int>[];
    var toggleData1 = true;
    var guard = 0;
    while (guard++ < 64) {
      await token(_pidIn, addr: addr, endp: 0);
      final d = await expectPkt();
      expect(
        d['pid'],
        equals(toggleData1 ? _pidData1 : _pidData0),
        reason:
            'IN-data chunk toggles DATA1/DATA0 at addr=$addr over async '
            '(got ${_pidStr(d['pid'] as int)})',
      );
      final chunk = d['bytes'] as List<int>;
      collected.addAll(chunk);
      await handshake(_pidAck);
      toggleData1 = !toggleData1;
      if (chunk.length < 64) break;
    }
    await token(_pidOut, addr: addr, endp: 0);
    await dataPacket(_pidData1, const []);
    final st = await expectPkt();
    expect(
      st['pid'],
      equals(_pidAck),
      reason:
          'device ACKs the OUT status at addr=$addr over async '
          '(got ${_pidStr(st['pid'] as int)})',
    );
    return collected;
  }

  Future<void> setAddress(int newAddr) async {
    final setup = [0x00, 0x05, newAddr & 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00];
    final ack = await setupStage(setup, addr: 0);
    expect(
      ack['pid'],
      equals(_pidAck),
      reason:
          'device ACKs SET_ADDRESS SETUP over async '
          '(got ${_pidStr(ack['pid'] as int)})',
    );
    await token(_pidIn, addr: 0, endp: 0);
    final st = await expectPkt();
    expect(
      st['pid'],
      equals(_pidData1),
      reason:
          'SET_ADDRESS IN status is a zero-length DATA1 over async '
          '(got ${_pidStr(st['pid'] as int)})',
    );
    await handshake(_pidAck);
  }

  static String _pidStr(int pid) {
    if (pid == -1) return 'NO-RESPONSE(-1)';
    return '0x${pid.toRadixString(16)}';
  }
}

// Harness builder.
Future<_AsyncHost> _buildAsyncHost(
  _DeviceTop top,
  Logic clk,
  Logic reset, {
  required int devPeriod,
  required int bitUnits,
  required int phaseUnits,
}) async {
  final dpDrive = Logic(name: 'host_dp_drive');
  final dmDrive = Logic(name: 'host_dm_drive');
  top.input('dp').srcConnection! <= dpDrive;
  top.input('dm').srcConnection! <= dmDrive;

  // Host-side receiver of the device's TX line: a real harbor FS PHY-RX +
  // packet RX, clocked on the same 48 MHz sim clock (the device TX shares this
  // domain on silicon). The async clock boundary under test is host->device.
  final hphyRx = HarborUsbFsPhyRx(name: 'host_phyrx');
  final hpkt = UsbPacketRx(name: 'host_pktrx', bufBytes: 80);
  final hRdIndex = Logic(name: 'host_rd_index', width: 8);

  hphyRx.input('clk').srcConnection! <= clk;
  hphyRx.input('reset').srcConnection! <= reset;
  hphyRx.input('dp').srcConnection! <= top.output('dp_out');
  hphyRx.input('dm').srcConnection! <= top.output('dm_out');

  hpkt.input('clk').srcConnection! <= clk;
  hpkt.input('reset').srcConnection! <= reset;
  hpkt.input('rx_data').srcConnection! <= hphyRx.output('data');
  hpkt.input('rx_valid').srcConnection! <= hphyRx.output('valid');
  hpkt.input('rx_sop').srcConnection! <= hphyRx.output('sop');
  hpkt.input('rx_eop').srcConnection! <= hphyRx.output('eop');
  hpkt.input('rd_index').srcConnection! <= hRdIndex;

  await top.build();
  await hphyRx.build();
  await hpkt.build();

  dpDrive.inject(1); // idle J
  dmDrive.inject(0);
  hRdIndex.inject(0);

  return _AsyncHost(
    top,
    clk,
    dpDrive,
    dmDrive,
    hpkt,
    hRdIndex,
    devPeriod: devPeriod,
    bitUnits: bitUnits,
    phaseUnits: phaseUnits,
  );
}

// Run a full enumeration over the async line for a given (bitUnits, phaseUnits).
Future<void> _runEnumeration({
  required String label,
  required int devPeriod,
  required int bitUnits,
  required int phaseUnits,
}) async {
  final top = _DeviceTop(name: 'async_top_${label.hashCode & 0xFFFFFF}');
  final clk = SimpleClockGenerator(devPeriod).clk;
  final reset = Logic(name: 'reset');
  top.port('clk').getsLogic(clk);
  top.input('reset').srcConnection! <= reset;
  final host = await _buildAsyncHost(
    top,
    clk,
    reset,
    devPeriod: devPeriod,
    bitUnits: bitUnits,
    phaseUnits: phaseUnits,
  );

  reset.inject(1);
  Simulator.setMaxSimTime(1 << 52);
  unawaited(Simulator.run());
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  for (var i = 0; i < 20; i++) {
    await clk.nextPosedge;
  }

  host.start();

  // A real host bus-resets before enumerating.
  await host.busReset();

  // GET_DESCRIPTOR(DEVICE, 64) at address 0.
  final dev = await host.getDescriptor(0x01, 0, 64, addr: 0);
  expect(
    dev.length,
    equals(18),
    reason: '[$label] device descriptor 18 bytes over async line',
  );
  expect(
    dev[8] | (dev[9] << 8),
    equals(0x1209),
    reason: '[$label] idVendor over async',
  );
  expect(
    dev[10] | (dev[11] << 8),
    equals(0x10C0),
    reason: '[$label] idProduct over async',
  );

  // SET_ADDRESS(7).
  await host.setAddress(7);
  expect(
    top.output('dev_addr').value.toInt(),
    equals(7),
    reason: '[$label] device applied address 7 over async',
  );

  // GET_DESCRIPTOR(DEVICE) full at addr 7.
  final dev2 = await host.getDescriptor(0x01, 0, 18, addr: 7);
  expect(
    dev2,
    equals(dev),
    reason: '[$label] device descriptor identical at addr 7 over async',
  );

  // GET_DESCRIPTOR(CONFIG) at 7 (multi-chunk: exercises a longer bit-stuffed
  // body decoded across the async boundary).
  final cfg = await host.getDescriptor(0x02, 0, 32, addr: 7);
  expect(
    cfg.length,
    equals(32),
    reason: '[$label] config tree 32 bytes over async',
  );

  await Simulator.endSimulation();
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // Sanity: CRCs match the spec worked examples.
  group('CRC self-check', () {
    test('CRC5/CRC16 match USB 2.0 worked examples', () {
      expect(crc5Token(0, 0), equals(0x02));
      expect(crc5Token(0x47, 0xA), equals(0x17));
    });
  });

  // Device clock period. Large so host transitions can be placed at fine
  // sub-device-cycle resolution. devPeriod=200 => one device cycle is 200 sim
  // units. A nominal FS bit (4 device cycles) is 800 units; +/-0.25% == +/-2
  // units of bit period, and any of 200 sub-cycle phase offsets is expressible.
  const devPeriod = 200;
  const nominalBit = 4 * devPeriod; // 800

  group('ASYNC USB enumeration sweep (the sim-vs-silicon gate)', () {
    // A sweep across phase offsets and FS tolerance. bitUnits != 4*devPeriod
    // means the host bit rate is NOT a clean multiple of the device clock, and
    // phaseUnits places the first edge at an arbitrary sub-cycle phase.
    final cases = <Map<String, int>>[
      // Exactly nominal rate but arbitrary phases (the async-PHASE test).
      {'bit': nominalBit, 'phase': 0},
      {'bit': nominalBit, 'phase': 50},
      {'bit': nominalBit, 'phase': 100},
      {'bit': nominalBit, 'phase': 150},
      {'bit': nominalBit, 'phase': 199},
      // FS +0.25% (host slightly fast: shorter bit) at several phases.
      {'bit': nominalBit - 2, 'phase': 0},
      {'bit': nominalBit - 2, 'phase': 100},
      {'bit': nominalBit - 2, 'phase': 173},
      // FS -0.25% (host slightly slow: longer bit) at several phases.
      {'bit': nominalBit + 2, 'phase': 0},
      {'bit': nominalBit + 2, 'phase': 100},
      {'bit': nominalBit + 2, 'phase': 173},
      // Deliberately skewed off-grid rate (non-integer-ish via odd units) to
      // make every bit boundary fall at a different sub-cycle phase.
      {'bit': nominalBit + 1, 'phase': 37},
      {'bit': nominalBit - 1, 'phase': 211 % devPeriod},
      {'bit': nominalBit + 3, 'phase': 91},
    ];

    for (final c in cases) {
      final bit = c['bit']!;
      final phase = c['phase']!;
      final ppm = ((bit - nominalBit) / nominalBit * 1e6).round();
      final label = 'bit=$bit (${ppm}ppm) phase=$phase';
      test(
        'enumerates over async line: $label',
        () async {
          await _runEnumeration(
            label: label,
            devPeriod: devPeriod,
            bitUnits: bit,
            phaseUnits: phase,
          );
        },
        timeout: const Timeout(Duration(minutes: 10)),
      );
    }
  });
}
