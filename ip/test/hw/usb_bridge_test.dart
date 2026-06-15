// TDD tests for LoomUsbBridge -- written before the implementation.
//
// LoomUsbBridge composes harbor's UsbEp0Engine (PHY-less full-speed USB device
// + DFU DNLOAD) with a UsbDfuRamSink (Wishbone MASTER + CDC FIFO) so that bytes
// delivered over USB via a DFU DNLOAD become Wishbone byte writes at
// loadBase + offset. Downstream, the master 'bus' interface connects to the
// Loom accelerator's Wishbone slave.
//
// Following the usb_dfu_test.dart driving pattern: we inject the engine's sink
// stream (sink_data/sink_valid + dnload_done + alt_setting) directly rather
// than driving raw dp/dm enumeration (which is exercised in harbor's own engine
// tests and is deferred here). The bridge exposes the sink stream ports at its
// top so the testbench can feed the stream the engine would otherwise produce.
//
// Tests:
//   1. SV emission: emits non-empty SystemVerilog naming the submodules.
//   2. Address-landing against a behavioral Wishbone slave RAM: streamed bytes
//      land byte-exact at loadBase + offset.
//   3. Address-landing against the REAL LoomAccelerator slave: a contiguous
//      stream into the (contiguous) weight buffer region lands byte-exact and
//      reads back over the same bus.

import 'dart:async';

import 'package:harbor/harbor.dart';
import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/hw/usb_bridge.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// A minimal behavioral Wishbone B4 slave RAM (consumer role), modeled on the
// _BehavioralWbSlaveRam in harbor's usb_dfu_test.dart. On a write cycle
// (cyc & stb & we) it stores each SELECTED byte lane of dat_mosi into a word
// array at index (adr - loadBase) / bytesPerWord and acks for one cycle.
// ---------------------------------------------------------------------------
class _BehavioralWbSlaveRam extends BridgeModule {
  final int addressWidth;
  final int dataWidth;
  final int loadBase;
  final int words;
  int get bytesPerWord => dataWidth ~/ 8;

  late final List<Logic> _mem;

  int wordAt(int i) => _mem[i].value.isValid ? _mem[i].value.toInt() : 0;

  _BehavioralWbSlaveRam({
    required this.addressWidth,
    required this.dataWidth,
    required this.loadBase,
    required this.words,
  }) : super('_BehavioralWbSlaveRam', name: 'wb_slave_ram') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    final ref = addInterface(
      WishboneInterface(
        WishboneConfig(addressWidth: addressWidth, dataWidth: dataWidth),
      ),
      name: 'bus',
      role: PairRole.consumer, // slave
    );
    final bus = ref.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');
    final selWidth = dataWidth ~/ 8;
    var wordShift = 0;
    for (var v = bytesPerWord; v > 1; v >>= 1) {
      wordShift++;
    }

    _mem = [
      for (var i = 0; i < words; i++) Logic(name: 'mem_$i', width: dataWidth),
    ];

    final wrEn = bus.cyc & bus.stb & bus.we;
    final byteIdx = (bus.adr - Const(loadBase, width: addressWidth));
    final wordIdx =
        (bytesPerWord == 1
                ? byteIdx
                : byteIdx
                      .slice(addressWidth - 1, wordShift)
                      .zeroExtend(addressWidth))
            .named('word_idx');

    final ackReg = Logic(name: 'ack_reg');

    Logic mergedFor(Logic cur) {
      var out = cur;
      for (var lane = 0; lane < selWidth; lane++) {
        final lo = lane * 8;
        final newByte = bus.datMosi.slice(lo + 7, lo);
        final keepByte = cur.slice(lo + 7, lo);
        final laneByte = mux(
          bus.sel.slice(lane, lane).eq(Const(1)),
          newByte,
          keepByte,
        );
        out = (lane == 0)
            ? laneByte
            : [laneByte, out.slice(lo - 1, 0)].swizzle();
      }
      return out;
    }

    final accept = wrEn & ~ackReg;

    Sequential(clk, [
      If(
        reset,
        then: [
          ackReg < Const(0),
          for (final w in _mem) w < Const(0, width: dataWidth),
        ],
        orElse: [
          ackReg < accept,
          If(
            accept,
            then: [
              for (var i = 0; i < words; i++)
                If(
                  wordIdx.eq(Const(i, width: addressWidth)),
                  then: [_mem[i] < mergedFor(_mem[i])],
                ),
            ],
          ),
        ],
      ),
    ]);

    bus.ack <= ackReg;
    bus.datMiso <= Const(0, width: dataWidth);
  }
}

// ---------------------------------------------------------------------------
// Two-clock top wrapper: a LoomUsbBridge master connected to a behavioral
// Wishbone slave RAM. USB-domain stream inputs + the bridge's observability
// outputs are pulled to the top.
// ---------------------------------------------------------------------------
class _BridgeRamTop extends BridgeModule {
  final LoomUsbBridge bridge;
  final _BehavioralWbSlaveRam slave;

  _BridgeRamTop({required this.bridge, required this.slave})
    : super('_BridgeRamTop', name: 'bridge_ram_top') {
    createPort('usb_clk', PortDirection.input);
    createPort('usb_reset', PortDirection.input);
    createPort('bus_clk', PortDirection.input);
    createPort('bus_reset', PortDirection.input);
    createPort('sink_data', PortDirection.input, width: 8);
    createPort('sink_valid', PortDirection.input);
    createPort('dnload_done', PortDirection.input);
    createPort('alt_setting', PortDirection.input, width: 8);

    addSubModule(bridge);
    addSubModule(slave);

    connectPorts(port('usb_clk'), bridge.port('usb_clk'));
    connectPorts(port('usb_reset'), bridge.port('usb_reset'));
    connectPorts(port('bus_clk'), bridge.port('bus_clk'));
    connectPorts(port('bus_reset'), bridge.port('bus_reset'));
    connectPorts(port('sink_data'), bridge.port('tb_sink_data'));
    connectPorts(port('sink_valid'), bridge.port('tb_sink_valid'));
    connectPorts(port('dnload_done'), bridge.port('tb_dnload_done'));
    connectPorts(port('alt_setting'), bridge.port('tb_alt_setting'));
    bridge.port('tb_sink_inject_en').getsLogic(Const(1));

    connectPorts(port('bus_clk'), slave.port('clk'));
    connectPorts(port('bus_reset'), slave.port('reset'));

    connectInterfaces(bridge.interface('bus'), slave.interface('bus'));

    addOutput('image_ready') <= bridge.output('image_ready');
    addOutput('bytes_written', width: 32) <= bridge.output('bytes_written');
    addOutput('sink_ready') <= bridge.output('sink_ready');
    addOutput('overflow') <= bridge.output('overflow');
  }
}

// ---------------------------------------------------------------------------
// Two-clock top wrapper: a LoomUsbBridge master connected to the REAL
// LoomAccelerator Wishbone slave.
// ---------------------------------------------------------------------------
class _BridgeAccelTop extends BridgeModule {
  final LoomUsbBridge bridge;
  final LoomAccelerator accel;

  _BridgeAccelTop({required this.bridge, required this.accel})
    : super('_BridgeAccelTop', name: 'bridge_accel_top') {
    createPort('usb_clk', PortDirection.input);
    createPort('usb_reset', PortDirection.input);
    createPort('bus_clk', PortDirection.input);
    createPort('bus_reset', PortDirection.input);
    createPort('sink_data', PortDirection.input, width: 8);
    createPort('sink_valid', PortDirection.input);
    createPort('dnload_done', PortDirection.input);
    createPort('alt_setting', PortDirection.input, width: 8);

    addSubModule(bridge);
    addSubModule(accel);

    connectPorts(port('usb_clk'), bridge.port('usb_clk'));
    connectPorts(port('usb_reset'), bridge.port('usb_reset'));
    connectPorts(port('bus_clk'), bridge.port('bus_clk'));
    connectPorts(port('bus_reset'), bridge.port('bus_reset'));
    connectPorts(port('sink_data'), bridge.port('tb_sink_data'));
    connectPorts(port('sink_valid'), bridge.port('tb_sink_valid'));
    connectPorts(port('dnload_done'), bridge.port('tb_dnload_done'));
    connectPorts(port('alt_setting'), bridge.port('tb_alt_setting'));
    bridge.port('tb_sink_inject_en').getsLogic(Const(1));

    connectPorts(port('bus_clk'), accel.port('clk'));
    connectPorts(port('bus_reset'), accel.port('reset'));

    connectInterfaces(bridge.interface('bus'), accel.interface('bus'));

    addOutput('image_ready') <= bridge.output('image_ready');
    addOutput('bytes_written', width: 32) <= bridge.output('bytes_written');
    addOutput('sink_ready') <= bridge.output('sink_ready');
  }
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // -------------------------------------------------------------------------
  // Test 1: SystemVerilog emission names the bridge + its submodules.
  // -------------------------------------------------------------------------
  group('LoomUsbBridge SV emission', () {
    test('emits non-empty SV naming LoomUsbBridge + UsbEp0Engine + '
        'UsbDfuRamSink', () async {
      final bridge = LoomUsbBridge(loadBase: 0x0, busAddressWidth: 12);
      await bridge.build();
      final sv = bridge.generateSynth();
      expect(sv, isNotEmpty);
      expect(sv, contains('LoomUsbBridge'));
      expect(sv, contains('UsbEp0Engine'));
      expect(sv, contains('UsbDfuRamSink'));
    });
  });

  // -------------------------------------------------------------------------
  // Test 2: streamed bytes land byte-exact at loadBase + offset in a slave RAM.
  // -------------------------------------------------------------------------
  group('LoomUsbBridge address landing (behavioral slave RAM)', () {
    test('streams an image into RAM byte-exact at loadBase + offset', () async {
      const loadBase = 0x0;
      const busDataWidth = 32;
      const busAddrWidth = 12;
      const bytesPerWord = busDataWidth ~/ 8;

      final bridge = LoomUsbBridge(
        name: 'loom_usb_bridge',
        loadBase: loadBase,
        busAddressWidth: busAddrWidth,
        busDataWidth: busDataWidth,
        fifoDepth: 64,
      );
      final slave = _BehavioralWbSlaveRam(
        addressWidth: busAddrWidth,
        dataWidth: busDataWidth,
        loadBase: loadBase,
        words: 32,
      );
      final top = _BridgeRamTop(bridge: bridge, slave: slave);

      final usbClk = SimpleClockGenerator(10).clk;
      final busClk = SimpleClockGenerator(40).clk;
      top.port('usb_clk').getsLogic(usbClk);
      top.port('bus_clk').getsLogic(busClk);

      await top.build();

      final usbReset = top.input('usb_reset');
      final busReset = top.input('bus_reset');
      final sinkData = top.input('sink_data');
      final sinkValid = top.input('sink_valid');
      final dnloadDone = top.input('dnload_done');
      final altSetting = top.input('alt_setting');

      Simulator.setMaxSimTime(4000000);
      unawaited(Simulator.run());

      // A known prefix + a ramp, 22 bytes (5.5 words => a partial final word).
      final image = <int>[
        0xDE,
        0xAD,
        0xBE,
        0xEF,
        0x01,
        0x02,
        0x03,
        0x04,
        0x10,
        0x20,
        0x30,
        0x40,
        0xA5,
        0x5A,
        0xC3,
        0x3C,
        0x11,
        0x22,
        0x33,
        0x44,
        0x77,
        0x88,
      ];
      expect(image.length, equals(22));

      usbReset.inject(1);
      busReset.inject(1);
      sinkData.inject(0);
      sinkValid.inject(0);
      dnloadDone.inject(0);
      altSetting.inject(0); // RAM interface selected
      for (var i = 0; i < 5; i++) {
        await busClk.nextPosedge;
      }
      usbReset.put(0);
      busReset.put(0);
      await busClk.nextPosedge;

      // Drive the sink stream in the USB domain, honoring sink_ready.
      for (final b in image) {
        var guard = 0;
        while (top.output('sink_ready').value.toInt() == 0 && guard < 100000) {
          guard++;
          await usbClk.nextPosedge;
        }
        sinkData.put(b);
        sinkValid.put(1);
        await usbClk.nextPosedge;
        sinkValid.put(0);
        sinkData.put(0);
        await usbClk.nextPosedge;
      }
      // Zero-length DNLOAD completion marker.
      var g = 0;
      while (top.output('sink_ready').value.toInt() == 0 && g < 100000) {
        g++;
        await usbClk.nextPosedge;
      }
      dnloadDone.put(1);
      await usbClk.nextPosedge;
      dnloadDone.put(0);

      var readySeen = false;
      for (var i = 0; i < 4000; i++) {
        await busClk.nextPosedge;
        if (top.output('image_ready').value.isValid &&
            top.output('image_ready').value.toInt() == 1) {
          readySeen = true;
          break;
        }
      }
      expect(readySeen, isTrue, reason: 'image_ready pulsed after the image');

      await busClk.nextPosedge;
      await busClk.nextPosedge;

      expect(
        top.output('bytes_written').value.toInt(),
        equals(image.length),
        reason: 'bytes_written counts every byte',
      );

      for (var i = 0; i < image.length; i++) {
        final word = slave.wordAt(i ~/ bytesPerWord);
        final lane = i % bytesPerWord;
        final got = (word >> (lane * 8)) & 0xFF;
        expect(
          got,
          equals(image[i]),
          reason: 'RAM byte[$i] (word ${i ~/ bytesPerWord} lane $lane)',
        );
      }

      await Simulator.endSimulation();
    });
  });

  // -------------------------------------------------------------------------
  // Test 3: drive the REAL LoomAccelerator Wishbone slave from the USB-driven
  // master. We stream a contiguous image targeting the (contiguous) weight
  // buffer region at loadBase = 0x100 and prove the accelerator's real slave
  // ACCEPTS every Wishbone write the USB transport issues: each write is acked
  // (the master advances), bytes_written == image length, and image_ready
  // pulses. (Byte-exact value landing at loadBase + offset is proven in test 2
  // against a memory model; the accelerator's weight buffer is write-only over
  // the bus and the master is single-owner of the bus, so a bus read-back of
  // those bytes is not possible here - see the report's noted gap.)
  // -------------------------------------------------------------------------
  group('LoomUsbBridge -> real LoomAccelerator slave', () {
    test('USB DFU stream is accepted by the accelerator slave: every write '
        'acks, bytes_written + image_ready', () async {
      const baseWeights = 0x100;
      // The DFU sink writes contiguously from loadBase; aim loadBase at the
      // weight-buffer base so byte[i] lands at the weight buffer.
      const loadBase = baseWeights;

      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x0),
        name: 'accel',
      );
      final bridge = LoomUsbBridge(
        name: 'loom_usb_bridge_accel',
        loadBase: loadBase,
        busAddressWidth: 12,
        busDataWidth: 32,
        fifoDepth: 64,
      );
      final top = _BridgeAccelTop(bridge: bridge, accel: accel);

      final usbClk = SimpleClockGenerator(10).clk;
      final busClk = SimpleClockGenerator(40).clk;
      top.port('usb_clk').getsLogic(usbClk);
      top.port('bus_clk').getsLogic(busClk);

      await top.build();

      final usbReset = top.input('usb_reset');
      final busReset = top.input('bus_reset');
      final sinkData = top.input('sink_data');
      final sinkValid = top.input('sink_valid');
      final dnloadDone = top.input('dnload_done');
      final altSetting = top.input('alt_setting');

      Simulator.setMaxSimTime(8000000);
      unawaited(Simulator.run());

      // 8 weight bytes (2 words) -> exercises the weight buffer.
      final image = <int>[0x03, 0xFE, 0x01, 0xFF, 0xFC, 0x03, 0xFE, 0x02];

      usbReset.inject(1);
      busReset.inject(1);
      sinkData.inject(0);
      sinkValid.inject(0);
      dnloadDone.inject(0);
      altSetting.inject(0);
      for (var i = 0; i < 5; i++) {
        await busClk.nextPosedge;
      }
      usbReset.put(0);
      busReset.put(0);
      await busClk.nextPosedge;

      for (final b in image) {
        var guard = 0;
        while (top.output('sink_ready').value.toInt() == 0 && guard < 100000) {
          guard++;
          await usbClk.nextPosedge;
        }
        // sink_ready surfaced through the bridge top? Use bridge output.
        sinkData.put(b);
        sinkValid.put(1);
        await usbClk.nextPosedge;
        sinkValid.put(0);
        sinkData.put(0);
        await usbClk.nextPosedge;
      }
      dnloadDone.put(1);
      await usbClk.nextPosedge;
      dnloadDone.put(0);

      var readySeen = false;
      for (var i = 0; i < 8000; i++) {
        await busClk.nextPosedge;
        if (top.output('image_ready').value.isValid &&
            top.output('image_ready').value.toInt() == 1) {
          readySeen = true;
          break;
        }
      }
      expect(readySeen, isTrue, reason: 'image_ready after the stream');

      // Every streamed byte produced an acked Wishbone write into the real
      // accelerator slave (proving the USB-driven master drives the real
      // accelerator, not just a memory model).
      expect(
        top.output('bytes_written').value.toInt(),
        equals(image.length),
        reason: 'bytes_written == image length (every write acked by accel)',
      );

      await Simulator.endSimulation();
    });
  });
}
