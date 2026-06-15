import 'dart:async';

import 'package:loom/src/hw/controller.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // -------------------------------------------------------------------------
  // Configuration tests
  // -------------------------------------------------------------------------

  group('LoomControllerConfig', () {
    test('default values', () {
      const cfg = LoomControllerConfig();
      expect(cfg.addressWidth, equals(8));
      expect(cfg.dataWidth, equals(32));
      expect(cfg.numScratch, equals(1));
      expect(cfg.versionMagic, equals(0x4C4F4F4D));
    });

    test('validate passes on valid config', () {
      expect(() => const LoomControllerConfig().validate(), returnsNormally);
    });

    test('validate rejects dataWidth != 32', () {
      expect(
        () => const LoomControllerConfig(dataWidth: 8).validate(),
        throwsArgumentError,
      );
    });

    test('validate rejects addressWidth < 3', () {
      expect(
        () => const LoomControllerConfig(addressWidth: 2).validate(),
        throwsArgumentError,
      );
    });

    test('validate rejects numScratch < 1', () {
      expect(
        () => const LoomControllerConfig(numScratch: 0).validate(),
        throwsArgumentError,
      );
    });
  });

  // -------------------------------------------------------------------------
  // Structural / SV-emission tests (no simulator needed)
  // -------------------------------------------------------------------------

  group('LoomController structural', () {
    late LoomController ctrl;

    setUp(() async {
      ctrl = LoomController(baseAddress: 0x60000000);
      await ctrl.build();
    });

    test('builds and emits non-empty SystemVerilog', () {
      final sv = ctrl.generateSynth();
      expect(sv, isNotEmpty);
    });

    test('SV contains the module name', () {
      final sv = ctrl.generateSynth();
      expect(sv, contains('LoomController'));
    });

    test('SV contains Wishbone input signal names', () {
      final sv = ctrl.generateSynth();
      expect(sv, contains('bus_CYC'));
      expect(sv, contains('bus_STB'));
      expect(sv, contains('bus_WE'));
      expect(sv, contains('bus_ADR'));
      expect(sv, contains('bus_DAT_MOSI'));
      expect(sv, contains('bus_SEL'));
    });

    test('SV contains Wishbone output signal names', () {
      final sv = ctrl.generateSynth();
      expect(sv, contains('bus_ACK'));
      expect(sv, contains('bus_DAT_MISO'));
    });

    test('has Wishbone bus slave port', () {
      expect(ctrl.bus, isNotNull);
    });

    test('bus has 32-bit data width', () {
      expect(ctrl.bus.dataIn.width, equals(32));
      expect(ctrl.bus.dataOut.width, equals(32));
    });
  });

  // -------------------------------------------------------------------------
  // Register-map / SVD tests
  // -------------------------------------------------------------------------

  group('LoomController SVD / register map', () {
    test('svdPeripheral name is LOOM_CONTROLLER', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      expect(ctrl.svdPeripheral.name, equals('LOOM_CONTROLLER'));
    });

    test('svdPeripheral baseAddress matches constructor arg', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      expect(ctrl.svdPeripheral.baseAddress, equals(0x60000000));
    });

    test('svdPeripheral registers include VERSION at offset 0x00', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      final regs = ctrl.svdPeripheral.registers;
      expect(regs, isNotNull);
      final versionField = regs!['VERSION'];
      expect(versionField, isNotNull);
      expect(versionField!.offset, equals(0x00));
      expect(versionField.readOnly, isTrue);
    });

    test('svdPeripheral registers include CONTROL at offset 0x04', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      final regs = ctrl.svdPeripheral.registers;
      final controlField = regs!['CONTROL'];
      expect(controlField, isNotNull);
      expect(controlField!.offset, equals(0x04));
      expect(controlField.readOnly, isFalse);
    });

    test('svdPeripheral registers include STATUS at offset 0x08', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      final regs = ctrl.svdPeripheral.registers;
      final statusField = regs!['STATUS'];
      expect(statusField, isNotNull);
      expect(statusField!.offset, equals(0x08));
      expect(statusField.readOnly, isTrue);
    });

    test('svdPeripheral registers include SCRATCH at offset 0x0C', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      final regs = ctrl.svdPeripheral.registers;
      final scratchField = regs!['SCRATCH'];
      expect(scratchField, isNotNull);
      expect(scratchField!.offset, equals(0x0C));
      expect(scratchField.readOnly, isFalse);
    });

    test('register map has no overlaps', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      final errors = ctrl.svdPeripheral.registers!.validate();
      expect(errors, isEmpty);
    });
  });

  // -------------------------------------------------------------------------
  // Device-tree tests
  // -------------------------------------------------------------------------

  group('LoomController device tree', () {
    test('dtNode has correct compatible string', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      expect(ctrl.dtNode.compatible, contains('midstall,loom-controller'));
    });

    test('dtNode reg matches baseAddress', () {
      final ctrl = LoomController(baseAddress: 0x60000000);
      expect(ctrl.dtNode.reg.start, equals(0x60000000));
    });
  });

  // -------------------------------------------------------------------------
  // Bus-level simulation: Wishbone read/write
  //
  // Driving pattern from harbor's ddr_train_test.dart:
  //   module.input('bus_CYC').srcConnection! <= signal
  //
  // All sim tests share one bringUp/teardown per group (setUp/tearDown).
  // Splitting into separate test files per ROHD house style would also work,
  // but keeping them here as individual tests with shared setUp is cleaner.
  // -------------------------------------------------------------------------

  group('LoomController Wishbone simulation', () {
    late Logic clk, reset, cyc, stb, we, adr, datMosi, sel;
    late LoomController ctrl;

    setUp(() async {
      clk = SimpleClockGenerator(10).clk;
      reset = Logic(name: 'reset');
      ctrl = LoomController(baseAddress: 0x60000000);

      cyc = Logic(name: 'cyc');
      stb = Logic(name: 'stb');
      we = Logic(name: 'we');
      adr = Logic(name: 'adr', width: ctrl.input('bus_ADR').width);
      datMosi = Logic(name: 'datMosi', width: ctrl.input('bus_DAT_MOSI').width);
      sel = Logic(name: 'sel', width: ctrl.input('bus_SEL').width);

      ctrl.input('clk').srcConnection! <= clk;
      ctrl.input('reset').srcConnection! <= reset;
      ctrl.input('bus_CYC').srcConnection! <= cyc;
      ctrl.input('bus_STB').srcConnection! <= stb;
      ctrl.input('bus_WE').srcConnection! <= we;
      ctrl.input('bus_ADR').srcConnection! <= adr;
      ctrl.input('bus_DAT_MOSI').srcConnection! <= datMosi;
      ctrl.input('bus_SEL').srcConnection! <= sel;

      await ctrl.build();

      for (final s in [cyc, stb, we, adr, datMosi, sel]) {
        s.inject(0);
      }
      reset.inject(1);
      Simulator.setMaxSimTime(200000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;
    });

    Logic ack() => ctrl.output('bus_ACK');
    Logic miso() => ctrl.output('bus_DAT_MISO');

    Future<void> wbWrite(int address, int data) async {
      cyc.inject(1);
      stb.inject(1);
      we.inject(1);
      adr.inject(address);
      datMosi.inject(data);
      var guard = 0;
      while (!ack().value.toBool() && guard++ < 100) {
        await clk.nextPosedge;
      }
      await clk.nextPosedge;
      cyc.inject(0);
      stb.inject(0);
      we.inject(0);
      await clk.nextPosedge;
    }

    Future<int> wbRead(int address) async {
      cyc.inject(1);
      stb.inject(1);
      we.inject(0);
      adr.inject(address);
      var guard = 0;
      while (!ack().value.toBool() && guard++ < 100) {
        await clk.nextPosedge;
      }
      final v = miso().value.toInt();
      await clk.nextPosedge;
      cyc.inject(0);
      stb.inject(0);
      await clk.nextPosedge;
      return v;
    }

    tearDown(() async {
      await Simulator.endSimulation();
    });

    test('VERSION register reads the magic constant', () async {
      // The bus uses raw byte addresses; VERSION is at byte offset 0.
      final v = await wbRead(0x00);
      expect(
        v,
        equals(0x4C4F4F4D),
        reason: 'VERSION should read 0x4C4F4F4D ("LOOM")',
      );
    });

    test('SCRATCH register round-trips a value', () async {
      // Write 0xDEADC0DE to SCRATCH (byte offset 0x0C).
      await wbWrite(0x0C, 0xDEADC0DE);
      final v = await wbRead(0x0C);
      expect(
        v,
        equals(0xDEADC0DE),
        reason: 'SCRATCH should read back the written value',
      );
    });

    test('SCRATCH second round-trip with different value', () async {
      await wbWrite(0x0C, 0x12345678);
      final v = await wbRead(0x0C);
      expect(v, equals(0x12345678));
    });

    test('STATUS reads as 0 (skeleton)', () async {
      final v = await wbRead(0x08);
      expect(v, equals(0), reason: 'STATUS skeleton is always 0');
    });

    test('CONTROL is writable and readable', () async {
      // Write bit 1 (soft-reset), not bit 0 (start, self-clears).
      await wbWrite(0x04, 0x2);
      final v = await wbRead(0x04);
      // bit 0 may have self-cleared; bit 1 stays.
      expect(v & 0x2, equals(0x2), reason: 'CONTROL bit 1 should be sticky');
    });
  });
}
