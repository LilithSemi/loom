// Tests Loom SoC composition and harbor generateAll emission: LoomUsbBridge
// as the Wishbone bus master, LoomAccelerator as the slave, emitted to a temp
// directory. Asserts the emitted artifacts and modules exist.

import 'dart:io';

import 'package:test/test.dart';

import '../../bin/loom_genip.dart'
    show buildLoomSoc, LoomTransport, OverlayDatapath;

void main() {
  group('Loom SoC generateAll', () {
    late Directory tmp;

    setUp(() {
      tmp = Directory.systemTemp.createTempSync('loom_soc_gen_');
    });

    tearDown(() {
      if (tmp.existsSync()) tmp.deleteSync(recursive: true);
    });

    test('composes USB master + accelerator slave and emits RTL', () async {
      final soc = buildLoomSoc(
        name: 'LoomSoC',
        transport: LoomTransport.usb,
        datapath: const OverlayDatapath(),
      );
      await soc.generateAll(tmp);

      final rtlDir = Directory('${tmp.path}/rtl');
      expect(rtlDir.existsSync(), isTrue, reason: 'rtl/ directory missing');

      final svFiles = rtlDir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.sv'))
          .toList();
      expect(svFiles, isNotEmpty, reason: 'no .sv files emitted');

      // Concatenate all emitted SV for content assertions.
      final allSv = svFiles.map((f) => f.readAsStringSync()).join('\n');
      expect(allSv.trim(), isNotEmpty, reason: 'emitted SV is empty');

      final topSv = File('${tmp.path}/rtl/LoomSoC.sv');
      expect(topSv.existsSync(), isTrue, reason: 'LoomSoC.sv missing');
      final top = topSv.readAsStringSync();
      expect(top, contains('module LoomSoC'));
      // Bus master (vendor USB device), slave (accelerator), and the decoder
      // between.
      expect(
        top,
        contains('LoomUsbDevice'),
        reason: 'top must instantiate the vendor USB device master',
      );
      expect(
        top,
        contains('LoomAccelerator'),
        reason: 'top must instantiate the accelerator slave',
      );
      expect(
        top,
        contains('WishboneDecoder'),
        reason: 'top must instantiate the wishbone decoder',
      );

      // The accelerator and the USB device modules have their own SV files.
      expect(
        svFiles.any((f) => f.path.endsWith('LoomAccelerator.sv')),
        isTrue,
        reason: 'LoomAccelerator.sv missing',
      );
      expect(
        svFiles.any((f) => f.path.endsWith('LoomUsbDevice.sv')),
        isTrue,
        reason: 'LoomUsbDevice.sv missing',
      );
      // The EP0 + bulk engine sub-module proves USB transport (enumeration +
      // EP1 bulk) made it into RTL: harbor's UsbEp0Engine, configured with
      // loom's vendor descriptors and bulk endpoints.
      expect(
        svFiles.any((f) => f.path.endsWith('UsbEp0Engine.sv')),
        isTrue,
        reason: 'UsbEp0Engine.sv (harbor USB transport) missing',
      );
      // The harbor PHY/packet building blocks are reused, not reinvented.
      expect(
        svFiles.any((f) => f.path.endsWith('HarborUsbFsPhyRx.sv')),
        isTrue,
        reason: 'HarborUsbFsPhyRx.sv (reused PHY) missing',
      );

      // The accelerator's own module references its name (non-empty body).
      final accelSv = File(
        '${tmp.path}/rtl/LoomAccelerator.sv',
      ).readAsStringSync();
      expect(accelSv, contains('module LoomAccelerator'));

      final dts = File('${tmp.path}/LoomSoC.dts');
      expect(dts.existsSync(), isTrue, reason: 'LoomSoC.dts missing');
      expect(dts.readAsStringSync(), contains('midstall,loom'));

      final svd = File('${tmp.path}/LoomSoC.svd');
      expect(svd.existsSync(), isTrue, reason: 'LoomSoC.svd missing');
      expect(svd.readAsStringSync().trim(), isNotEmpty);

      final lpf = File('${tmp.path}/LoomSoC.lpf');
      expect(lpf.existsSync(), isTrue, reason: 'LoomSoC.lpf missing');
      final lpfText = lpf.readAsStringSync();
      // The 48 MHz clock and the USB pins must be constrained.
      expect(lpfText, contains('"clk"'));
      expect(lpfText, contains('"usb_dp"'));

      expect(
        File('${tmp.path}/synth.tcl').existsSync(),
        isTrue,
        reason: 'synth.tcl missing',
      );
      expect(
        File('${tmp.path}/Makefile').existsSync(),
        isTrue,
        reason: 'Makefile missing',
      );
    });

    test('composes UART master + accelerator slave and emits RTL', () async {
      final soc = buildLoomSoc(
        name: 'LoomUartSoC',
        transport: LoomTransport.uart,
        datapath: const OverlayDatapath(),
      );
      await soc.generateAll(tmp);

      final topSv = File('${tmp.path}/rtl/LoomUartSoC.sv');
      expect(topSv.existsSync(), isTrue, reason: 'LoomUartSoC.sv missing');
      final top = topSv.readAsStringSync();
      expect(top, contains('module LoomUartSoC'));
      // UART bridge master, accelerator slave, decoder between.
      expect(
        top,
        contains('LoomUartBridge'),
        reason: 'top must instantiate the UART bridge master',
      );
      expect(
        top,
        contains('LoomAccelerator'),
        reason: 'top must instantiate the accelerator slave',
      );
      expect(
        top,
        contains('WishboneDecoder'),
        reason: 'top must instantiate the wishbone decoder',
      );

      final rtlDir = Directory('${tmp.path}/rtl');
      final svFiles = rtlDir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.sv'))
          .toList();
      // The UART engine and the shared command engine (same one as the USB
      // path) both make it into RTL.
      expect(
        svFiles.any((f) => f.path.endsWith('LoomUart.sv')),
        isTrue,
        reason: 'LoomUart.sv missing',
      );
      expect(
        svFiles.any((f) => f.path.endsWith('LoomUsbCmdEngine.sv')),
        isTrue,
        reason: 'reused LoomUsbCmdEngine.sv missing',
      );

      // The UART pins are constrained in the bare-SoC LPF.
      final lpf = File('${tmp.path}/LoomUartSoC.lpf').readAsStringSync();
      expect(lpf, contains('"clk"'));
      expect(lpf, contains('"uart_tx"'));
      expect(lpf, contains('"uart_rx"'));
    });
  });
}
