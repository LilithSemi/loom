// The fp16 W4A8 linear SoC as a real HarborSoC: UART host bridge +
// LoomFpLinearAccelerator (CSR slave peripheral AND weight-read master) +
// HarborSram (weights), wired by HarborSoC.buildFabric (auto-arbiter for the
// two masters). Device runs the W4A8 matmuls, host orchestrates the
// nonlinear glue. This uses flop buffers (bring-up size). Full SmolLM2 dims
// need BRAM buffers + host row tiling.

import 'dart:io';

import 'package:harbor/harbor.dart';
import 'package:loom/loom.dart' show LoomDdrBoard;
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

import '../../bin/loom_genip.dart' show buildLoomSoc, LoomTransport, FpDatapath;

HarborFpgaTarget _ecp5() => HarborFpgaTarget.ecp5(
  device: '25f',
  package: 'CSFBGA285',
  frequency: 48000000,
);

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    'fp linear SoC composes as a HarborSoC and emits all submodules',
    () async {
      final soc = buildLoomSoc(
        name: 'LoomFpSoC',
        transport: LoomTransport.uart,
        target: _ecp5(),
        fabricHz: 16000000,
        datapath: const FpDatapath(),
      );
      await soc.build();
      final sv = soc.generateSynth();
      expect(sv, contains('LoomFpLinearAccelerator'));
      expect(sv, contains('LoomFpLinear'));
      expect(sv, contains('LoomActQuant'));
      expect(sv, contains('LoomDequant'));
      expect(sv, contains('LoomStreamMatmul'));
      expect(sv, contains('LoomUartBridge'));
      expect(sv, contains('HarborSram'));
      expect(sv, contains('WishboneArbiter_M2')); // 2 masters -> auto-arbiter
      expect(sv, contains('WishboneDecoder'));
    },
  );

  test(
    'FPGA target uses internal POR: flashable from clk + UART (no reset pin)',
    () async {
      final soc = buildLoomSoc(
        name: 'LoomFpSoC',
        transport: LoomTransport.uart,
        target: _ecp5(),
        fabricHz: 16000000,
        datapath: const FpDatapath(),
      );
      await soc.build();
      expect(soc.tryInput('clk'), isNotNull);
      expect(soc.tryInput('uart_rx'), isNotNull);
      expect(soc.tryOutput('uart_tx'), isNotNull);
      expect(
        soc.tryInput('reset'),
        isNull,
        reason: 'internal POR, no reset pin',
      );
    },
  );

  test('Harbor generates the full ECP5 flow (no hand-written SV)', () async {
    final soc = buildLoomSoc(
      name: 'LoomFpSoC',
      transport: LoomTransport.uart,
      target: _ecp5(),
      fabricHz: 16000000,
      datapath: const FpDatapath(),
    );
    final dir = Directory.systemTemp.createTempSync('loom_fp_soc_');
    try {
      await soc.generateAll(dir);
      expect(File('${dir.path}/filelist.f').existsSync(), isTrue);
      expect(File('${dir.path}/LoomFpSoC.lpf').existsSync(), isTrue);
      expect(File('${dir.path}/synth.tcl').existsSync(), isTrue);
      expect(File('${dir.path}/Makefile').existsSync(), isTrue);
      expect(Directory('${dir.path}/rtl').listSync().isNotEmpty, isTrue);
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test(
    'weights-in-DDR variant composes with HarborDdrController + SDRAM pads',
    () async {
      final soc = buildLoomSoc(
        name: 'LoomFpSoC',
        transport: LoomTransport.uart,
        target: _ecp5(),
        fabricHz: 16000000,
        datapath: FpDatapath(ddr: LoomDdrBoard.require('orangecrab-25f')),
      );
      await soc.build();
      final sv = soc.generateSynth();
      expect(sv, contains('HarborDdrController'));
      expect(sv, contains('LoomFpLinearAccelerator'));
      expect(soc.tryOutput('sdram_addr'), isNotNull);
      expect(soc.tryInOut('sdram_dq'), isNotNull);
    },
  );

  test(
    'tiered BRAM cache variant adds a second HarborSram weight store',
    () async {
      // bramCacheKb adds a 90KB on-chip HarborSram hot-weight store next to the
      // flash cold store, reachable by both the UART bridge master and the
      // accelerator mem-master via the fabric decoder's address range.
      final soc = buildLoomSoc(
        name: 'LoomFpSoC',
        transport: LoomTransport.uart,
        target: _ecp5(),
        fabricHz: 16000000,
        datapath: const FpDatapath(weightsInFlash: true, bramCacheKb: 90),
      );
      await soc.build();
      final sv = soc.generateSynth();
      expect(sv, contains('LoomFpLinearAccelerator'));
      expect(sv, contains('HarborSpiFlashController'));
      // Two HarborSram definitions: the 16KB scratchpad + the 90KB store.
      expect(sv, contains('HarborSram_92160'));
      expect(sv, contains('WishboneArbiter_M2')); // still two masters
      // The 90KB store maps into DP16KD (x9-per-byte-lane), not the sim model.
      expect(sv, contains('DP16KD'));
    },
  );
}
