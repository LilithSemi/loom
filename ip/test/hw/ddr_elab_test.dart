// Structural check that Harbor's OrangeCrab DDR3 controller instantiates and
// emits RTL in our build (the ECP5 PHY uses blackbox primitives with no sim
// model, so we verify elaboration + SV emission, not behaviour).

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('HarborDdrController(orangeCrab) elaborates and emits SV', () async {
    final ddr = HarborDdrController(
      config: const HarborDdrConfig.orangeCrab(),
      baseAddress: 0x20000000,
      clockHz: 48000000, // DLL-off bring-up: CK = system clock
      busAddressWidth: 32,
      target: HarborFpgaTarget.ecp5(
        device: '25f',
        package: 'CSFBGA285',
        frequency: 48000000,
      ),
    );
    await ddr.build();

    final sv = ddr.generateSynth();
    expect(sv, contains('module HarborDdrController'));
    expect(sv, isNotEmpty);

    // 128 MB part, 32-bit bus -> the device-tree reg range is the full size.
    expect(ddr.dtNode.reg.size, equals(128 * 1024 * 1024));
    // The SDRAM data pads should be present (DDR x16).
    expect(
      ddr.tryOutput('sdram_addr') ?? ddr.tryInput('sdram_addr'),
      isNotNull,
    );
  });
}
