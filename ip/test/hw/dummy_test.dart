import 'package:rohd/rohd.dart';
import 'package:test/test.dart';
import 'package:loom/src/hw/dummy.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('LoomDummy builds and emits non-empty SystemVerilog', () async {
    final inp = Logic(name: 'inp');
    final mod = LoomDummy(inp);
    await mod.build();

    final sv = mod.generateSynth();

    expect(sv, isNotEmpty);
    expect(sv, contains('LoomDummy'));
  });

  test('LoomDummy output follows input in simulation', () async {
    final inp = Logic(name: 'inp');
    final mod = LoomDummy(inp);
    await mod.build();

    inp.put(0);
    expect(mod.out.value.toInt(), equals(0));

    inp.put(1);
    expect(mod.out.value.toInt(), equals(1));
  });
}
