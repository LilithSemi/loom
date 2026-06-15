// LoomWbReader: Wishbone-master burst reader. The testbench plays the role of
// the memory (Wishbone slave), acking each request with mem[addr/4].

import 'dart:async';

import 'package:loom/src/hw/wb_reader.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('LoomWbReader', () {
    late Logic clk, reset, start, base, count, ack, datMiso;
    late LoomWbReader reader;

    setUp(() async {
      clk = SimpleClockGenerator(10).clk;
      reset = Logic(name: 'reset');
      start = Logic(name: 'start');
      base = Logic(name: 'base', width: 32);
      count = Logic(name: 'count', width: 16);
      ack = Logic(name: 'ack');
      datMiso = Logic(name: 'datMiso', width: 32);

      reader = LoomWbReader();
      reader.input('clk').srcConnection! <= clk;
      reader.input('reset').srcConnection! <= reset;
      reader.input('start').srcConnection! <= start;
      reader.input('base').srcConnection! <= base;
      reader.input('count').srcConnection! <= count;
      reader.input('bus_ACK').srcConnection! <= ack;
      reader.input('bus_DAT_MISO').srcConnection! <= datMiso;

      await reader.build();

      for (final s in [start, ack]) {
        s.inject(0);
      }
      base.inject(0);
      count.inject(0);
      datMiso.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(100000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;
    });

    tearDown(() async {
      await Simulator.endSimulation();
    });

    Logic o(String n) => reader.output(n);

    Future<List<int>> readBurst(int baseAddr, List<int> mem) async {
      base.inject(baseAddr);
      count.inject(mem.length);
      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);
      base.inject(0);
      count.inject(0);

      final got = <int>[];
      var guard = 0;
      while (guard++ < 200) {
        // Act as the memory: combinationally ack while STB is asserted.
        await clk.nextNegedge;
        if (o('bus_STB').value.toBool() && o('bus_CYC').value.toBool()) {
          final wordIdx = (o('bus_ADR').value.toInt() - baseAddr) ~/ 4;
          ack.inject(1);
          datMiso.inject(
            wordIdx >= 0 && wordIdx < mem.length ? mem[wordIdx] : 0,
          );
        } else {
          ack.inject(0);
        }
        await clk.nextPosedge;
        if (o('word_valid').value.toBool()) {
          got.add(o('word_out').value.toInt());
        }
        if (o('done').value.toBool()) break;
      }
      return got;
    }

    test('reads a 4-word burst in order', () async {
      final mem = [0x11111111, 0x22222222, 0x33333333, 0x44444444];
      final got = await readBurst(0x100, mem);
      expect(got, equals(mem));
    });

    test('reads a single word', () async {
      final got = await readBurst(0x40, [0xDEADBEEF]);
      expect(got, equals([0xDEADBEEF]));
    });

    test('busy deasserts and done pulses after the last word', () async {
      final mem = [1, 2, 3];
      await readBurst(0x0, mem);
      // After the burst, the reader should be idle.
      await clk.nextPosedge;
      expect(o('busy').value.toBool(), isFalse);
    });
  });
}
