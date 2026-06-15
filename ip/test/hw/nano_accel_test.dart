// Diff-test: Loom Nano on-chip inference (LoomAccelerator) vs the Dart golden
// NanoModel. The device owns the context and computes argmax(W.onehot(ctx));
// the test drives only RESET/PUSH/STEP and reads INFER_OUT, exactly as the
// model-agnostic runtime does.

import 'dart:async';

import 'package:loom/src/hw/accelerator.dart';
import 'package:loom/src/nano/nano_model.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

const _regInferReset = 0x020;
const _regInferPush = 0x024;
const _regInferStep = 0x028;
const _regInferOut = 0x02C;

int _pattern4(int i) => [0, 0, 1, 1][i % 4];

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('LoomAccelerator Nano on-chip inference', () {
    late Logic clk, reset, cyc, stb, we, adr, datMosi, sel;
    late LoomAccelerator accel;
    late NanoModel model;
    const cfg = NanoConfig(contextBits: 3);

    setUp(() async {
      clk = SimpleClockGenerator(10).clk;
      reset = Logic(name: 'reset');

      final target = [for (var i = 0; i < 64; i++) _pattern4(i)];
      model = NanoModel.train(cfg, target);

      accel = LoomAccelerator(
        config: LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          nanoContextBits: cfg.contextBits,
          nanoWeights: model.weights,
        ),
      );

      cyc = Logic(name: 'cyc');
      stb = Logic(name: 'stb');
      we = Logic(name: 'we');
      adr = Logic(name: 'adr', width: accel.input('bus_ADR').width);
      datMosi = Logic(
        name: 'datMosi',
        width: accel.input('bus_DAT_MOSI').width,
      );
      sel = Logic(name: 'sel', width: accel.input('bus_SEL').width);

      accel.input('clk').srcConnection! <= clk;
      accel.input('reset').srcConnection! <= reset;
      accel.input('bus_CYC').srcConnection! <= cyc;
      accel.input('bus_STB').srcConnection! <= stb;
      accel.input('bus_WE').srcConnection! <= we;
      accel.input('bus_ADR').srcConnection! <= adr;
      accel.input('bus_DAT_MOSI').srcConnection! <= datMosi;
      accel.input('bus_SEL').srcConnection! <= sel;

      await accel.build();

      for (final s in [cyc, stb, we, adr, datMosi, sel]) {
        s.inject(0);
      }
      reset.inject(1);
      Simulator.setMaxSimTime(2000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;
    });

    tearDown(() async {
      await Simulator.endSimulation();
    });

    Logic ack() => accel.output('bus_ACK');
    Logic miso() => accel.output('bus_DAT_MISO');

    Future<void> wbWrite(int address, int data) async {
      cyc.inject(1);
      stb.inject(1);
      we.inject(1);
      adr.inject(address);
      datMosi.inject(data & 0xFFFFFFFF);
      var guard = 0;
      while (!ack().value.toBool() && guard++ < 200) {
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
      while (!ack().value.toBool() && guard++ < 200) {
        await clk.nextPosedge;
      }
      final v = miso().value.toInt();
      await clk.nextPosedge;
      cyc.inject(0);
      stb.inject(0);
      await clk.nextPosedge;
      return v;
    }

    // Prime the device context to [state] by pushing its bits MSB-first.
    Future<void> primeContext(int state) async {
      await wbWrite(_regInferReset, 0);
      for (var i = cfg.contextBits - 1; i >= 0; i--) {
        await wbWrite(_regInferPush, (state >> i) & 1);
      }
    }

    // Generate [n] bits on chip, reading each via INFER_OUT.
    Future<List<int>> deviceGenerate(int startState, int n) async {
      await primeContext(startState);
      final out = <int>[];
      for (var i = 0; i < n; i++) {
        await wbWrite(_regInferStep, 0);
        final v = await wbRead(_regInferOut);
        out.add(v & 1);
      }
      return out;
    }

    test('on-chip generation is bit-exact to the golden NanoModel', () async {
      for (final start in [0, 1, 0x3, 0x5, 0x7]) {
        final hw = await deviceGenerate(start, 16);
        final golden = model.generate(start, 16);
        expect(hw, equals(golden), reason: 'start=$start');
      }
    });

    test('INFER_OUT exposes the running context in the upper bits', () async {
      await primeContext(0x5); // ctx = 0b101
      final v = await wbRead(_regInferOut);
      expect((v >> 1) & cfg.contextMask, equals(0x5));
    });
  });
}
