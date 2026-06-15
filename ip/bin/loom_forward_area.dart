// Throwaway area-check harness: emit LoomForward SV at the real stories260K
// config for a given maxSeq, so yosys can stat cell counts. Not functional.
//
//   dart run bin/loom_forward_area.dart <maxSeq> <out.sv>
//
// LoomForward.generateSynth() throws because its many differently-sized
// LoomFpLinear/LoomStreamMatmul engines all reserve the same definitionName
// ("LoomStreamMatmul"), and ROHD's SynthBuilder refuses to uniquify a reserved
// name. We only need cell counts, so we replicate SynthBuilder here with a
// uniquifier that DOES rename collisions (LoomStreamMatmul, LoomStreamMatmul_0,
// ...). Structurally-identical modules still dedupe to one definition.
library;

import 'dart:io';

import 'package:rohd/rohd.dart';

import 'package:loom/src/hw/forward.dart';

/// Emit one merged SystemVerilog string for [top], renaming (not reserving)
/// definition-name collisions so a design with several distinct same-named
/// module definitions still emits.
String emitMerged(Module top) {
  final synth = SystemVerilogSynthesizer();

  // BFS collect every module that needs a definition (SynthBuilder order).
  final toParse = <Module>[top];
  for (var i = 0; i < toParse.length; i++) {
    final m = toParse[i];
    if (!synth.generatesDefinition(m)) continue;
    toParse.addAll(m.subModules);
  }

  final moduleToName = <Module, String>{};
  final results = <SynthesisResult>[];
  final resultToName = <SynthesisResult, String>{};
  final takenNames = <String>{};
  final counters = <String, int>{};

  String unique(String base) {
    if (!takenNames.contains(base)) {
      takenNames.add(base);
      return base;
    }
    counters[base] ??= -1;
    String candidate;
    do {
      counters[base] = counters[base]! + 1;
      candidate = '${base}_${counters[base]}';
    } while (takenNames.contains(candidate));
    takenNames.add(candidate);
    return candidate;
  }

  String instanceType(Module m) {
    if (!synth.generatesDefinition(m)) return '*NONE*';
    final existing = moduleToName[m];
    if (existing != null) return existing;
    final result = synth.synthesize(m, instanceType);
    String name;
    final dupIdx = results.indexWhere((r) => r == result);
    if (dupIdx >= 0) {
      name = resultToName[results[dupIdx]]!;
    } else {
      results.add(result);
      name = unique(m.definitionName);
      resultToName[result] = name;
    }
    moduleToName[m] = name;
    return name;
  }

  // Bottom-up so children are named before parents reference them.
  for (final m in toParse.reversed) {
    if (synth.generatesDefinition(m)) instanceType(m);
  }

  final chunks = <String>[];
  for (final r in results) {
    for (final f in r.toSynthFileContents()) {
      chunks.add(f.contents);
    }
  }
  return chunks.join('\n\n////////////////////\n\n');
}

Future<void> main(List<String> args) async {
  final maxSeq = args.isNotEmpty ? int.parse(args[0]) : 64;
  final outPath = args.length > 1 ? args[1] : 'loom_forward_$maxSeq.sv';

  final m = LoomForward(
    hidden: 64,
    numHeads: 8,
    numKvHeads: 4,
    headDim: 8,
    intermediateSize: 172,
    maxSeq: maxSeq,
    numLayers: 5,
    vocab: 512,
  );
  await m.build();
  final sv = emitMerged(m);
  File(outPath).writeAsStringSync(sv);
  stderr.writeln('wrote $outPath (${sv.length} bytes), maxSeq=$maxSeq');
}
