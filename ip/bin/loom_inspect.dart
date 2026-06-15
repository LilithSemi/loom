/// Parses a HuggingFace config.json and prints the resulting Loom IR summary.
///
/// Usage: `loom_inspect [--help] [--version] <config.json>`
///
/// Flags:
///   -h, --help     Print this usage information and exit.
///       --version  Print the loom library version and exit.
///
/// Positional argument:
///   `<config.json>`  Path to a HuggingFace config.json file to inspect.
library;

import 'dart:convert';
import 'dart:io';

import 'package:args/args.dart';
import 'package:loom/loom.dart';

void main(List<String> argv) {
  final parser = ArgParser()
    ..addFlag(
      'help',
      abbr: 'h',
      negatable: false,
      help: 'Print usage and exit.',
    )
    ..addFlag(
      'version',
      negatable: false,
      help: 'Print the loom version and exit.',
    );

  final ArgResults args;
  try {
    args = parser.parse(argv);
  } on FormatException catch (e) {
    stderr.writeln(e.message);
    stderr.writeln('Usage: loom_inspect [--help] [--version] <config.json>');
    exit(2);
  }

  if (args['help'] as bool) {
    stdout.writeln('Usage: loom_inspect [--help] [--version] <config.json>');
    stdout.writeln();
    stdout.writeln(parser.usage);
    exit(0);
  }

  if (args['version'] as bool) {
    stdout.writeln(loomVersion);
    exit(0);
  }

  if (args.rest.isEmpty) {
    stderr.writeln('usage: loom_inspect <config.json>');
    exit(2);
  }

  final json =
      jsonDecode(File(args.rest.first).readAsStringSync())
          as Map<String, dynamic>;
  final g = parseHfConfig(json);
  final a = g.layers.first.attention;
  stdout
    ..writeln('model:  ${g.name}')
    ..writeln('arch:   ${g.arch.name}')
    ..writeln('hidden: ${g.hiddenSize}')
    ..writeln('layers: ${g.numLayers}')
    ..writeln('vocab:  ${g.vocabSize}')
    ..writeln('heads:  ${a.numHeads} q / ${a.numKvHeads} kv x ${a.headDim}')
    ..writeln(
      'mlp:    ${g.layers.first.mlp.intermediateSize} '
      '(${g.layers.first.mlp.activation.name}'
      '${g.layers.first.mlp.gated ? ', gated' : ''})',
    )
    ..writeln('tied:   ${g.tieEmbeddings}');
}
