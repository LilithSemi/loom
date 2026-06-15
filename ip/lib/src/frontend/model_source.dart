import 'dart:convert';
import 'dart:io';

import '../ir/model_graph.dart';
import '../weights/binding.dart';
import '../weights/gguf.dart';
import '../weights/llama2c.dart';
import '../weights/safetensors.dart';
import '../weights/sharded_safetensors.dart';
import '../weights/weight_store.dart';
import 'gguf_config.dart';
import 'gguf_names.dart';
import 'gguf_tokenizer.dart';
import 'hf_config.dart';

/// A model loaded from any supported source, reduced to the common shape the
/// artifact generator consumes: the arch graph, the bound weights, the max
/// sequence length, and an optional tokenizer file to carry into the artifacts.
class LoadedModel {
  final ModelGraph graph;
  final BoundModel model;
  final int maxSeq;

  /// A tokenizer file to copy into the artifacts (HF `tokenizer.json` or a
  /// llama2.c `tok*.bin`), when the source has one on disk.
  final String? tokenizerPath;

  /// Tokenizer content extracted from the source (GGUF embeds it in metadata,
  /// there is no file), as a `tokenizer.json` string for the generator to write.
  final String? tokenizerJson;

  const LoadedModel({
    required this.graph,
    required this.model,
    required this.maxSeq,
    this.tokenizerPath,
    this.tokenizerJson,
  });
}

/// Auto-detect the input format and load it. A directory is a HuggingFace model
/// dir (config.json + model.safetensors + tokenizer.json). A `.gguf` file is
/// reserved for frontend slice 2. Anything else is a llama2.c `.bin`.
LoadedModel loadModelSource(String path) {
  if (FileSystemEntity.isDirectorySync(path)) return _loadHfDir(path);
  if (path.toLowerCase().endsWith('.gguf')) return _loadGguf(path);
  return _loadLlama2c(path);
}

LoadedModel _loadGguf(String path) {
  final store = GgufStore.parse(File(path).readAsBytesSync());
  final name = path
      .split(Platform.pathSeparator)
      .where((s) => s.isNotEmpty)
      .last;
  final graph = parseGgufConfig(store, name: name);
  final a = graph.layers.first.attention;
  final view = GgufHfView(
    store,
    numHeads: a.numHeads,
    numKvHeads: a.numKvHeads,
    headDim: a.headDim,
  );
  final model = bindWeights(graph, view);
  // GGUF embeds its tokenizer in metadata, extract it to a tokenizer.json string
  // the generator can emit (best effort, a GGUF without tokenizer metadata just
  // yields no tokenizer artifact rather than failing the whole compile).
  String? tokJson;
  try {
    tokJson = jsonEncode(ggufTokenizerJson(store));
  } on ArgumentError {
    tokJson = null;
  }
  return LoadedModel(
    graph: graph,
    model: model,
    maxSeq: graph.maxSeqLen,
    tokenizerJson: tokJson,
  );
}

LoadedModel _loadLlama2c(String path) {
  final ckpt = Llama2cCheckpoint.parse(File(path).readAsBytesSync());
  return LoadedModel(
    graph: ckpt.graph,
    model: bindWeights(ckpt.graph, ckpt.store),
    maxSeq: ckpt.maxSeq,
  );
}

LoadedModel _loadHfDir(String dir) {
  final cfg = File('$dir/config.json');
  final index = File('$dir/model.safetensors.index.json');
  final single = File('$dir/model.safetensors');
  final tok = File('$dir/tokenizer.json');
  if (!cfg.existsSync()) {
    throw ArgumentError('HF model dir is missing config.json: $dir');
  }
  final name = dir
      .split(Platform.pathSeparator)
      .where((s) => s.isNotEmpty)
      .last;
  final graph = parseHfConfig(
    jsonDecode(cfg.readAsStringSync()) as Map<String, dynamic>,
    name: name,
  );
  // A sharded checkpoint (index.json + weight_map) wins, else the single file.
  final WeightStore store;
  if (index.existsSync()) {
    store = ShardedSafetensorsStore.fromIndexFile(index.path);
  } else if (single.existsSync()) {
    store = SafetensorsStore.parse(single.readAsBytesSync());
  } else {
    throw ArgumentError(
      'HF model dir has no model.safetensors or model.safetensors.index.json: $dir',
    );
  }
  final model = bindWeights(graph, store);
  return LoadedModel(
    graph: graph,
    model: model,
    maxSeq: graph.maxSeqLen,
    tokenizerPath: tok.existsSync() ? tok.path : null,
  );
}
