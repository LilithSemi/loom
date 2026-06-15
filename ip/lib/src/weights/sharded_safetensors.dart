import 'dart:convert';
import 'dart:io';

import '../ir/tensor.dart';
import 'safetensors.dart';
import 'weight_store.dart';

/// A [WeightStore] over a SHARDED safetensors checkpoint: a
/// `model.safetensors.index.json` whose `weight_map` routes each tensor name to
/// one of several `model-0000K-of-0000N.safetensors` shard files. Every shard is
/// parsed once into a [SafetensorsStore], and lookups route by the weight map.
/// This is what every model larger than one shard (most >=1B models) ships as.
class ShardedSafetensorsStore implements WeightStore {
  final Map<String, SafetensorsStore> _shards; // shard filename -> parsed store
  final Map<String, String> _weightMap; // tensor name -> shard filename

  ShardedSafetensorsStore._(this._shards, this._weightMap);

  /// Load from a `model.safetensors.index.json` path. Shard files are resolved
  /// relative to the index file's directory.
  factory ShardedSafetensorsStore.fromIndexFile(String indexPath) {
    final index = jsonDecode(File(indexPath).readAsStringSync());
    if (index is! Map<String, dynamic> || index['weight_map'] == null) {
      throw ArgumentError('safetensors index has no weight_map: $indexPath');
    }
    final rawMap = index['weight_map'] as Map<String, dynamic>;
    final weightMap = rawMap.map((k, v) => MapEntry(k, v as String));
    final dir = File(indexPath).parent.path;
    final shards = <String, SafetensorsStore>{};
    for (final shardFile in weightMap.values.toSet()) {
      final f = File('$dir/$shardFile');
      if (!f.existsSync()) {
        throw ArgumentError(
          'safetensors shard missing: $shardFile (from $indexPath)',
        );
      }
      shards[shardFile] = SafetensorsStore.parse(f.readAsBytesSync());
    }
    return ShardedSafetensorsStore._(shards, weightMap);
  }

  @override
  bool contains(String name) => _weightMap.containsKey(name);

  @override
  TensorView get(String name) {
    final shardFile = _weightMap[name];
    if (shardFile == null) {
      throw ArgumentError('tensor not in any shard: $name');
    }
    return _shards[shardFile]!.get(name);
  }

  @override
  Iterable<String> get names => _weightMap.keys;
}
