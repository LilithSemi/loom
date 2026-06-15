library;

import '../ir/tensor.dart';

/// A name-addressable collection of tensors loaded from a model file.
abstract class WeightStore {
  /// All tensor names available in this store.
  Iterable<String> get names;

  /// Whether [name] exists in this store.
  bool contains(String name);

  /// Returns the tensor [name]; throws [ArgumentError] if absent.
  TensorView get(String name);
}

/// A [WeightStore] backed by an in-memory map (useful for tests/assembly).
class MapWeightStore implements WeightStore {
  final Map<String, TensorView> _tensors;
  MapWeightStore(this._tensors);

  @override
  Iterable<String> get names => _tensors.keys;

  @override
  bool contains(String name) => _tensors.containsKey(name);

  @override
  TensorView get(String name) =>
      _tensors[name] ??
      (throw ArgumentError.value(name, 'name', 'no such tensor'));
}
