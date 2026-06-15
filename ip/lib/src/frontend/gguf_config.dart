library;

import '../ir/model_graph.dart';
import '../weights/gguf.dart';
import 'hf_config.dart';

/// Build a [ModelGraph] from a GGUF file's metadata by mapping the GGUF keys
/// to the HuggingFace config shape and reusing [parseHfConfig]. Only the
/// modern llama-family metadata layout is handled.
///
/// This is a metadata-only building block: it produces a graph shape, not a
/// golden-correct inference path. RoPE un-permutation and the GGML tensor
/// transpose convention are handled elsewhere (or not yet at all). A
/// reference-GGUF end-to-end match is a later slice.
ModelGraph parseGgufConfig(GgufStore store, {String? name}) {
  final arch = store.metaString('general.architecture');
  if (arch == null) {
    throw ArgumentError('GGUF is missing general.architecture');
  }
  final heads = store.metaInt('$arch.attention.head_count');
  final hidden = store.metaInt('$arch.embedding_length');
  if (heads == null || hidden == null) {
    throw ArgumentError('GGUF metadata missing head_count/embedding_length');
  }
  final tokens = store.meta('tokenizer.ggml.tokens');
  final vocab = (tokens is List) ? tokens.length : null;
  final cfg = <String, dynamic>{
    'model_type': arch,
    'hidden_size': hidden,
    'num_hidden_layers': store.metaInt('$arch.block_count'),
    'num_attention_heads': heads,
    'num_key_value_heads':
        store.metaInt('$arch.attention.head_count_kv') ?? heads,
    'intermediate_size': store.metaInt('$arch.feed_forward_length'),
    'vocab_size': vocab,
    'max_position_embeddings': store.metaInt('$arch.context_length') ?? 2048,
    'rms_norm_eps':
        (store.meta('$arch.attention.layer_norm_rms_epsilon') as num?)
            ?.toDouble() ??
        1e-5,
    'rope_theta':
        (store.meta('$arch.rope.freq_base') as num?)?.toDouble() ?? 10000.0,
    'tie_word_embeddings': !store.contains('output.weight'),
    'hidden_act': 'silu',
    if (store.metaInt('$arch.rope.dimension_count') != null)
      'head_dim': store.metaInt('$arch.rope.dimension_count'),
  };
  return parseHfConfig(cfg, name: name);
}
