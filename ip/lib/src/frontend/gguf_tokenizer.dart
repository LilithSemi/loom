import '../weights/gguf.dart';

// GGUF tokenizer token types (tokenizer.ggml.token_type). CONTROL and
// USER_DEFINED tokens are the special/added tokens.
const int _ggufTokenControl = 3;
const int _ggufTokenUserDefined = 4;

/// Extract a HuggingFace-style `tokenizer.json` map from a GGUF file's embedded
/// tokenizer metadata (`tokenizer.ggml.tokens` / `.merges` / `.token_type`), so
/// it can feed `BpeTokenizer.fromJson`. GGUF stores the vocabulary as an array
/// (index == token id); we invert it to the {token: id} map fromJson expects.
///
/// Targets the byte-level BPE (gpt2-pretokenizer) family, which is what SmolLM2
/// and the llama-family GGUFs we ingest use.
Map<String, dynamic> ggufTokenizerJson(GgufStore store) {
  final rawTokens = store.meta('tokenizer.ggml.tokens');
  if (rawTokens is! List) {
    throw ArgumentError('GGUF has no tokenizer.ggml.tokens array');
  }
  final tokens = rawTokens.cast<String>();
  final rawMerges = store.meta('tokenizer.ggml.merges');
  final merges = (rawMerges is List)
      ? rawMerges.cast<String>()
      : const <String>[];
  final rawTypes = store.meta('tokenizer.ggml.token_type');
  final types = (rawTypes is List) ? rawTypes.cast<int>() : null;

  final vocab = <String, dynamic>{};
  for (var i = 0; i < tokens.length; i++) {
    vocab[tokens[i]] = i;
  }

  final added = <Map<String, dynamic>>[];
  if (types != null) {
    for (var i = 0; i < tokens.length && i < types.length; i++) {
      if (types[i] == _ggufTokenControl || types[i] == _ggufTokenUserDefined) {
        added.add({'content': tokens[i], 'id': i});
      }
    }
  }

  return {
    'model': {'type': 'BPE', 'vocab': vocab, 'merges': merges},
    'added_tokens': added,
  };
}
