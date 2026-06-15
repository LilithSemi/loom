library;

import 'dart:convert';
import 'dart:typed_data';

/// Decoder for llama2.c's `tokenizer.bin` format (e.g. `tok512.bin` shipped
/// with `stories260K`). Mirrors run.c's `decode`:
///
/// File layout: `int32 max_token_length`, then `vocab_size` entries of
/// `{float32 score, int32 len, byte[len] piece}`. SentencePiece whitespace is
/// stored as `U+2581` and rendered as a space. The leading space of the first
/// piece after BOS is dropped; `<0xXX>` pieces are raw byte fallbacks.
class Llama2cTokenizer {
  /// Token id -> piece string (with the whitespace marker normalized to ' ').
  final List<String> vocab;
  final int bos;
  final int eos;

  Llama2cTokenizer._(this.vocab, {this.bos = 1, this.eos = 2});

  factory Llama2cTokenizer.parse(Uint8List bytes, {int bos = 1, int eos = 2}) {
    final bd = ByteData.sublistView(bytes);
    var p = 0;
    // int32 max_token_length (unused for decoding).
    p += 4;

    final vocab = <String>[];
    while (p + 8 <= bytes.length) {
      // float32 score (unused for decoding).
      p += 4;
      final len = bd.getInt32(p, Endian.little);
      p += 4;
      if (len < 0 || p + len > bytes.length) break;
      final piece = utf8.decode(
        bytes.sublist(p, p + len),
        allowMalformed: true,
      );
      p += len;
      // SentencePiece stores whitespace as U+2581 ('▁'). Render as a space.
      vocab.add(piece.replaceAll('▁', ' '));
    }

    return Llama2cTokenizer._(vocab, bos: bos, eos: eos);
  }

  /// Number of tokens in the vocabulary.
  int get length => vocab.length;

  /// Decodes [tokens] to text. BOS/EOS render as empty. The leading space of a
  /// piece immediately following BOS is dropped (SentencePiece convention).
  String decode(List<int> tokens) {
    final sb = StringBuffer();
    for (var i = 0; i < tokens.length; i++) {
      final t = tokens[i];
      if (t < 0 || t >= vocab.length) continue;
      if (t == bos || t == eos) continue;
      var piece = vocab[t];
      final prev = i > 0 ? tokens[i - 1] : -1;
      if (prev == bos && piece.startsWith(' ')) {
        piece = piece.substring(1);
      }
      // Raw byte fallback: "<0xXX>" -> the byte value XX.
      final m = RegExp(r'^<0x([0-9A-Fa-f]{2})>$').firstMatch(piece);
      if (m != null) {
        sb.writeCharCode(int.parse(m.group(1)!, radix: 16));
        continue;
      }
      sb.write(piece);
    }
    return sb.toString();
  }
}
