import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// Synthetic GGUF writer helper (copied verbatim from test/weights/gguf_test.dart,
// plus the float32 / string-array kv helpers this test needs).

/// A small byte-buffer builder that grows as needed.
class _BufWriter {
  final _chunks = <int>[];

  void writeUint8(int v) => _chunks.add(v & 0xFF);

  void writeUint32(int v) {
    final bd = ByteData(4)..setUint32(0, v, Endian.little);
    for (var i = 0; i < 4; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeInt8(int v) {
    final bd = ByteData(1)..setInt8(0, v);
    _chunks.add(bd.getUint8(0));
  }

  void writeUint16(int v) {
    final bd = ByteData(2)..setUint16(0, v, Endian.little);
    for (var i = 0; i < 2; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeUint64(int v) {
    // Dart int is 64-bit. Split into two 32-bit halves.
    final lo = v & 0xFFFFFFFF;
    final hi = (v >> 32) & 0xFFFFFFFF;
    writeUint32(lo);
    writeUint32(hi);
  }

  void writeFloat32(double v) {
    final bd = ByteData(4)..setFloat32(0, v, Endian.little);
    for (var i = 0; i < 4; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeGgufString(String s) {
    final bytes = _utf8Encode(s);
    writeUint64(bytes.length);
    for (final b in bytes) {
      _chunks.add(b);
    }
  }

  void writeBytes(List<int> bytes) {
    for (final b in bytes) {
      _chunks.add(b);
    }
  }

  Uint8List toBytes() => Uint8List.fromList(_chunks);

  int get length => _chunks.length;

  static List<int> _utf8Encode(String s) {
    final units = <int>[];
    for (final rune in s.runes) {
      if (rune < 0x80) {
        units.add(rune);
      } else if (rune < 0x800) {
        units.add(0xC0 | (rune >> 6));
        units.add(0x80 | (rune & 0x3F));
      } else if (rune < 0x10000) {
        units.add(0xE0 | (rune >> 12));
        units.add(0x80 | ((rune >> 6) & 0x3F));
        units.add(0x80 | (rune & 0x3F));
      } else {
        units.add(0xF0 | (rune >> 18));
        units.add(0x80 | ((rune >> 12) & 0x3F));
        units.add(0x80 | ((rune >> 6) & 0x3F));
        units.add(0x80 | (rune & 0x3F));
      }
    }
    return units;
  }
}

// GGUF metadata value_type constants (only those used by the writer helpers)
const int _kvUint32 = 4;
const int _kvString = 8;

/// Write magic + version + tensor_count + kv_count header to a writer.
void _writeGgufPrologue(
  _BufWriter w, {
  required int version,
  required int tensorCount,
  required int kvCount,
}) {
  // Magic: 'GGUF' = 0x46554747 LE
  w.writeUint8(0x47);
  w.writeUint8(0x47);
  w.writeUint8(0x55);
  w.writeUint8(0x46);
  w.writeUint32(version);
  w.writeUint64(tensorCount);
  w.writeUint64(kvCount);
}

/// Write a STRING kv pair.
void _writeKvString(_BufWriter w, String key, String value) {
  w.writeGgufString(key);
  w.writeUint32(_kvString);
  w.writeGgufString(value);
}

/// Write a UINT32 kv pair.
void _writeKvUint32(_BufWriter w, String key, int value) {
  w.writeGgufString(key);
  w.writeUint32(_kvUint32);
  w.writeUint32(value);
}

const int _kvFloat32 = 6;
void _writeKvFloat32(_BufWriter w, String key, double value) {
  w.writeGgufString(key);
  w.writeUint32(_kvFloat32);
  w.writeFloat32(value);
}

const int _kvStringArr = 9;
const int _kvStringType = 8;
void _writeKvStringArray(_BufWriter w, String key, List<String> values) {
  w.writeGgufString(key);
  w.writeUint32(_kvStringArr);
  w.writeUint32(_kvStringType); // arr element type = string
  w.writeUint64(values.length);
  for (final s in values) {
    w.writeGgufString(s);
  }
}

Uint8List _tinyLlamaGguf() {
  final w = _BufWriter();
  // 10 metadata kv pairs.
  _writeGgufPrologue(w, version: 3, tensorCount: 0, kvCount: 10);
  _writeKvString(w, 'general.architecture', 'llama');
  _writeKvUint32(w, 'llama.embedding_length', 8);
  _writeKvUint32(w, 'llama.block_count', 2);
  _writeKvUint32(w, 'llama.attention.head_count', 2);
  _writeKvUint32(w, 'llama.attention.head_count_kv', 2);
  _writeKvUint32(w, 'llama.feed_forward_length', 16);
  _writeKvUint32(w, 'llama.context_length', 32);
  _writeKvFloat32(w, 'llama.attention.layer_norm_rms_epsilon', 1e-5);
  _writeKvFloat32(w, 'llama.rope.freq_base', 10000.0);
  _writeKvStringArray(
    w,
    'tokenizer.ggml.tokens',
    List.generate(10, (i) => 't$i'),
  );
  return w.toBytes();
}

void main() {
  test('parseGgufConfig builds a llama ModelGraph from GGUF metadata', () {
    final store = GgufStore.parse(_tinyLlamaGguf());
    final g = parseGgufConfig(store, name: 'tiny');
    expect(g.hiddenSize, 8);
    expect(g.layers.length, 2);
    expect(g.vocabSize, 10);
    expect(g.maxSeqLen, 32);
    // No output.weight tensor was written, so embeddings are tied.
    expect(g.tieEmbeddings, isTrue);
  });
}
