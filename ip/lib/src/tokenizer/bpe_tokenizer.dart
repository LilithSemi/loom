import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Byte-level BPE tokenizer compatible with HuggingFace `tokenizer.json`
/// files of the GPT-2 / Llama (SmolLM2) family.
///
/// Pipeline mirrors the reference exactly:
///   1. split out literal special / added tokens,
///   2. `Digits(individual_digits)` then `ByteLevel(use_regex)` pre-tokenizing,
///   3. map UTF-8 bytes to the GPT-2 byte-to-unicode alphabet,
///   4. merge by rank (honouring `ignore_merges`),
///   5. look up ids in the vocab.
class BpeTokenizer {
  final Map<String, int> _vocab;
  final Map<int, String> _idToToken;
  final Map<String, int> _mergeRank; // "$a $b" -> rank
  final bool _ignoreMerges;

  /// content -> id for special / added tokens.
  final Map<String, int> _specialToId;
  final Map<int, String> _idToSpecial;
  final RegExp? _specialPattern;

  static final Map<int, String> _byteToUnicode = _buildByteToUnicode();
  static final Map<String, int> _unicodeToByte = {
    for (final e in _byteToUnicode.entries) e.value: e.key,
  };

  // GPT-2 ByteLevel split regex. Digits are pre-split to single chars upstream,
  // so ` ?\p{N}+` only ever sees one digit at a time.
  static final RegExp _gpt2Pattern = RegExp(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    unicode: true,
  );

  BpeTokenizer._({
    required Map<String, int> vocab,
    required Map<String, int> mergeRank,
    required bool ignoreMerges,
    required Map<String, int> specialToId,
  }) : _vocab = vocab,
       _idToToken = {for (final e in vocab.entries) e.value: e.key},
       _mergeRank = mergeRank,
       _ignoreMerges = ignoreMerges,
       _specialToId = specialToId,
       _idToSpecial = {for (final e in specialToId.entries) e.value: e.key},
       _specialPattern = specialToId.isEmpty
           ? null
           : RegExp(
               (specialToId.keys.toList()
                     ..sort((a, b) => b.length.compareTo(a.length)))
                   .map(RegExp.escape)
                   .join('|'),
             );

  /// Builds a tokenizer from a parsed `tokenizer.json` map.
  factory BpeTokenizer.fromJson(Map<String, dynamic> json) {
    final model = json['model'] as Map<String, dynamic>;
    final rawVocab = model['vocab'] as Map<String, dynamic>;
    final vocab = <String, int>{
      for (final e in rawVocab.entries) e.key: e.value as int,
    };

    final mergeRank = <String, int>{};
    final merges = (model['merges'] as List?) ?? const [];
    for (var i = 0; i < merges.length; i++) {
      final m = merges[i];
      String a, b;
      if (m is String) {
        final sp = m.indexOf(' ');
        a = m.substring(0, sp);
        b = m.substring(sp + 1);
      } else {
        final pair = m as List;
        a = pair[0] as String;
        b = pair[1] as String;
      }
      mergeRank['$a $b'] = i;
    }

    final specialToId = <String, int>{};
    final added = (json['added_tokens'] as List?) ?? const [];
    for (final a in added) {
      final t = a as Map<String, dynamic>;
      specialToId[t['content'] as String] = t['id'] as int;
    }

    return BpeTokenizer._(
      vocab: vocab,
      mergeRank: mergeRank,
      ignoreMerges: (model['ignore_merges'] as bool?) ?? false,
      specialToId: specialToId,
    );
  }

  /// Loads a tokenizer from a `tokenizer.json` file on disk.
  factory BpeTokenizer.fromFile(String path) => BpeTokenizer.fromJson(
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>,
  );

  /// Number of entries in the vocabulary (excluding pure added tokens not in
  /// the base vocab map).
  int get vocabSize => _vocab.length;

  /// Encodes [text] into token ids.
  List<int> encode(String text) {
    if (text.isEmpty) return const [];
    final ids = <int>[];
    for (final segment in _splitSpecials(text)) {
      if (segment.specialId != null) {
        ids.add(segment.specialId!);
      } else {
        _encodeSegment(segment.text, ids);
      }
    }
    return ids;
  }

  /// Serializes to the runtime's LTB1 tokenizer.bin format (see the BPE plan).
  Uint8List toLtb1({
    required int bosId,
    required int eosId,
    required bool addBos,
    bool splitDigits = true,
  }) {
    final b = BytesBuilder();
    void u8(int v) => b.addByte(v & 0xff);
    void u16(int v) => b.add([v & 0xff, (v >> 8) & 0xff]);
    void u32(int v) =>
        b.add([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff]);
    void i32(int v) => u32(v & 0xffffffff);
    void str(String s) {
      final bytes = utf8.encode(s);
      u16(bytes.length);
      b.add(bytes);
    }

    b.add(utf8.encode('LTB1'));
    var flags = 0;
    if (splitDigits) flags |= 0x01;
    if (_ignoreMerges) flags |= 0x02;
    u8(flags);
    i32(bosId);
    i32(eosId);
    u8(addBos ? 1 : 0);

    // Vocab, dense by id 0..maxId. Ids are contiguous in these tokenizers.
    final maxId = _idToToken.keys.fold<int>(-1, (m, k) => k > m ? k : m);
    u32(maxId + 1);
    for (var id = 0; id <= maxId; id++) {
      str(_idToToken[id] ?? '');
    }

    // Merges as id pairs, in rank order (mergeRank value == rank).
    final ranked = _mergeRank.entries.toList()
      ..sort((x, y) => x.value.compareTo(y.value));
    u32(ranked.length);
    for (final e in ranked) {
      final sp = e.key.indexOf(' ');
      final a = e.key.substring(0, sp);
      final c = e.key.substring(sp + 1);
      u32(_vocab[a]!);
      u32(_vocab[c]!);
    }

    // Specials.
    u32(_specialToId.length);
    for (final e in _specialToId.entries) {
      str(e.key);
      u32(e.value);
    }
    return b.toBytes();
  }

  /// Decodes token [ids] back into text.
  String decode(List<int> ids) {
    final out = StringBuffer();
    final byteBuf = <int>[];

    void flush() {
      if (byteBuf.isEmpty) return;
      out.write(utf8.decode(byteBuf, allowMalformed: true));
      byteBuf.clear();
    }

    for (final id in ids) {
      final special = _idToSpecial[id];
      if (special != null) {
        flush();
        out.write(special);
        continue;
      }
      final tok = _idToToken[id];
      if (tok == null) continue; // unknown id, skip rather than crash
      for (final ch in tok.runes) {
        final byte = _unicodeToByte[String.fromCharCode(ch)];
        if (byte != null) byteBuf.add(byte);
      }
    }
    flush();
    return out.toString();
  }

  // Internals

  void _encodeSegment(String segment, List<int> out) {
    for (final chunk in _splitDigits(segment)) {
      for (final m in _gpt2Pattern.allMatches(chunk)) {
        final piece = m[0]!;
        final byteLevel = _toByteLevel(piece);
        for (final tok in _bpe(byteLevel)) {
          final id = _vocab[tok];
          if (id != null) out.add(id);
        }
      }
    }
  }

  /// Maps a raw substring to its GPT-2 byte-level unicode form.
  String _toByteLevel(String piece) {
    final sb = StringBuffer();
    for (final byte in utf8.encode(piece)) {
      sb.write(_byteToUnicode[byte]);
    }
    return sb.toString();
  }

  /// Applies BPE merges to one byte-level word, returning its sub-tokens.
  List<String> _bpe(String word) {
    if (_ignoreMerges && _vocab.containsKey(word)) return [word];

    var symbols = word.runes.map(String.fromCharCode).toList();
    if (symbols.length < 2) return symbols;

    while (true) {
      var bestRank = -1;
      var bestIdx = -1;
      for (var i = 0; i < symbols.length - 1; i++) {
        final rank = _mergeRank['${symbols[i]} ${symbols[i + 1]}'];
        if (rank != null && (bestIdx == -1 || rank < bestRank)) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx == -1) break;

      final a = symbols[bestIdx];
      final b = symbols[bestIdx + 1];
      final merged = a + b;
      final next = <String>[];
      for (var i = 0; i < symbols.length;) {
        if (i < symbols.length - 1 && symbols[i] == a && symbols[i + 1] == b) {
          next.add(merged);
          i += 2;
        } else {
          next.add(symbols[i]);
          i += 1;
        }
      }
      symbols = next;
      if (symbols.length < 2) break;
    }
    return symbols;
  }

  /// Splits a non-special segment so each digit becomes its own chunk and
  /// non-digit runs stay together (the `Digits(individual_digits)` step).
  List<String> _splitDigits(String s) {
    final chunks = <String>[];
    final buf = StringBuffer();
    for (final ch in s.runes) {
      final isDigit = ch >= 0x30 && ch <= 0x39;
      if (isDigit) {
        if (buf.isNotEmpty) {
          chunks.add(buf.toString());
          buf.clear();
        }
        chunks.add(String.fromCharCode(ch));
      } else {
        buf.writeCharCode(ch);
      }
    }
    if (buf.isNotEmpty) chunks.add(buf.toString());
    return chunks;
  }

  /// Splits [text] around literal special-token strings.
  List<_Segment> _splitSpecials(String text) {
    final pat = _specialPattern;
    if (pat == null) return [_Segment(text: text)];

    final segments = <_Segment>[];
    var last = 0;
    for (final m in pat.allMatches(text)) {
      if (m.start > last) {
        segments.add(_Segment(text: text.substring(last, m.start)));
      }
      segments.add(_Segment(specialId: _specialToId[m[0]!]));
      last = m.end;
    }
    if (last < text.length) {
      segments.add(_Segment(text: text.substring(last)));
    }
    return segments;
  }

  static Map<int, String> _buildByteToUnicode() {
    final bs = <int>[];
    for (var i = '!'.codeUnitAt(0); i <= '~'.codeUnitAt(0); i++) {
      bs.add(i);
    }
    for (var i = 0xA1; i <= 0xAC; i++) {
      bs.add(i);
    }
    for (var i = 0xAE; i <= 0xFF; i++) {
      bs.add(i);
    }
    final cs = List<int>.from(bs);
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    return {
      for (var i = 0; i < bs.length; i++) bs[i]: String.fromCharCode(cs[i]),
    };
  }
}

class _Segment {
  final String text;
  final int? specialId;
  _Segment({this.text = '', this.specialId});
}
