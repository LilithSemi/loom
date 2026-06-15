import 'dart:convert';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

// Minimal HF BPE json: vocab of single-char byte-alphabet tokens + one merge.
Map<String, dynamic> _miniJson() => {
  'added_tokens': [
    {'id': 0, 'content': '<|endoftext|>'},
  ],
  'model': {
    'type': 'BPE',
    'ignore_merges': false,
    'vocab': {'<|endoftext|>': 0, 'a': 1, 'b': 2, 'ab': 3},
    'merges': ['a b'],
  },
};

void main() {
  test('toLtb1 lays out magic, flags, ids, vocab, merges, specials', () {
    final t = BpeTokenizer.fromJson(_miniJson());
    final bytes = t.toLtb1(bosId: 0, eosId: 0, addBos: false);
    final bd = ByteData.sublistView(bytes);
    expect(utf8.decode(bytes.sublist(0, 4)), 'LTB1');
    var p = 4;
    expect(bd.getUint8(p), 0x01); // flags: split_digits only
    p += 1;
    expect(bd.getInt32(p, Endian.little), 0); // bos_id
    p += 4;
    expect(bd.getInt32(p, Endian.little), 0); // eos_id
    p += 4;
    expect(bd.getUint8(p), 0); // add_bos
    p += 1;
    expect(bd.getUint32(p, Endian.little), 4); // vocab_count
    p += 4;
    // id 0 "<|endoftext|>"
    final l0 = bd.getUint16(p, Endian.little);
    p += 2;
    expect(utf8.decode(bytes.sublist(p, p + l0)), '<|endoftext|>');
    p += l0;
    // ids 1,2,3 = "a","b","ab"
    for (final want in ['a', 'b', 'ab']) {
      final l = bd.getUint16(p, Endian.little);
      p += 2;
      expect(utf8.decode(bytes.sublist(p, p + l)), want);
      p += l;
    }
    expect(bd.getUint32(p, Endian.little), 1); // merges_count
    p += 4;
    expect(bd.getUint32(p, Endian.little), 1); // id_a = "a"
    p += 4;
    expect(bd.getUint32(p, Endian.little), 2); // id_b = "b"
    p += 4;
    expect(bd.getUint32(p, Endian.little), 1); // specials_count
    p += 4;
    final ls = bd.getUint16(p, Endian.little);
    p += 2;
    expect(utf8.decode(bytes.sublist(p, p + ls)), '<|endoftext|>');
    p += ls;
    expect(bd.getUint32(p, Endian.little), 0); // special id
  });
}
