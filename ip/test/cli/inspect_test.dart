@TestOn('vm')
library;

import 'dart:io';
import 'package:test/test.dart';

void main() {
  test('inspect prints a summary for a config.json', () async {
    final result = await Process.run('dart', [
      'run',
      'bin/loom_inspect.dart',
      'test/fixtures/smollm2_config.json',
    ]);
    expect(result.exitCode, 0);
    expect(result.stdout, contains('llama'));
    expect(result.stdout, contains('layers: 30'));
    expect(result.stdout, contains('hidden: 576'));
  });

  test('--version prints version and exits 0', () async {
    final result = await Process.run('dart', [
      'run',
      'bin/loom_inspect.dart',
      '--version',
    ]);
    expect(result.exitCode, 0);
    expect(result.stdout, contains('0.0.1'));
  });

  test('--help exits 0 and prints usage', () async {
    final result = await Process.run('dart', [
      'run',
      'bin/loom_inspect.dart',
      '--help',
    ]);
    expect(result.exitCode, 0);
    expect((result.stdout as String).toLowerCase(), contains('usage'));
  });
}
