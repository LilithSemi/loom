@TestOn('vm')
library;

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('library version constant is exported', () {
    expect(loomVersion, isNotEmpty);
  });
}
