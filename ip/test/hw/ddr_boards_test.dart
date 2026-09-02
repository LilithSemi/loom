import 'package:harbor/harbor.dart';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  group('LoomDdrBoard', () {
    test('orangecrab-25f carries its own pad sites', () {
      final pins = LoomDdrBoard.pinsFor('orangecrab-25f');
      expect(pins['sdram_ck'], 'J18 SSTL135_I SLEWRATE=FAST');
      expect(pins['sdram_dq[0]'], contains('C17'));
      expect(pins['sdram_dqs_n[1]'], contains('H17'));
      expect(
        LoomDdrBoard.require('orangecrab-25f').config.size,
        128 * 1024 * 1024,
      );
    });

    test('arty-s7-50 takes its pad sites from the board catalog', () {
      final pins = LoomDdrBoard.pinsFor('arty-s7-50');
      final catalog = HarborBoard.get('arty-s7-50').pins;
      expect(pins['sdram_ck'], catalog['sdram_ck']);
      expect(pins['sdram_dq[15]'], catalog['sdram_dq[15]']);
      // The OrangeCrab sites must not leak into another board.
      expect(pins['sdram_ck'], isNot('J18 SSTL135_I SLEWRATE=FAST'));
      expect(pins.keys, everyElement(startsWith('sdram_')));
      expect(LoomDdrBoard.require('arty-s7-50').config.size, 256 * 1024 * 1024);
    });

    test('a board with no entry gives an error that lists the known names', () {
      expect(
        () => LoomDdrBoard.require('ulx3s-85f'),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            allOf(contains('orangecrab-25f'), contains('arty-s7-50')),
          ),
        ),
      );
    });

    test('every pad port of the controller has a site on each board', () {
      for (final board in LoomDdrBoard.byName.keys) {
        final pins = LoomDdrBoard.pinsFor(board);
        for (final port in LoomDdrBoard.padPorts) {
          final constrained = pins.keys.any(
            (k) => k == port || k.startsWith('$port['),
          );
          expect(constrained, isTrue, reason: '$board has no $port site');
        }
      }
    });
  });
}
