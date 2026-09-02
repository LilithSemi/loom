import 'package:harbor/harbor.dart';

/// Board-level DDR3 definitions: the DRAM part configuration and the pad
/// constraint table of each board that Loom attaches a weight store to.
///
/// `--ddr` reads the entry of the `--board` name. A board that this table does
/// not hold gives an error that lists the known names, so a build cannot take
/// the pad sites of a different board. This is the rule that River uses in
/// `DdrBoard.byName`.
class LoomDdrBoard {
  /// The DRAM part configuration for [HarborDdrController].
  final HarborDdrConfig config;

  /// Pad constraints, keyed by the pad port names of the controller. A vector
  /// pad uses `port[index]`, which is the name in the synthesized netlist. The
  /// value is `SITE [IO_TYPE] [ATTR=VAL...]`.
  ///
  /// An empty table means that Harbor's board catalog holds the sites for this
  /// board. [pinsFor] reads them from there.
  final Map<String, String> pins;

  const LoomDdrBoard({required this.config, this.pins = const {}});

  /// The pad ports of the controller. `loom_genip` exposes each one as an
  /// external pin of the SoC.
  static const padPorts = <String>[
    'sdram_ck',
    'sdram_cke',
    'sdram_cs_n',
    'sdram_ras_n',
    'sdram_cas_n',
    'sdram_we_n',
    'sdram_ba',
    'sdram_addr',
    'sdram_dm',
    'sdram_dq',
    'sdram_ck_n',
    'sdram_dqs',
    'sdram_dqs_n',
    'sdram_odt',
    'sdram_reset_n',
  ];

  /// The DDR boards, keyed by the name in Harbor's board catalog.
  static const byName = <String, LoomDdrBoard>{
    'orangecrab-25f': _orangeCrab25f,
    'arty-s7-50': _artyS750,
  };

  /// The entry for [board]. It gives an error that lists the known names when
  /// the table does not hold that board.
  static LoomDdrBoard require(String board) {
    final entry = byName[board];
    if (entry == null) {
      throw ArgumentError(
        'Board "$board" has no DDR pad table. Known: ${byName.keys.join(', ')}. '
        'To use DDR on another board, add it here or give every sdram_* site '
        'with --pin.',
      );
    }
    return entry;
  }

  /// The pad constraints for [board]: the table of the entry, or the `sdram_*`
  /// sites of Harbor's board catalog when the entry holds none.
  static Map<String, String> pinsFor(String board) {
    final entry = require(board);
    if (entry.pins.isNotEmpty) return entry.pins;
    return {
      for (final e in HarborBoard.get(board).pins.entries)
        if (e.key.startsWith('sdram_')) e.key: e.value,
    };
  }

  /// OrangeCrab r0.2: a Micron MT41K64M16 DDR3L, 128 MB on 16 bits. The sites
  /// come from River's `DdrBoard.byName['orangecrab']`, which is proven on this
  /// board with the same controller. Harbor's board catalog does not hold them
  /// yet. The data pads add `TERMINATION=OFF`. Centre the read eye with the
  /// static read tap (`--ddr-read-tap`).
  static const _orangeCrab25f = LoomDdrBoard(
    config: HarborDdrConfig.orangeCrab(),
    pins: _orangeCrab25fPins,
  );

  /// Digilent Arty S7-50: a Micron MT41K128M16 DDR3L, 256 MB on 16 bits.
  /// Harbor's board catalog holds the `sdram_*` sites, so this entry gives the
  /// part configuration only. Nobody has proven a Loom DDR build on this board.
  static const _artyS750 = LoomDdrBoard(config: HarborDdrConfig.artyS7());
}

/// The OrangeCrab r0.2 DDR3 pad sites, SSTL135_I.
const Map<String, String> _orangeCrab25fPins = {
  'sdram_ck': 'J18 SSTL135_I SLEWRATE=FAST',
  'sdram_ck_n': 'K18 SSTL135_I SLEWRATE=FAST',
  'sdram_cke': 'D18 SSTL135_I SLEWRATE=FAST',
  'sdram_cs_n': 'A12 SSTL135_I SLEWRATE=FAST',
  'sdram_ras_n': 'C12 SSTL135_I SLEWRATE=FAST',
  'sdram_cas_n': 'D13 SSTL135_I SLEWRATE=FAST',
  'sdram_we_n': 'B12 SSTL135_I SLEWRATE=FAST',
  'sdram_odt': 'C13 SSTL135_I SLEWRATE=FAST',
  'sdram_reset_n': 'L18 SSTL135_I SLEWRATE=FAST',
  'sdram_ba[0]': 'D6 SSTL135_I SLEWRATE=FAST',
  'sdram_ba[1]': 'B7 SSTL135_I SLEWRATE=FAST',
  'sdram_ba[2]': 'A6 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[0]': 'C4 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[1]': 'D2 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[2]': 'D3 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[3]': 'A3 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[4]': 'A4 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[5]': 'D4 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[6]': 'C3 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[7]': 'B2 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[8]': 'B1 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[9]': 'D1 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[10]': 'A7 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[11]': 'C2 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[12]': 'B6 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[13]': 'C1 SSTL135_I SLEWRATE=FAST',
  'sdram_addr[14]': 'A2 SSTL135_I SLEWRATE=FAST',
  'sdram_dm[0]': 'G16 SSTL135_I SLEWRATE=FAST',
  'sdram_dm[1]': 'D16 SSTL135_I SLEWRATE=FAST',
  'sdram_dq[0]': 'C17 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[1]': 'D15 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[2]': 'B17 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[3]': 'C16 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[4]': 'A15 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[5]': 'B13 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[6]': 'A17 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[7]': 'A13 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[8]': 'F17 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[9]': 'F16 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[10]': 'G15 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[11]': 'F15 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[12]': 'J16 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[13]': 'C18 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[14]': 'H16 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dq[15]': 'F18 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dqs[0]': 'B15 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dqs[1]': 'G18 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dqs_n[0]': 'A16 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
  'sdram_dqs_n[1]': 'H17 SSTL135_I SLEWRATE=FAST TERMINATION=OFF',
};
