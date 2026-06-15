library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_rmsnorm.dart';
import 'linear_seq.dart';

/// Balanced mux-tree select: `entries[index]`.
Logic _sel(Logic index, List<Logic> entries) {
  if (entries.length == 1) return entries[0];
  final next = <Logic>[];
  for (var i = 0; i < entries.length; i += 2) {
    next.add(
      i + 1 < entries.length
          ? mux(index[0], entries[i + 1], entries[i])
          : entries[i],
    );
  }
  return _sel(index.getRange(1, index.width), next);
}

/// The model back-end: turns the decoder's final `hidden[H]` into the next-token
/// id, mirroring [GoldenRunner] "Step 3: final norm + lm_head" (runner.dart
/// ~149): `logits = Wcls @ rmsNorm(hidden, finalNormGamma, eps)`, then argmax.
///
///   * final RMSNorm via [LoomRmsNorm] (single-shot: held in reset off-phase so
///     its accumulator/reciprocal start clean each run) -> f[H].
///   * logits = Wcls @ f  [vocab, H] via [LoomLinearSeq] (row-tiled by the
///     engine. Wcls may be tied to the embedding but is read as a normal weight
///     matrix from `mem` at `wb_cls`).
///   * STREAMING argmax: as the `vocab` logits stream out of the matmul, a
///     running (bestKey, bestIdx) is updated. fp16 logits are compared via a
///     monotonic unsigned key (`key = sign ? ~bits : bits | 0x8000`), which
///     preserves the real-value ordering, so a plain unsigned `>` picks the max
///     without any float comparator. Ties keep the lower index (strict `>`).
///
/// Load protocol (state LOAD): stream `hidden`+`finalNormGamma` on
/// `x_en`/`x_in`/`gamma_in`, the Wcls per-row scales on `sc_en`/`sc_in`. Set
/// `wb_cls`, `inv_n` (=1/H), `eps`. Pulse `start`. The chosen token id is held on
/// `token` with `token_valid`/`done` pulsing at the end.
class LoomLmHead extends BridgeModule {
  static const int _load = 0;
  static const int _nAcc = 1;
  static const int _nComp = 2;
  static const int _nWait = 3;
  static const int _nNorm = 4;
  static const int _lFeed = 5;
  static const int _lStart = 6;
  static const int _lRun = 7;
  static const int _fin = 8;

  late final WishboneInterface mem;

  LoomLmHead({
    required int hidden,
    required int vocab,
    int recipIterations = 4,
    String? name,
  }) : super('LoomLmHead', name: name ?? 'loom_lm_head') {
    const aw = 32;
    final H = hidden;
    final tokenW = vocab.bitLength;
    final colTiles = (H + 1) ~/ 2;
    final rowBlocks = (vocab + 1) ~/ 2;
    final feedMax = H > vocab ? H : vocab;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('wb_cls', PortDirection.input, width: aw);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('gamma_in', PortDirection.input, width: 16);
    createPort('sc_en', PortDirection.input);
    createPort('sc_in', PortDirection.input, width: 16);
    final tokenP = addOutput('token', width: tokenW);
    final tokenValidP = addOutput('token_valid');
    final doneP = addOutput('done');
    final busyP = addOutput('busy');

    final memRef = addInterface(
      WishboneInterface(WishboneConfig(addressWidth: aw, dataWidth: 32)),
      name: 'mem',
      role: PairRole.provider,
    );
    mem = memRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');
    final start = input('start');
    final one = Const(1, width: 16);

    final state = Logic(name: 'state', width: 4);
    final hc = Logic(name: 'hc', width: 16); // hidden/gamma load count
    final scc = Logic(name: 'scc', width: 16); // scale load count
    final k = Logic(name: 'k', width: 16); // norm element index
    final j = Logic(name: 'j', width: 16); // feed index
    final row = Logic(name: 'row', width: 16); // logit row index
    final bestKey = Logic(name: 'best_key', width: 16);
    final bestIdx = Logic(name: 'best_idx', width: tokenW);

    final hbuf = [for (var i = 0; i < H; i++) Logic(name: 'hb$i', width: 16)];
    final gbuf = [for (var i = 0; i < H; i++) Logic(name: 'gb$i', width: 16)];
    final fbuf = [for (var i = 0; i < H; i++) Logic(name: 'fb$i', width: 16)];
    final scbuf = [
      for (var i = 0; i < vocab; i++) Logic(name: 'scb$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 4));
    final normPhase = st(_nAcc) | st(_nComp) | st(_nWait) | st(_nNorm);

    final rms = LoomRmsNorm(recipIterations: recipIterations);
    rms.input('clk').srcConnection! <= clk;
    rms.input('reset').srcConnection! <= reset | ~normPhase;
    rms.input('acc_en').srcConnection! <= st(_nAcc);
    rms.input('compute').srcConnection! <= st(_nComp);
    rms.input('norm_en').srcConnection! <= st(_nNorm);
    rms.input('x_in').srcConnection! <= _sel(k, hbuf);
    rms.input('gamma_in').srcConnection! <= _sel(k, gbuf);
    rms.input('eps').srcConnection! <= input('eps');
    rms.input('inv_n').srcConnection! <= input('inv_n');

    final ls = LoomLinearSeq(
      maxColTiles: colTiles,
      maxRowBlocks: rowBlocks,
      recipIterations: recipIterations,
    );
    ls.input('clk').srcConnection! <= clk;
    ls.input('reset').srcConnection! <= reset;
    ls.input('start').srcConnection! <= st(_lStart);
    ls.input('col_tiles').srcConnection! <= Const(colTiles, width: 16);
    ls.input('row_blocks').srcConnection! <= Const(rowBlocks, width: 16);
    ls.input('weight_base').srcConnection! <= input('wb_cls');
    ls.input('x_en').srcConnection! <= st(_lFeed) & j.lt(Const(H, width: 16));
    ls.input('x_in').srcConnection! <= _sel(j, fbuf);
    ls.input('rs_en').srcConnection! <=
        st(_lFeed) & j.lt(Const(vocab, width: 16));
    ls.input('rs_in').srcConnection! <= _sel(j, scbuf);

    mem.cyc <= ls.output('mem_CYC');
    mem.stb <= ls.output('mem_STB');
    mem.we <= ls.output('mem_WE');
    mem.adr <= ls.output('mem_ADR');
    mem.datMosi <= ls.output('mem_DAT_MOSI');
    mem.sel <= ls.output('mem_SEL');
    ls.input('mem_ACK').srcConnection! <= mem.ack;
    ls.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    // Monotonic unsigned key for the packed fp16 logit: negative -> invert all
    // bits. Non-negative -> set the sign bit. Ordering is preserved so a plain
    // unsigned compare finds the max.
    final y = ls.output('y');
    final key = mux(
      y[15],
      ~y,
      y | Const(0x8000, width: 16),
    ).named('argmax_key');

    tokenP <= bestIdx;
    tokenValidP <= st(_fin);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('x_en') & hc.eq(Const(i, width: 16)),
          then: [hbuf[i] < input('x_in'), gbuf[i] < input('gamma_in')],
        ),
      for (var i = 0; i < vocab; i++)
        If(
          st(_load) & input('sc_en') & scc.eq(Const(i, width: 16)),
          then: [scbuf[i] < input('sc_in')],
        ),
    ];
    final normWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_nNorm) & rms.output('y_valid') & k.eq(Const(i, width: 16)),
          then: [fbuf[i] < rms.output('y')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          hc < Const(0, width: 16),
          scc < Const(0, width: 16),
          k < Const(0, width: 16),
          j < Const(0, width: 16),
          row < Const(0, width: 16),
          bestKey < Const(0, width: 16),
          bestIdx < Const(0, width: tokenW),
          for (final b in hbuf) b < Const(0, width: 16),
          for (final b in gbuf) b < Const(0, width: 16),
          for (final b in fbuf) b < Const(0, width: 16),
          for (final b in scbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...normWrites,
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('x_en'), then: [hc < hc + one]),
              If(input('sc_en'), then: [scc < scc + one]),
              If(
                start,
                then: [k < Const(0, width: 16), state < Const(_nAcc, width: 4)],
              ),
            ]),
            CaseItem(Const(_nAcc, width: 4), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_nComp, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_nComp, width: 4), [
              state < Const(_nWait, width: 4),
            ]),
            CaseItem(Const(_nWait, width: 4), [
              If(
                rms.output('ready'),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_nNorm, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_nNorm, width: 4), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_lFeed, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_lFeed, width: 4), [
              If(
                j.eq(Const(feedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  row < Const(0, width: 16),
                  bestKey < Const(0, width: 16),
                  bestIdx < Const(0, width: tokenW),
                  state < Const(_lStart, width: 4),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_lStart, width: 4), [
              state < Const(_lRun, width: 4),
            ]),
            CaseItem(Const(_lRun, width: 4), [
              If(
                ls.output('y_valid') & row.lt(Const(vocab, width: 16)),
                then: [
                  If(
                    key.gt(bestKey),
                    then: [bestKey < key, bestIdx < row.getRange(0, tokenW)],
                  ),
                  row < row + one,
                ],
              ),
              If(ls.output('done'), then: [state < Const(_fin, width: 4)]),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              hc < Const(0, width: 16),
              scc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
