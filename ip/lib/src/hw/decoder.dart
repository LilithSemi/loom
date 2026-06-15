library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'decoder_layer.dart';

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

/// The multi-layer decoder: runs ONE reused [LoomDecoderLayer] over an on-chip
/// resident `hidden[H]` for `numLayers` layers, carrying hidden forward layer to
/// layer, mirroring [GoldenRunner]'s `for l in layers` loop (runner.dart
/// ~78-147). The single reused compute datapath is possible because the attn
/// block's KV cache is layer-indexed (each layer keeps its own slab), so the
/// layers do not collide.
///
/// Per-layer parameters are streamed IN from the host, layer by layer (the same
/// load protocol [LoomDecoderLayer] exposes, sequenced `numLayers` times):
///   * Load the INITIAL hidden once (on `x_en`), plus layer 0's gammas/scales;
///     set layer 0's weight bases. Pulse `start` -> layer 0 runs, its output is
///     captured back into the resident hidden.
///   * The decoder returns to LOAD (`busy` low) between layers. Stream the next
///     layer's gammas/scales, set its weight bases, pulse `start` again. Do NOT
///     drive `x_en` for layers > 0 - hidden is resident and carried forward.
///   * After the last layer, the final hidden streams out on `h_out`/`h_valid`
///     and `done` pulses.
///
/// `pos` and cos/sin are the same across all layers of a token. The weight
/// bases, gammas, and scales differ per layer. Baking these as generated
/// constants / flash reads is a later SoC step. This module stays generic.
class LoomDecoder extends BridgeModule {
  static const int _load = 0;
  static const int _lfeed = 1;
  static const int _lstart = 2;
  static const int _lrun = 3;
  static const int _emit = 4;
  static const int _fin = 5;

  late final WishboneInterface mem;

  LoomDecoder({
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int intermediateSize,
    required int maxSeq,
    required int numLayers,
    int recipIterations = 4,
    String? name,
  }) : super('LoomDecoder', name: name ?? 'loom_decoder') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    if (numLayers < 1) {
      throw ArgumentError.value(numLayers, 'numLayers', 'must be >= 1');
    }
    const aw = 32;
    final H = hidden;
    final iSize = intermediateSize;
    final hd = headDim;
    final half = hd ~/ 2;
    final qDim = numHeads * hd;
    final kvDim = numKvHeads * hd;
    final posW = maxSeq.bitLength;
    final hasLayer = numLayers > 1;
    final layerW = numLayers.bitLength;
    final feedMax = [H, qDim, kvDim, iSize].reduce((a, b) => a > b ? a : b);

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('pos', PortDirection.input, width: posW);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('wb_q', PortDirection.input, width: aw);
    createPort('wb_k', PortDirection.input, width: aw);
    createPort('wb_v', PortDirection.input, width: aw);
    createPort('wb_o', PortDirection.input, width: aw);
    createPort('wb_gate', PortDirection.input, width: aw);
    createPort('wb_up', PortDirection.input, width: aw);
    createPort('wb_down', PortDirection.input, width: aw);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('g_en', PortDirection.input);
    createPort('iga_in', PortDirection.input, width: 16);
    createPort('pga_in', PortDirection.input, width: 16);
    createPort('sq_en', PortDirection.input);
    createPort('sq_in', PortDirection.input, width: 16);
    createPort('sk_en', PortDirection.input);
    createPort('sk_in', PortDirection.input, width: 16);
    createPort('sv_en', PortDirection.input);
    createPort('sv_in', PortDirection.input, width: 16);
    createPort('so_en', PortDirection.input);
    createPort('so_in', PortDirection.input, width: 16);
    createPort('sg_en', PortDirection.input);
    createPort('sg_in', PortDirection.input, width: 16);
    createPort('su_en', PortDirection.input);
    createPort('su_in', PortDirection.input, width: 16);
    createPort('sd_en', PortDirection.input);
    createPort('sd_in', PortDirection.input, width: 16);
    for (var jj = 0; jj < half; jj++) {
      createPort('cos_q$jj', PortDirection.input, width: 16);
      createPort('sin_q$jj', PortDirection.input, width: 16);
      createPort('cos_k$jj', PortDirection.input, width: 16);
      createPort('sin_k$jj', PortDirection.input, width: 16);
    }
    final hOutP = addOutput('h_out', width: 16);
    final hValidP = addOutput('h_valid');
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

    final state = Logic(name: 'state', width: 3);
    final hc = Logic(name: 'hc', width: 16); // hidden load count
    final gc = Logic(name: 'gc', width: 16); // gamma load count
    final sqc = Logic(name: 'sqc', width: 16);
    final skc = Logic(name: 'skc', width: 16);
    final svc = Logic(name: 'svc', width: 16);
    final soc = Logic(name: 'soc', width: 16);
    final sgc = Logic(name: 'sgc', width: 16);
    final suc = Logic(name: 'suc', width: 16);
    final sdc = Logic(name: 'sdc', width: 16);
    final lReg = Logic(name: 'l_reg', width: 16); // current layer index
    final j = Logic(name: 'j', width: 16); // feed index
    final rc = Logic(name: 'rc', width: 16); // layer-output capture index
    final ec = Logic(name: 'ec', width: 16); // emit index

    final hbuf = [for (var i = 0; i < H; i++) Logic(name: 'hb$i', width: 16)];
    final igabuf = [
      for (var i = 0; i < H; i++) Logic(name: 'iga$i', width: 16),
    ];
    final pgabuf = [
      for (var i = 0; i < H; i++) Logic(name: 'pga$i', width: 16),
    ];
    final sqbuf = [
      for (var i = 0; i < qDim; i++) Logic(name: 'sqb$i', width: 16),
    ];
    final skbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'skb$i', width: 16),
    ];
    final svbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'svb$i', width: 16),
    ];
    final sobuf = [for (var i = 0; i < H; i++) Logic(name: 'sob$i', width: 16)];
    final sgbuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'sgb$i', width: 16),
    ];
    final subuf = [
      for (var i = 0; i < iSize; i++) Logic(name: 'sub$i', width: 16),
    ];
    final sdbuf = [for (var i = 0; i < H; i++) Logic(name: 'sdb$i', width: 16)];

    Logic st(int s) => state.eq(Const(s, width: 3));

    final dl = LoomDecoderLayer(
      hidden: H,
      numHeads: numHeads,
      numKvHeads: numKvHeads,
      headDim: hd,
      intermediateSize: iSize,
      maxSeq: maxSeq,
      numLayers: numLayers,
      recipIterations: recipIterations,
    );
    dl.input('clk').srcConnection! <= clk;
    dl.input('reset').srcConnection! <= reset;
    dl.input('start').srcConnection! <= st(_lstart);
    dl.input('pos').srcConnection! <= input('pos');
    if (hasLayer) {
      dl.input('layer').srcConnection! <= lReg.getRange(0, layerW);
    }
    dl.input('inv_n').srcConnection! <= input('inv_n');
    dl.input('eps').srcConnection! <= input('eps');
    dl.input('wb_q').srcConnection! <= input('wb_q');
    dl.input('wb_k').srcConnection! <= input('wb_k');
    dl.input('wb_v').srcConnection! <= input('wb_v');
    dl.input('wb_o').srcConnection! <= input('wb_o');
    dl.input('wb_gate').srcConnection! <= input('wb_gate');
    dl.input('wb_up').srcConnection! <= input('wb_up');
    dl.input('wb_down').srcConnection! <= input('wb_down');
    dl.input('x_en').srcConnection! <= st(_lfeed) & j.lt(Const(H, width: 16));
    dl.input('x_in').srcConnection! <= _sel(j, hbuf);
    dl.input('iga_in').srcConnection! <= _sel(j, igabuf);
    dl.input('pga_in').srcConnection! <= _sel(j, pgabuf);
    dl.input('sq_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(qDim, width: 16));
    dl.input('sq_in').srcConnection! <= _sel(j, sqbuf);
    dl.input('sk_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(kvDim, width: 16));
    dl.input('sk_in').srcConnection! <= _sel(j, skbuf);
    dl.input('sv_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(kvDim, width: 16));
    dl.input('sv_in').srcConnection! <= _sel(j, svbuf);
    dl.input('so_en').srcConnection! <= st(_lfeed) & j.lt(Const(H, width: 16));
    dl.input('so_in').srcConnection! <= _sel(j, sobuf);
    dl.input('sg_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(iSize, width: 16));
    dl.input('sg_in').srcConnection! <= _sel(j, sgbuf);
    dl.input('su_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(iSize, width: 16));
    dl.input('su_in').srcConnection! <= _sel(j, subuf);
    dl.input('sd_en').srcConnection! <= st(_lfeed) & j.lt(Const(H, width: 16));
    dl.input('sd_in').srcConnection! <= _sel(j, sdbuf);
    for (var jj = 0; jj < half; jj++) {
      dl.input('cos_q$jj').srcConnection! <= input('cos_q$jj');
      dl.input('sin_q$jj').srcConnection! <= input('sin_q$jj');
      dl.input('cos_k$jj').srcConnection! <= input('cos_k$jj');
      dl.input('sin_k$jj').srcConnection! <= input('sin_k$jj');
    }

    // Single weight master forwarded up to the top.
    mem.cyc <= dl.output('mem_CYC');
    mem.stb <= dl.output('mem_STB');
    mem.we <= dl.output('mem_WE');
    mem.adr <= dl.output('mem_ADR');
    mem.datMosi <= dl.output('mem_DAT_MOSI');
    mem.sel <= dl.output('mem_SEL');
    dl.input('mem_ACK').srcConnection! <= mem.ack;
    dl.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    hOutP <= _sel(ec, hbuf);
    hValidP <= st(_emit);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('x_en') & hc.eq(Const(i, width: 16)),
          then: [hbuf[i] < input('x_in')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('g_en') & gc.eq(Const(i, width: 16)),
          then: [igabuf[i] < input('iga_in'), pgabuf[i] < input('pga_in')],
        ),
      for (var i = 0; i < qDim; i++)
        If(
          st(_load) & input('sq_en') & sqc.eq(Const(i, width: 16)),
          then: [sqbuf[i] < input('sq_in')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_load) & input('sk_en') & skc.eq(Const(i, width: 16)),
          then: [skbuf[i] < input('sk_in')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_load) & input('sv_en') & svc.eq(Const(i, width: 16)),
          then: [svbuf[i] < input('sv_in')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('so_en') & soc.eq(Const(i, width: 16)),
          then: [sobuf[i] < input('so_in')],
        ),
      for (var i = 0; i < iSize; i++)
        If(
          st(_load) & input('sg_en') & sgc.eq(Const(i, width: 16)),
          then: [sgbuf[i] < input('sg_in')],
        ),
      for (var i = 0; i < iSize; i++)
        If(
          st(_load) & input('su_en') & suc.eq(Const(i, width: 16)),
          then: [subuf[i] < input('su_in')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('sd_en') & sdc.eq(Const(i, width: 16)),
          then: [sdbuf[i] < input('sd_in')],
        ),
    ];

    // Capture this layer's output back into the resident hidden buffer.
    final captureWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_lrun) & dl.output('h_valid') & rc.eq(Const(i, width: 16)),
          then: [hbuf[i] < dl.output('h_out')],
        ),
    ];

    // Fresh Conditionals each call - a Conditional object may only be
    // registered once in the Sequential tree.
    List<Conditional> resetLoadCounts() => [
      hc < Const(0, width: 16),
      gc < Const(0, width: 16),
      sqc < Const(0, width: 16),
      skc < Const(0, width: 16),
      svc < Const(0, width: 16),
      soc < Const(0, width: 16),
      sgc < Const(0, width: 16),
      suc < Const(0, width: 16),
      sdc < Const(0, width: 16),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 3),
          ...resetLoadCounts(),
          lReg < Const(0, width: 16),
          j < Const(0, width: 16),
          rc < Const(0, width: 16),
          ec < Const(0, width: 16),
          for (final b in hbuf) b < Const(0, width: 16),
          for (final b in igabuf) b < Const(0, width: 16),
          for (final b in pgabuf) b < Const(0, width: 16),
          for (final b in sqbuf) b < Const(0, width: 16),
          for (final b in skbuf) b < Const(0, width: 16),
          for (final b in svbuf) b < Const(0, width: 16),
          for (final b in sobuf) b < Const(0, width: 16),
          for (final b in sgbuf) b < Const(0, width: 16),
          for (final b in subuf) b < Const(0, width: 16),
          for (final b in sdbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...captureWrites,
          Case(state, [
            CaseItem(Const(_load, width: 3), [
              If(input('x_en'), then: [hc < hc + one]),
              If(input('g_en'), then: [gc < gc + one]),
              If(input('sq_en'), then: [sqc < sqc + one]),
              If(input('sk_en'), then: [skc < skc + one]),
              If(input('sv_en'), then: [svc < svc + one]),
              If(input('so_en'), then: [soc < soc + one]),
              If(input('sg_en'), then: [sgc < sgc + one]),
              If(input('su_en'), then: [suc < suc + one]),
              If(input('sd_en'), then: [sdc < sdc + one]),
              If(
                start,
                then: [
                  j < Const(0, width: 16),
                  rc < Const(0, width: 16),
                  state < Const(_lfeed, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_lfeed, width: 3), [
              If(
                j.eq(Const(feedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_lstart, width: 3),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_lstart, width: 3), [
              state < Const(_lrun, width: 3),
            ]),
            CaseItem(Const(_lrun, width: 3), [
              If(dl.output('h_valid'), then: [rc < rc + one]),
              If(
                dl.output('done'),
                then: [
                  If(
                    lReg.eq(Const(numLayers - 1, width: 16)),
                    then: [
                      ec < Const(0, width: 16),
                      state < Const(_emit, width: 3),
                    ],
                    orElse: [
                      lReg < lReg + one,
                      ...resetLoadCounts(),
                      state < Const(_load, width: 3),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_emit, width: 3), [
              If(
                ec.eq(Const(H - 1, width: 16)),
                then: [state < Const(_fin, width: 3)],
                orElse: [ec < ec + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 3), [
              state < Const(_load, width: 3),
              lReg < Const(0, width: 16),
              ...resetLoadCounts(),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
