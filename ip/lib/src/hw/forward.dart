library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'decoder.dart';
import 'lm_head.dart';

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

/// The whole model forward for ONE token, mirroring [GoldenRunner.forward]:
/// `embed(token) -> LoomDecoder (all layers) -> LoomLmHead -> next token`.
///
///   * embed: read the `hidden[H]` embedding row from `mem`
///     (`wb_embed + (token*H + i)` words, one fp16 per 32-bit word) into an
///     on-chip buffer, via a small Wishbone read FSM.
///   * run [LoomDecoder] over that hidden for all `numLayers` layers (the
///     decoder carries hidden forward and owns the per-layer KV caches). This
///     module drives the decoder's per-layer load protocol from buffered host
///     params, injecting the embedded hidden as the layer-0 input.
///   * feed the decoder's final hidden into [LoomLmHead] -> next-token id.
///
/// One top-level `mem` master is shared across the three phases (embed reader,
/// decoder, lm_head), which never overlap, via a state-driven mux.
///
/// All per-layer params (7 weight bases, input/post gammas, all row scales),
/// plus the final-norm gamma, Wcls scales, and the embed/Wcls bases, are
/// streamed IN from the host during LOAD and buffered (baking them as generated
/// constants / flash reads is the later SoC step). Load protocol (state LOAD):
/// stream all layers' gammas on `ig_en` (contiguous, layer-major) and each scale
/// table on its enable. The per-layer weight bases on `wbl_en` (7 per layer:
/// q,k,v,o,gate,up,down). The final-norm gamma on `fg_en`. The Wcls scales on
/// `cs_en`. Hold `in_token`, `wb_embed`, `wb_cls`, cos/sin, `pos`, `inv_n`,
/// `eps`. Pulse `start`. The next token id is held on `token` with
/// `token_valid`/`done` at the end.
class LoomForward extends BridgeModule {
  static const int _load = 0;
  static const int _embed = 1;
  static const int _dfeed = 2;
  static const int _dstart = 3;
  static const int _drun = 4;
  static const int _lfeed = 5;
  static const int _lstart = 6;
  static const int _lrun = 7;
  static const int _fin = 8;

  late final WishboneInterface mem;

  LoomForward({
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int intermediateSize,
    required int maxSeq,
    required int numLayers,
    required int vocab,
    int recipIterations = 4,
    String? name,
  }) : super('LoomForward', name: name ?? 'loom_forward') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    const aw = 32;
    final H = hidden;
    final iSize = intermediateSize;
    final hd = headDim;
    final half = hd ~/ 2;
    final qDim = numHeads * hd;
    final kvDim = numKvHeads * hd;
    final posW = maxSeq.bitLength;
    final tokenW = vocab.bitLength;
    final feedMaxDec = [H, qDim, kvDim, iSize].reduce((a, b) => a > b ? a : b);
    final feedMaxLm = H > vocab ? H : vocab;
    final lastLayer = numLayers - 1;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('pos', PortDirection.input, width: posW);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('in_token', PortDirection.input, width: tokenW);
    createPort('wb_embed', PortDirection.input, width: aw);
    createPort('wb_cls', PortDirection.input, width: aw);
    createPort('ig_en', PortDirection.input);
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
    createPort('wbl_en', PortDirection.input);
    createPort('wbl_in', PortDirection.input, width: aw);
    createPort('fg_en', PortDirection.input);
    createPort('fg_in', PortDirection.input, width: 16);
    createPort('cs_en', PortDirection.input);
    createPort('cs_in', PortDirection.input, width: 16);
    for (var jj = 0; jj < half; jj++) {
      createPort('cos_q$jj', PortDirection.input, width: 16);
      createPort('sin_q$jj', PortDirection.input, width: 16);
      createPort('cos_k$jj', PortDirection.input, width: 16);
      createPort('sin_k$jj', PortDirection.input, width: 16);
    }
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
    final igc = Logic(name: 'igc', width: 16);
    final sqc = Logic(name: 'sqc', width: 16);
    final skc = Logic(name: 'skc', width: 16);
    final svc = Logic(name: 'svc', width: 16);
    final soc = Logic(name: 'soc', width: 16);
    final sgc = Logic(name: 'sgc', width: 16);
    final suc = Logic(name: 'suc', width: 16);
    final sdc = Logic(name: 'sdc', width: 16);
    final wblc = Logic(name: 'wblc', width: 16);
    final fgc = Logic(name: 'fgc', width: 16);
    final csc = Logic(name: 'csc', width: 16);
    final tokReg = Logic(name: 'tok_reg', width: 16); // input token latched
    final ei = Logic(name: 'ei', width: 16); // embed read index
    final lc = Logic(name: 'lc', width: 16); // decoder layer index
    final j = Logic(name: 'j', width: 16); // feed index
    final rc = Logic(name: 'rc', width: 16); // final-hidden capture index
    final otok = Logic(name: 'otok', width: tokenW); // chosen token

    final hidbuf = [for (var i = 0; i < H; i++) Logic(name: 'hb$i', width: 16)];
    final lhbuf = [for (var i = 0; i < H; i++) Logic(name: 'lh$i', width: 16)];
    final igaFlat = [
      for (var i = 0; i < numLayers * H; i++) Logic(name: 'iga$i', width: 16),
    ];
    final pgaFlat = [
      for (var i = 0; i < numLayers * H; i++) Logic(name: 'pga$i', width: 16),
    ];
    final sqFlat = [
      for (var i = 0; i < numLayers * qDim; i++) Logic(name: 'sq$i', width: 16),
    ];
    final skFlat = [
      for (var i = 0; i < numLayers * kvDim; i++)
        Logic(name: 'sk$i', width: 16),
    ];
    final svFlat = [
      for (var i = 0; i < numLayers * kvDim; i++)
        Logic(name: 'sv$i', width: 16),
    ];
    final soFlat = [
      for (var i = 0; i < numLayers * H; i++) Logic(name: 'so$i', width: 16),
    ];
    final sgFlat = [
      for (var i = 0; i < numLayers * iSize; i++)
        Logic(name: 'sg$i', width: 16),
    ];
    final suFlat = [
      for (var i = 0; i < numLayers * iSize; i++)
        Logic(name: 'su$i', width: 16),
    ];
    final sdFlat = [
      for (var i = 0; i < numLayers * H; i++) Logic(name: 'sd$i', width: 16),
    ];
    final wbFlat = [
      for (var i = 0; i < numLayers * 7; i++) Logic(name: 'wb$i', width: aw),
    ];
    final fgFlat = [for (var i = 0; i < H; i++) Logic(name: 'fg$i', width: 16)];
    final csFlat = [
      for (var i = 0; i < vocab; i++) Logic(name: 'cs$i', width: 16),
    ];

    Logic st(int s) => state.eq(Const(s, width: 4));
    Logic mul16(Logic a, int b) => (a * Const(b, width: 16)).getRange(0, 16);
    Logic wbSel(int off) => _sel(mul16(lc, 7) + Const(off, width: 16), wbFlat);

    final embActive = st(_embed);
    final decActive = st(_dstart) | st(_drun);
    final lmActive = st(_lstart) | st(_lrun);

    final dec = LoomDecoder(
      hidden: H,
      numHeads: numHeads,
      numKvHeads: numKvHeads,
      headDim: hd,
      intermediateSize: iSize,
      maxSeq: maxSeq,
      numLayers: numLayers,
      recipIterations: recipIterations,
    );
    dec.input('clk').srcConnection! <= clk;
    dec.input('reset').srcConnection! <= reset;
    dec.input('start').srcConnection! <= st(_dstart);
    dec.input('pos').srcConnection! <= input('pos');
    dec.input('inv_n').srcConnection! <= input('inv_n');
    dec.input('eps').srcConnection! <= input('eps');
    dec.input('wb_q').srcConnection! <= wbSel(0);
    dec.input('wb_k').srcConnection! <= wbSel(1);
    dec.input('wb_v').srcConnection! <= wbSel(2);
    dec.input('wb_o').srcConnection! <= wbSel(3);
    dec.input('wb_gate').srcConnection! <= wbSel(4);
    dec.input('wb_up').srcConnection! <= wbSel(5);
    dec.input('wb_down').srcConnection! <= wbSel(6);
    // Layer-0 hidden comes from embed. Layers > 0 keep the decoder's resident
    // hidden (do not drive x_en).
    dec.input('x_en').srcConnection! <=
        st(_dfeed) & lc.eq(Const(0, width: 16)) & j.lt(Const(H, width: 16));
    dec.input('x_in').srcConnection! <= _sel(j, hidbuf);
    dec.input('g_en').srcConnection! <= st(_dfeed) & j.lt(Const(H, width: 16));
    dec.input('iga_in').srcConnection! <= _sel(mul16(lc, H) + j, igaFlat);
    dec.input('pga_in').srcConnection! <= _sel(mul16(lc, H) + j, pgaFlat);
    dec.input('sq_en').srcConnection! <=
        st(_dfeed) & j.lt(Const(qDim, width: 16));
    dec.input('sq_in').srcConnection! <= _sel(mul16(lc, qDim) + j, sqFlat);
    dec.input('sk_en').srcConnection! <=
        st(_dfeed) & j.lt(Const(kvDim, width: 16));
    dec.input('sk_in').srcConnection! <= _sel(mul16(lc, kvDim) + j, skFlat);
    dec.input('sv_en').srcConnection! <=
        st(_dfeed) & j.lt(Const(kvDim, width: 16));
    dec.input('sv_in').srcConnection! <= _sel(mul16(lc, kvDim) + j, svFlat);
    dec.input('so_en').srcConnection! <= st(_dfeed) & j.lt(Const(H, width: 16));
    dec.input('so_in').srcConnection! <= _sel(mul16(lc, H) + j, soFlat);
    dec.input('sg_en').srcConnection! <=
        st(_dfeed) & j.lt(Const(iSize, width: 16));
    dec.input('sg_in').srcConnection! <= _sel(mul16(lc, iSize) + j, sgFlat);
    dec.input('su_en').srcConnection! <=
        st(_dfeed) & j.lt(Const(iSize, width: 16));
    dec.input('su_in').srcConnection! <= _sel(mul16(lc, iSize) + j, suFlat);
    dec.input('sd_en').srcConnection! <= st(_dfeed) & j.lt(Const(H, width: 16));
    dec.input('sd_in').srcConnection! <= _sel(mul16(lc, H) + j, sdFlat);
    for (var jj = 0; jj < half; jj++) {
      dec.input('cos_q$jj').srcConnection! <= input('cos_q$jj');
      dec.input('sin_q$jj').srcConnection! <= input('sin_q$jj');
      dec.input('cos_k$jj').srcConnection! <= input('cos_k$jj');
      dec.input('sin_k$jj').srcConnection! <= input('sin_k$jj');
    }

    final lm = LoomLmHead(
      hidden: H,
      vocab: vocab,
      recipIterations: recipIterations,
    );
    lm.input('clk').srcConnection! <= clk;
    lm.input('reset').srcConnection! <= reset;
    lm.input('start').srcConnection! <= st(_lstart);
    lm.input('inv_n').srcConnection! <= input('inv_n');
    lm.input('eps').srcConnection! <= input('eps');
    lm.input('wb_cls').srcConnection! <= input('wb_cls');
    lm.input('x_en').srcConnection! <= st(_lfeed) & j.lt(Const(H, width: 16));
    lm.input('x_in').srcConnection! <= _sel(j, lhbuf);
    lm.input('gamma_in').srcConnection! <= _sel(j, fgFlat);
    lm.input('sc_en').srcConnection! <=
        st(_lfeed) & j.lt(Const(vocab, width: 16));
    lm.input('sc_in').srcConnection! <= _sel(j, csFlat);

    final embIdx = mul16(tokReg, H) + ei;
    final embAdr =
        (input('wb_embed') +
        (embIdx.zeroExtend(32) * Const(4, width: 32)).getRange(0, 32));
    final embStb = st(_embed) & ei.lt(Const(H, width: 16));

    mem.cyc <=
        mux(
          embActive,
          embStb,
          mux(lmActive, lm.output('mem_CYC'), dec.output('mem_CYC')),
        );
    mem.stb <=
        mux(
          embActive,
          embStb,
          mux(lmActive, lm.output('mem_STB'), dec.output('mem_STB')),
        );
    mem.we <=
        mux(
          embActive,
          Const(0),
          mux(lmActive, lm.output('mem_WE'), dec.output('mem_WE')),
        );
    mem.adr <=
        mux(
          embActive,
          embAdr,
          mux(lmActive, lm.output('mem_ADR'), dec.output('mem_ADR')),
        );
    mem.datMosi <=
        mux(
          embActive,
          Const(0, width: 32),
          mux(lmActive, lm.output('mem_DAT_MOSI'), dec.output('mem_DAT_MOSI')),
        );
    mem.sel <=
        mux(
          embActive,
          Const(0xF, width: mem.sel.width),
          mux(lmActive, lm.output('mem_SEL'), dec.output('mem_SEL')),
        );
    dec.input('mem_ACK').srcConnection! <= mem.ack & decActive;
    dec.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;
    lm.input('mem_ACK').srcConnection! <= mem.ack & lmActive;
    lm.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    tokenP <= otok;
    tokenValidP <= st(_fin);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < numLayers * H; i++)
        If(
          st(_load) & input('ig_en') & igc.eq(Const(i, width: 16)),
          then: [igaFlat[i] < input('iga_in'), pgaFlat[i] < input('pga_in')],
        ),
      for (var i = 0; i < numLayers * qDim; i++)
        If(
          st(_load) & input('sq_en') & sqc.eq(Const(i, width: 16)),
          then: [sqFlat[i] < input('sq_in')],
        ),
      for (var i = 0; i < numLayers * kvDim; i++)
        If(
          st(_load) & input('sk_en') & skc.eq(Const(i, width: 16)),
          then: [skFlat[i] < input('sk_in')],
        ),
      for (var i = 0; i < numLayers * kvDim; i++)
        If(
          st(_load) & input('sv_en') & svc.eq(Const(i, width: 16)),
          then: [svFlat[i] < input('sv_in')],
        ),
      for (var i = 0; i < numLayers * H; i++)
        If(
          st(_load) & input('so_en') & soc.eq(Const(i, width: 16)),
          then: [soFlat[i] < input('so_in')],
        ),
      for (var i = 0; i < numLayers * iSize; i++)
        If(
          st(_load) & input('sg_en') & sgc.eq(Const(i, width: 16)),
          then: [sgFlat[i] < input('sg_in')],
        ),
      for (var i = 0; i < numLayers * iSize; i++)
        If(
          st(_load) & input('su_en') & suc.eq(Const(i, width: 16)),
          then: [suFlat[i] < input('su_in')],
        ),
      for (var i = 0; i < numLayers * H; i++)
        If(
          st(_load) & input('sd_en') & sdc.eq(Const(i, width: 16)),
          then: [sdFlat[i] < input('sd_in')],
        ),
      for (var i = 0; i < numLayers * 7; i++)
        If(
          st(_load) & input('wbl_en') & wblc.eq(Const(i, width: 16)),
          then: [wbFlat[i] < input('wbl_in')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('fg_en') & fgc.eq(Const(i, width: 16)),
          then: [fgFlat[i] < input('fg_in')],
        ),
      for (var i = 0; i < vocab; i++)
        If(
          st(_load) & input('cs_en') & csc.eq(Const(i, width: 16)),
          then: [csFlat[i] < input('cs_in')],
        ),
    ];

    // Embed row capture, and final-hidden capture from the decoder.
    final captureWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_embed) & mem.ack & ei.eq(Const(i, width: 16)),
          then: [hidbuf[i] < mem.datMiso.getRange(0, 16)],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_drun) &
              lc.eq(Const(lastLayer, width: 16)) &
              dec.output('h_valid') &
              rc.eq(Const(i, width: 16)),
          then: [lhbuf[i] < dec.output('h_out')],
        ),
    ];

    List<Conditional> resetLoadCounts() => [
      igc < Const(0, width: 16),
      sqc < Const(0, width: 16),
      skc < Const(0, width: 16),
      svc < Const(0, width: 16),
      soc < Const(0, width: 16),
      sgc < Const(0, width: 16),
      suc < Const(0, width: 16),
      sdc < Const(0, width: 16),
      wblc < Const(0, width: 16),
      fgc < Const(0, width: 16),
      csc < Const(0, width: 16),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          ...resetLoadCounts(),
          tokReg < Const(0, width: 16),
          ei < Const(0, width: 16),
          lc < Const(0, width: 16),
          j < Const(0, width: 16),
          rc < Const(0, width: 16),
          otok < Const(0, width: tokenW),
          for (final b in hidbuf) b < Const(0, width: 16),
          for (final b in lhbuf) b < Const(0, width: 16),
          for (final b in igaFlat) b < Const(0, width: 16),
          for (final b in pgaFlat) b < Const(0, width: 16),
          for (final b in sqFlat) b < Const(0, width: 16),
          for (final b in skFlat) b < Const(0, width: 16),
          for (final b in svFlat) b < Const(0, width: 16),
          for (final b in soFlat) b < Const(0, width: 16),
          for (final b in sgFlat) b < Const(0, width: 16),
          for (final b in suFlat) b < Const(0, width: 16),
          for (final b in sdFlat) b < Const(0, width: 16),
          for (final b in wbFlat) b < Const(0, width: aw),
          for (final b in fgFlat) b < Const(0, width: 16),
          for (final b in csFlat) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...captureWrites,
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('ig_en'), then: [igc < igc + one]),
              If(input('sq_en'), then: [sqc < sqc + one]),
              If(input('sk_en'), then: [skc < skc + one]),
              If(input('sv_en'), then: [svc < svc + one]),
              If(input('so_en'), then: [soc < soc + one]),
              If(input('sg_en'), then: [sgc < sgc + one]),
              If(input('su_en'), then: [suc < suc + one]),
              If(input('sd_en'), then: [sdc < sdc + one]),
              If(input('wbl_en'), then: [wblc < wblc + one]),
              If(input('fg_en'), then: [fgc < fgc + one]),
              If(input('cs_en'), then: [csc < csc + one]),
              If(
                start,
                then: [
                  tokReg < input('in_token').zeroExtend(16),
                  ei < Const(0, width: 16),
                  lc < Const(0, width: 16),
                  state < Const(_embed, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_embed, width: 4), [
              If(
                mem.ack,
                then: [
                  If(
                    ei.eq(Const(H - 1, width: 16)),
                    then: [
                      ei < Const(0, width: 16),
                      j < Const(0, width: 16),
                      state < Const(_dfeed, width: 4),
                    ],
                    orElse: [ei < ei + one],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_dfeed, width: 4), [
              If(
                j.eq(Const(feedMaxDec - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  rc < Const(0, width: 16),
                  state < Const(_dstart, width: 4),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_dstart, width: 4), [
              state < Const(_drun, width: 4),
            ]),
            CaseItem(Const(_drun, width: 4), [
              If(
                lc.eq(Const(lastLayer, width: 16)),
                then: [
                  If(dec.output('h_valid'), then: [rc < rc + one]),
                  If(
                    dec.output('done'),
                    then: [
                      j < Const(0, width: 16),
                      state < Const(_lfeed, width: 4),
                    ],
                  ),
                ],
                orElse: [
                  // Non-final layer complete -> decoder back in LOAD.
                  If(
                    ~dec.output('busy'),
                    then: [
                      lc < lc + one,
                      j < Const(0, width: 16),
                      state < Const(_dfeed, width: 4),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_lfeed, width: 4), [
              If(
                j.eq(Const(feedMaxLm - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_lstart, width: 4),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_lstart, width: 4), [
              state < Const(_lrun, width: 4),
            ]),
            CaseItem(Const(_lrun, width: 4), [
              If(
                lm.output('done'),
                then: [
                  otok < lm.output('token'),
                  state < Const(_fin, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              ...resetLoadCounts(),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
