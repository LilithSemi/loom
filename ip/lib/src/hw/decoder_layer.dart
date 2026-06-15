library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'attn_block.dart';
import 'fp_residual.dart';
import 'fp_rmsnorm.dart';
import 'mlp_seq.dart';

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

/// One complete on-chip decoder layer for a token at position `t`, mirroring
/// [GoldenRunner]'s per-token layer (runner.dart lines 88-146):
///
///   1. Attention: hidden' = hidden + Wo @ attn(rmsNorm(hidden, iga), ...)
///      -> delegated wholesale to [LoomAttnBlock] (owns the persistent KV cache).
///   2. MLP (SwiGLU): n2 = rmsNorm(hidden', pga); d = Wd @ (silu(Wg@n2) * (Wu@n2));
///      hidden'' = hidden' + d  -> [LoomRmsNorm] + [LoomMlpSeq] + [LoomFpResidual].
///
/// The two weight-read masters ([LoomAttnBlock.mem] - itself an internal mux of
/// qkv_norm and o_proj - and [LoomMlpSeq.mem]) run in strictly disjoint phases
/// (attention fully completes before the MLP starts), so they are time-muxed
/// onto ONE top-level Wishbone `mem` provider by a state-driven mux. The
/// post-attn RMSNorm reads only from on-chip buffers (no master).
///
/// The post-attn [LoomRmsNorm] is single-shot (its accumulator and reciprocal
/// clear only on reset), so it is held in reset except during the norm phases,
/// guaranteeing a cleared accumulator each position (same gotcha handled in
/// [LoomAttnBlock] for qkv_norm).
///
/// Load protocol (state LOAD): stream `hidden`+`iga`(input-norm gamma)+`pga`
/// (post-attn-norm gamma) on `x_en`/`x_in`/`iga_in`/`pga_in`. The attention
/// weight-scale tables on `sq_en`/`sk_en`/`sv_en`/`so_en`. The MLP scale tables
/// on `sg_en`/`su_en`/`sd_en`. Hold `cos_q*`/`sin_q*` (1/sqrt(hd) folded) and
/// `cos_k*`/`sin_k*`, the seven weight bases, `inv_n` (=1/H), `eps`, `pos` (=t);
/// pulse `start`. The updated hidden streams out on `h_out`/`h_valid` (H values);
/// `done` pulses at the end. The KV cache is not reset between positions.
class LoomDecoderLayer extends BridgeModule {
  static const int _load = 0;
  static const int _aFeed = 1;
  static const int _aStart = 2;
  static const int _aRun = 3;
  static const int _mnAcc = 4;
  static const int _mnComp = 5;
  static const int _mnWait = 6;
  static const int _mnNorm = 7;
  static const int _mFeed = 8;
  static const int _mStart = 9;
  static const int _mRun = 10;
  static const int _emit = 11;
  static const int _fin = 12;

  late final WishboneInterface mem;

  LoomDecoderLayer({
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int intermediateSize,
    required int maxSeq,
    int numLayers = 1,
    int recipIterations = 4,
    String? name,
  }) : super('LoomDecoderLayer', name: name ?? 'loom_decoder_layer') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    const aw = 32;
    final H = hidden;
    final iSize = intermediateSize;
    final hd = headDim;
    final half = hd ~/ 2;
    // When part of a multi-layer decoder, a `layer` index selects the reused
    // attn_block's per-layer KV slab. With numLayers==1 there is no port.
    final hasLayer = numLayers > 1;
    final layerW = numLayers.bitLength;
    final qDim = numHeads * hd;
    final kvDim = numKvHeads * hd;
    final posW = maxSeq.bitLength;
    final aFeedMax = [H, qDim, kvDim].reduce((a, b) => a > b ? a : b);
    final mFeedMax = H > iSize ? H : iSize;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('pos', PortDirection.input, width: posW);
    if (hasLayer) createPort('layer', PortDirection.input, width: layerW);
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

    final state = Logic(name: 'state', width: 4);
    final xc = Logic(name: 'xc', width: 16); // hidden/iga/pga load count
    final sqc = Logic(name: 'sqc', width: 16);
    final skc = Logic(name: 'skc', width: 16);
    final svc = Logic(name: 'svc', width: 16);
    final soc = Logic(name: 'soc', width: 16);
    final sgc = Logic(name: 'sgc', width: 16);
    final suc = Logic(name: 'suc', width: 16);
    final sdc = Logic(name: 'sdc', width: 16);
    final j = Logic(name: 'j', width: 16); // shared feed index
    final k = Logic(name: 'k', width: 16); // norm element index
    final ac = Logic(name: 'ac', width: 16); // attn-out collect index
    final mc = Logic(name: 'mc', width: 16); // mlp-out collect index
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
    final h1buf = [for (var i = 0; i < H; i++) Logic(name: 'h1$i', width: 16)];
    final n2buf = [for (var i = 0; i < H; i++) Logic(name: 'n2$i', width: 16)];
    final dbuf = [for (var i = 0; i < H; i++) Logic(name: 'db$i', width: 16)];

    Logic st(int s) => state.eq(Const(s, width: 4));
    final normPhase = st(_mnAcc) | st(_mnComp) | st(_mnWait) | st(_mnNorm);

    final attn = LoomAttnBlock(
      hidden: H,
      numHeads: numHeads,
      numKvHeads: numKvHeads,
      headDim: hd,
      maxSeq: maxSeq,
      numLayers: numLayers,
      recipIterations: recipIterations,
    );
    attn.input('clk').srcConnection! <= clk;
    attn.input('reset').srcConnection! <= reset;
    attn.input('start').srcConnection! <= st(_aStart);
    attn.input('pos').srcConnection! <= input('pos');
    if (hasLayer) attn.input('layer').srcConnection! <= input('layer');
    attn.input('inv_n').srcConnection! <= input('inv_n');
    attn.input('eps').srcConnection! <= input('eps');
    attn.input('wb_q').srcConnection! <= input('wb_q');
    attn.input('wb_k').srcConnection! <= input('wb_k');
    attn.input('wb_v').srcConnection! <= input('wb_v');
    attn.input('wb_o').srcConnection! <= input('wb_o');
    attn.input('x_en').srcConnection! <= st(_aFeed) & j.lt(Const(H, width: 16));
    attn.input('x_in').srcConnection! <= _sel(j, hbuf);
    attn.input('gamma_in').srcConnection! <= _sel(j, igabuf);
    attn.input('sq_en').srcConnection! <=
        st(_aFeed) & j.lt(Const(qDim, width: 16));
    attn.input('sq_in').srcConnection! <= _sel(j, sqbuf);
    attn.input('sk_en').srcConnection! <=
        st(_aFeed) & j.lt(Const(kvDim, width: 16));
    attn.input('sk_in').srcConnection! <= _sel(j, skbuf);
    attn.input('sv_en').srcConnection! <=
        st(_aFeed) & j.lt(Const(kvDim, width: 16));
    attn.input('sv_in').srcConnection! <= _sel(j, svbuf);
    attn.input('so_en').srcConnection! <=
        st(_aFeed) & j.lt(Const(H, width: 16));
    attn.input('so_in').srcConnection! <= _sel(j, sobuf);
    for (var jj = 0; jj < half; jj++) {
      attn.input('cos_q$jj').srcConnection! <= input('cos_q$jj');
      attn.input('sin_q$jj').srcConnection! <= input('sin_q$jj');
      attn.input('cos_k$jj').srcConnection! <= input('cos_k$jj');
      attn.input('sin_k$jj').srcConnection! <= input('sin_k$jj');
    }

    final rms = LoomRmsNorm(recipIterations: recipIterations);
    rms.input('clk').srcConnection! <= clk;
    rms.input('reset').srcConnection! <= reset | ~normPhase;
    rms.input('acc_en').srcConnection! <= st(_mnAcc);
    rms.input('compute').srcConnection! <= st(_mnComp);
    rms.input('norm_en').srcConnection! <= st(_mnNorm);
    rms.input('x_in').srcConnection! <= _sel(k, h1buf);
    rms.input('gamma_in').srcConnection! <= _sel(k, pgabuf);
    rms.input('eps').srcConnection! <= input('eps');
    rms.input('inv_n').srcConnection! <= input('inv_n');

    final mlp = LoomMlpSeq(hidden: H, iSize: iSize);
    mlp.input('clk').srcConnection! <= clk;
    mlp.input('reset').srcConnection! <= reset;
    mlp.input('start').srcConnection! <= st(_mStart);
    mlp.input('wb_gate').srcConnection! <= input('wb_gate');
    mlp.input('wb_up').srcConnection! <= input('wb_up');
    mlp.input('wb_down').srcConnection! <= input('wb_down');
    mlp.input('x_en').srcConnection! <= st(_mFeed) & j.lt(Const(H, width: 16));
    mlp.input('x_in').srcConnection! <= _sel(j, n2buf);
    mlp.input('sg_en').srcConnection! <=
        st(_mFeed) & j.lt(Const(iSize, width: 16));
    mlp.input('sg_in').srcConnection! <= _sel(j, sgbuf);
    mlp.input('su_en').srcConnection! <=
        st(_mFeed) & j.lt(Const(iSize, width: 16));
    mlp.input('su_in').srcConnection! <= _sel(j, subuf);
    mlp.input('sd_en').srcConnection! <= st(_mFeed) & j.lt(Const(H, width: 16));
    mlp.input('sd_in').srcConnection! <= _sel(j, sdbuf);

    final res = LoomFpResidual();
    res.input('a').srcConnection! <= _sel(ec, h1buf);
    res.input('b').srcConnection! <= _sel(ec, dbuf);

    final mlpActive = st(_mStart) | st(_mRun);
    mem.cyc <= mux(mlpActive, mlp.output('mem_CYC'), attn.output('mem_CYC'));
    mem.stb <= mux(mlpActive, mlp.output('mem_STB'), attn.output('mem_STB'));
    mem.we <= mux(mlpActive, mlp.output('mem_WE'), attn.output('mem_WE'));
    mem.adr <= mux(mlpActive, mlp.output('mem_ADR'), attn.output('mem_ADR'));
    mem.datMosi <=
        mux(mlpActive, mlp.output('mem_DAT_MOSI'), attn.output('mem_DAT_MOSI'));
    mem.sel <= mux(mlpActive, mlp.output('mem_SEL'), attn.output('mem_SEL'));
    attn.input('mem_ACK').srcConnection! <= mem.ack & ~mlpActive;
    attn.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;
    mlp.input('mem_ACK').srcConnection! <= mem.ack & mlpActive;
    mlp.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    hOutP <= res.output('sum');
    hValidP <= st(_emit);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('x_en') & xc.eq(Const(i, width: 16)),
          then: [
            hbuf[i] < input('x_in'),
            igabuf[i] < input('iga_in'),
            pgabuf[i] < input('pga_in'),
          ],
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

    // Collect attn output (hidden') into h1buf, mlp output (d) into dbuf, and
    // the post-attn normalized vector into n2buf.
    final collectWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_aRun) & attn.output('h_valid') & ac.eq(Const(i, width: 16)),
          then: [h1buf[i] < attn.output('h_out')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_mnNorm) & rms.output('y_valid') & k.eq(Const(i, width: 16)),
          then: [n2buf[i] < rms.output('y')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_mRun) & mlp.output('o_valid') & mc.eq(Const(i, width: 16)),
          then: [dbuf[i] < mlp.output('o')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 4),
          xc < Const(0, width: 16),
          sqc < Const(0, width: 16),
          skc < Const(0, width: 16),
          svc < Const(0, width: 16),
          soc < Const(0, width: 16),
          sgc < Const(0, width: 16),
          suc < Const(0, width: 16),
          sdc < Const(0, width: 16),
          j < Const(0, width: 16),
          k < Const(0, width: 16),
          ac < Const(0, width: 16),
          mc < Const(0, width: 16),
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
          for (final b in h1buf) b < Const(0, width: 16),
          for (final b in n2buf) b < Const(0, width: 16),
          for (final b in dbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...collectWrites,
          Case(state, [
            CaseItem(Const(_load, width: 4), [
              If(input('x_en'), then: [xc < xc + one]),
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
                  ac < Const(0, width: 16),
                  state < Const(_aFeed, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_aFeed, width: 4), [
              If(
                j.eq(Const(aFeedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_aStart, width: 4),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_aStart, width: 4), [
              state < Const(_aRun, width: 4),
            ]),
            CaseItem(Const(_aRun, width: 4), [
              If(attn.output('h_valid'), then: [ac < ac + one]),
              If(
                attn.output('done'),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_mnAcc, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_mnAcc, width: 4), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_mnComp, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_mnComp, width: 4), [
              state < Const(_mnWait, width: 4),
            ]),
            CaseItem(Const(_mnWait, width: 4), [
              If(
                rms.output('ready'),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_mnNorm, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_mnNorm, width: 4), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_mFeed, width: 4),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_mFeed, width: 4), [
              If(
                j.eq(Const(mFeedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  mc < Const(0, width: 16),
                  state < Const(_mStart, width: 4),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_mStart, width: 4), [
              state < Const(_mRun, width: 4),
            ]),
            CaseItem(Const(_mRun, width: 4), [
              If(mlp.output('o_valid'), then: [mc < mc + one]),
              If(
                mlp.output('done'),
                then: [
                  ec < Const(0, width: 16),
                  state < Const(_emit, width: 4),
                ],
              ),
            ]),
            CaseItem(Const(_emit, width: 4), [
              If(
                ec.eq(Const(H - 1, width: 16)),
                then: [state < Const(_fin, width: 4)],
                orElse: [ec < ec + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 4), [
              state < Const(_load, width: 4),
              xc < Const(0, width: 16),
              sqc < Const(0, width: 16),
              skc < Const(0, width: 16),
              svc < Const(0, width: 16),
              soc < Const(0, width: 16),
              sgc < Const(0, width: 16),
              suc < Const(0, width: 16),
              sdc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
