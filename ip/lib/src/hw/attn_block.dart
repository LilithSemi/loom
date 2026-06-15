library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_residual.dart';
import 'linear_seq.dart';
import 'qkv_norm.dart';
import 'rope_vec.dart';
import 'attn_seq.dart';

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

/// The attention half of one on-chip decoder layer. An FSM chains the existing
/// sequencer bricks to turn `hidden[H]` at position `t` into
/// `hidden' = hidden + Wo @ attention(hidden, t)`, entirely on-chip, mirroring
/// [GoldenRunner]'s per-token attention path (RMSNorm -> Q/K/V -> RoPE -> causal
/// GQA -> o_proj -> residual).
///
/// Pipeline of reused bricks:
///   * ONE [LoomQkvNorm]  - RMSNorm(hidden) then Q/K/V through a reused matmul.
///   * ONE [LoomRopeVec]  - reused combinationally, one head per cycle, to RoPE
///     the Q heads (nH) and the K heads (nKV) at position `t`.
///   * ONE [LoomAttnSeq]  - reused, iterated over the nH query heads. The
///     persistent KV cache lives in THIS module and is replayed per head.
///   * ONE [LoomLinearSeq] - o_proj (Wo @ attnOut).
///   * ONE [LoomFpResidual] - the `hidden + O` add, streamed out.
///
/// The 1/sqrt(headDim) attention scale is folded into Q by pre-scaling the Q
/// RoPE cos/sin inputs (rotation commutes with a scalar, so a scaled cos/sin
/// yields a scaled roped-Q). The caller supplies `cos_q*`/`sin_q*` already
/// multiplied by 1/sqrt(headDim) and plain `cos_k*`/`sin_k*` for K.
///
/// The two brick weight masters (qkv_norm's and linear_seq's) never read
/// concurrently - qkv_norm reads during its run phase, linear_seq during
/// o_proj - so they are time-multiplexed onto ONE top-level Wishbone `mem`
/// provider by a state-driven mux (no arbiter needed).
///
/// Load protocol (state LOAD): stream `hidden`+`gamma` on `x_en`/`x_in`/
/// `gamma_in`. The Q/K/V weight-scale tables on `sq_en`/`sk_en`/`sv_en` and the
/// o_proj row scales on `so_en`. Hold the per-pair `cos_q*`/`sin_q*` (with
/// 1/sqrt(hd) folded in) and `cos_k*`/`sin_k*`, the weight bases `wb_q/k/v/o`,
/// `inv_n` (=1/H), `eps`, and `pos` (= t); pulse `start`. The updated residual
/// streams out on `h_out`/`h_valid` (H values, index order); `done` pulses at
/// the end. The KV cache is NOT reset between calls (only on `reset`), so
/// successive positions accumulate keys/values.
class LoomAttnBlock extends BridgeModule {
  static const int _load = 0;
  static const int _qnFeed = 1;
  static const int _qnStart = 2;
  static const int _qnRun = 3;
  static const int _ropeQ = 4;
  static const int _ropeK = 5;
  static const int _atFeedQ = 6;
  static const int _atFeedK = 7;
  static const int _atFeedV = 8;
  static const int _atStart = 9;
  static const int _atRun = 10;
  static const int _opFeed = 11;
  static const int _opStart = 12;
  static const int _opRun = 13;
  static const int _emit = 14;
  static const int _fin = 15;

  late final WishboneInterface mem;

  LoomAttnBlock({
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int maxSeq,
    int numLayers = 1,
    int recipIterations = 4,
    String? name,
  }) : super('LoomAttnBlock', name: name ?? 'loom_attn_block') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    if (numHeads % numKvHeads != 0) {
      throw ArgumentError('numHeads must be divisible by numKvHeads');
    }
    const aw = 32;
    final H = hidden;
    final nH = numHeads;
    final nKV = numKvHeads;
    final hd = headDim;
    final half = hd ~/ 2;
    final group = nH ~/ nKV;
    final qDim = nH * hd;
    final kvDim = nKV * hd;
    // The KV cache is layer-indexed so ONE attn_block can be reused across all
    // decoder layers without cache collisions: layer l occupies the slab
    // [l*perLayerCache, (l+1)*perLayerCache). With numLayers==1 there is no
    // `layer` port and layerReg stays 0.
    final hasLayer = numLayers > 1;
    final layerW = numLayers.bitLength;
    final perLayerCache = nKV * maxSeq * hd;
    final cacheLen = numLayers * perLayerCache;
    final oColTiles = (qDim + 1) ~/ 2;
    final oRowBlocks = (H + 1) ~/ 2;
    final posW = maxSeq.bitLength;
    final lenW = maxSeq.bitLength;
    final qnFeedMax = [H, qDim, kvDim].reduce((a, b) => a > b ? a : b);
    final opFeedMax = H > qDim ? H : qDim;

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
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('gamma_in', PortDirection.input, width: 16);
    createPort('sq_en', PortDirection.input);
    createPort('sq_in', PortDirection.input, width: 16);
    createPort('sk_en', PortDirection.input);
    createPort('sk_in', PortDirection.input, width: 16);
    createPort('sv_en', PortDirection.input);
    createPort('sv_in', PortDirection.input, width: 16);
    createPort('so_en', PortDirection.input);
    createPort('so_in', PortDirection.input, width: 16);
    for (var j = 0; j < half; j++) {
      createPort('cos_q$j', PortDirection.input, width: 16);
      createPort('sin_q$j', PortDirection.input, width: 16);
      createPort('cos_k$j', PortDirection.input, width: 16);
      createPort('sin_k$j', PortDirection.input, width: 16);
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

    final state = Logic(name: 'state', width: 5);
    final hc = Logic(name: 'hc', width: 16); // hidden/gamma load count
    final sqc = Logic(name: 'sqc', width: 16);
    final skc = Logic(name: 'skc', width: 16);
    final svc = Logic(name: 'svc', width: 16);
    final soc = Logic(name: 'soc', width: 16);
    final posReg = Logic(name: 'pos_reg', width: 16);
    final layerReg = Logic(name: 'layer_reg', width: 16); // captured at LOAD
    final j = Logic(name: 'j', width: 16); // shared feed index
    final qcnt = Logic(name: 'qcnt', width: 16); // qkv collect counts
    final kcnt = Logic(name: 'kcnt', width: 16);
    final vcnt = Logic(name: 'vcnt', width: 16);
    final hr = Logic(name: 'hr', width: 16); // rope head index
    final dCtr = Logic(name: 'd_ctr', width: 16); // KV write lane (per head)
    final qh = Logic(name: 'qh', width: 16); // attention query head
    final aoc = Logic(name: 'aoc', width: 16); // attn out collect index
    final opc = Logic(name: 'opc', width: 16); // o_proj collect index
    final ec = Logic(name: 'ec', width: 16); // emit index

    final hbuf = [for (var i = 0; i < H; i++) Logic(name: 'hb$i', width: 16)];
    final gbuf = [for (var i = 0; i < H; i++) Logic(name: 'gb$i', width: 16)];
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
    final qbuf = [
      for (var i = 0; i < qDim; i++) Logic(name: 'qb$i', width: 16),
    ];
    final kbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'kb$i', width: 16),
    ];
    final vbuf = [
      for (var i = 0; i < kvDim; i++) Logic(name: 'vb$i', width: 16),
    ];
    final qropebuf = [
      for (var i = 0; i < qDim; i++) Logic(name: 'qrb$i', width: 16),
    ];
    // KV cache lives in block RAM (HarborBram), not flops: it dominated the FF
    // budget (~66% at maxSeq=64). Addresses are computed in 32 bits then
    // truncated to the RAM's address width.
    final awCache = (cacheLen - 1).bitLength.clamp(1, 32);
    Logic mulC(Logic a, int b) =>
        (a.zeroExtend(32) * Const(b, width: 32)).getRange(0, 32);
    final attnOut = [
      for (var i = 0; i < qDim; i++) Logic(name: 'ao$i', width: 16),
    ];
    final obuf = [for (var i = 0; i < H; i++) Logic(name: 'ob$i', width: 16)];

    Logic st(int s) => state.eq(Const(s, width: 5));
    Logic mul16(Logic a, int b) => (a * Const(b, width: 16)).getRange(0, 16);

    final phase = Logic(name: 'phase', width: 2); // reflects qkv_norm y_phase

    final qn = LoomQkvNorm(
      hidden: H,
      qDim: qDim,
      kvDim: kvDim,
      recipIterations: recipIterations,
    );
    qn.input('clk').srcConnection! <= clk;
    // LoomQkvNorm embeds a single-shot LoomRmsNorm whose sum-of-squares
    // accumulator and reciprocal only clear on reset (it parks in its READY
    // state). To reuse it every position, hold it in reset throughout our LOAD
    // state so each position starts from a cleared accumulator. We re-stream all
    // of its inputs each position, so nothing durable is lost.
    qn.input('reset').srcConnection! <= reset | st(_load);
    qn.input('start').srcConnection! <= st(_qnStart);
    qn.input('wb_q').srcConnection! <= input('wb_q');
    qn.input('wb_k').srcConnection! <= input('wb_k');
    qn.input('wb_v').srcConnection! <= input('wb_v');
    qn.input('inv_n').srcConnection! <= input('inv_n');
    qn.input('eps').srcConnection! <= input('eps');
    qn.input('x_en').srcConnection! <= st(_qnFeed) & j.lt(Const(H, width: 16));
    qn.input('x_in').srcConnection! <= _sel(j, hbuf);
    qn.input('gamma_in').srcConnection! <= _sel(j, gbuf);
    qn.input('sq_en').srcConnection! <=
        st(_qnFeed) & j.lt(Const(qDim, width: 16));
    qn.input('sq_in').srcConnection! <= _sel(j, sqbuf);
    qn.input('sk_en').srcConnection! <=
        st(_qnFeed) & j.lt(Const(kvDim, width: 16));
    qn.input('sk_in').srcConnection! <= _sel(j, skbuf);
    qn.input('sv_en').srcConnection! <=
        st(_qnFeed) & j.lt(Const(kvDim, width: 16));
    qn.input('sv_in').srcConnection! <= _sel(j, svbuf);
    phase <= qn.output('y_phase');

    final rope = LoomRopeVec(headDim: hd);
    final ropingK = st(_ropeK);
    for (var i = 0; i < hd; i++) {
      final qIdx = mul16(hr, hd) + Const(i, width: 16);
      rope.input('x$i').srcConnection! <=
          mux(ropingK, _sel(qIdx, kbuf), _sel(qIdx, qbuf));
    }
    for (var jj = 0; jj < half; jj++) {
      rope.input('cos$jj').srcConnection! <=
          mux(ropingK, input('cos_k$jj'), input('cos_q$jj'));
      rope.input('sin$jj').srcConnection! <=
          mux(ropingK, input('sin_k$jj'), input('sin_q$jj'));
    }

    final atn = LoomAttnSeq(headDim: hd, maxSeq: maxSeq);
    final lkeys = posReg + one; // L = t + 1
    final lhd = mul16(lkeys, hd);
    // KV block-RAM addresses (32-bit math, truncated to the RAM address width).
    // The layer slab offset keeps each layer's KV in its own region.
    final layerBaseC = hasLayer
        ? mulC(layerReg, perLayerCache)
        : Const(0, width: 32);
    // Read address: layer slab + kv head (qh/group) region + flat [key][dim] j.
    final kvHeadOff = _sel(qh, [
      for (var h = 0; h < nH; h++) Const((h ~/ group) * maxSeq * hd, width: 32),
    ]);
    final kvReadAddr = (layerBaseC + kvHeadOff + j.zeroExtend(32)).getRange(
      0,
      awCache,
    );
    // Write address at ROPEK: layer slab + kv head hr + position posReg + lane.
    final kvWriteAddr =
        (layerBaseC +
                mulC(hr, maxSeq * hd) +
                mulC(posReg, hd) +
                dCtr.zeroExtend(32))
            .getRange(0, awCache);
    final kvWe = st(_ropeK);
    // Roped-K lane dCtr of the current head, and this position's V lane.
    final kWrData = _sel(dCtr, [
      for (var d = 0; d < hd; d++) rope.output('y$d'),
    ]);
    final vWrData = _sel(mul16(hr, hd) + dCtr, vbuf);

    final kStore = HarborBram(
      clk,
      width: 16,
      depth: cacheLen,
      wrEn: kvWe,
      wrAddr: kvWriteAddr,
      wrData: kWrData,
      rdAddr: kvReadAddr,
      name: 'k_store',
    );
    final vStore = HarborBram(
      clk,
      width: 16,
      depth: cacheLen,
      wrEn: kvWe,
      wrAddr: kvWriteAddr,
      wrData: vWrData,
      rdAddr: kvReadAddr,
      name: 'v_store',
    );

    atn.input('clk').srcConnection! <= clk;
    atn.input('reset').srcConnection! <= reset;
    atn.input('start').srcConnection! <= st(_atStart);
    atn.input('seq_len').srcConnection! <= lkeys.getRange(0, lenW);
    atn.input('q_en').srcConnection! <= st(_atFeedQ);
    atn.input('q_in').srcConnection! <= _sel(mul16(qh, hd) + j, qropebuf);
    // Registered read: k_in/v_in land one cycle after the address, so k_en/v_en
    // skip the first (prime) feed cycle j==0. The replay counter j runs one step
    // longer (0..lhd) to compensate.
    atn.input('k_en').srcConnection! <=
        st(_atFeedK) & ~j.eq(Const(0, width: 16));
    atn.input('k_in').srcConnection! <= kStore.rdData;
    atn.input('v_en').srcConnection! <=
        st(_atFeedV) & ~j.eq(Const(0, width: 16));
    atn.input('v_in').srcConnection! <= vStore.rdData;

    final ls = LoomLinearSeq(
      maxColTiles: oColTiles,
      maxRowBlocks: oRowBlocks,
      recipIterations: recipIterations,
    );
    ls.input('clk').srcConnection! <= clk;
    ls.input('reset').srcConnection! <= reset;
    ls.input('start').srcConnection! <= st(_opStart);
    ls.input('col_tiles').srcConnection! <= Const(oColTiles, width: 16);
    ls.input('row_blocks').srcConnection! <= Const(oRowBlocks, width: 16);
    ls.input('weight_base').srcConnection! <= input('wb_o');
    ls.input('x_en').srcConnection! <=
        st(_opFeed) & j.lt(Const(qDim, width: 16));
    ls.input('x_in').srcConnection! <= _sel(j, attnOut);
    ls.input('rs_en').srcConnection! <= st(_opFeed) & j.lt(Const(H, width: 16));
    ls.input('rs_in').srcConnection! <= _sel(j, sobuf);

    final res = LoomFpResidual();
    res.input('a').srcConnection! <= _sel(ec, hbuf);
    res.input('b').srcConnection! <= _sel(ec, obuf);

    final opActive = st(_opStart) | st(_opRun);
    mem.cyc <= mux(opActive, ls.output('mem_CYC'), qn.output('mem_CYC'));
    mem.stb <= mux(opActive, ls.output('mem_STB'), qn.output('mem_STB'));
    mem.we <= mux(opActive, ls.output('mem_WE'), qn.output('mem_WE'));
    mem.adr <= mux(opActive, ls.output('mem_ADR'), qn.output('mem_ADR'));
    mem.datMosi <=
        mux(opActive, ls.output('mem_DAT_MOSI'), qn.output('mem_DAT_MOSI'));
    mem.sel <= mux(opActive, ls.output('mem_SEL'), qn.output('mem_SEL'));
    qn.input('mem_ACK').srcConnection! <= mem.ack & ~opActive;
    qn.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;
    ls.input('mem_ACK').srcConnection! <= mem.ack & opActive;
    ls.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    hOutP <= res.output('sum');
    hValidP <= st(_emit);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final loadWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_load) & input('x_en') & hc.eq(Const(i, width: 16)),
          then: [hbuf[i] < input('x_in'), gbuf[i] < input('gamma_in')],
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
    ];

    final collectWrites = <Conditional>[
      for (var i = 0; i < qDim; i++)
        If(
          st(_qnRun) &
              qn.output('y_valid') &
              phase.eq(Const(0, width: 2)) &
              qcnt.eq(Const(i, width: 16)),
          then: [qbuf[i] < qn.output('y')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_qnRun) &
              qn.output('y_valid') &
              phase.eq(Const(1, width: 2)) &
              kcnt.eq(Const(i, width: 16)),
          then: [kbuf[i] < qn.output('y')],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_qnRun) &
              qn.output('y_valid') &
              phase.eq(Const(2, width: 2)) &
              vcnt.eq(Const(i, width: 16)),
          then: [vbuf[i] < qn.output('y')],
        ),
    ];

    // RoPE Q: write roped head hr into qropebuf.
    final ropeQWrites = <Conditional>[
      for (var h = 0; h < nH; h++)
        If(
          st(_ropeQ) & hr.eq(Const(h, width: 16)),
          then: [
            for (var d = 0; d < hd; d++)
              qropebuf[h * hd + d] < rope.output('y$d'),
          ],
        ),
    ];
    // RoPE-K and V are written into the KV block RAMs one lane per cycle during
    // ROPEK (kStore/vStore write ports, driven combinationally above); no
    // separate flop writes are needed for this state.

    final attnOutWrites = <Conditional>[
      for (var h = 0; h < nH; h++)
        for (var d = 0; d < hd; d++)
          If(
            st(_atRun) &
                atn.output('o_valid') &
                qh.eq(Const(h, width: 16)) &
                aoc.eq(Const(d, width: 16)),
            then: [attnOut[h * hd + d] < atn.output('o')],
          ),
    ];

    final obufWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_opRun) & ls.output('y_valid') & opc.eq(Const(i, width: 16)),
          then: [obuf[i] < ls.output('y')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 5),
          hc < Const(0, width: 16),
          sqc < Const(0, width: 16),
          skc < Const(0, width: 16),
          svc < Const(0, width: 16),
          soc < Const(0, width: 16),
          posReg < Const(0, width: 16),
          layerReg < Const(0, width: 16),
          j < Const(0, width: 16),
          qcnt < Const(0, width: 16),
          kcnt < Const(0, width: 16),
          vcnt < Const(0, width: 16),
          hr < Const(0, width: 16),
          qh < Const(0, width: 16),
          aoc < Const(0, width: 16),
          opc < Const(0, width: 16),
          ec < Const(0, width: 16),
          for (final b in hbuf) b < Const(0, width: 16),
          for (final b in gbuf) b < Const(0, width: 16),
          for (final b in sqbuf) b < Const(0, width: 16),
          for (final b in skbuf) b < Const(0, width: 16),
          for (final b in svbuf) b < Const(0, width: 16),
          for (final b in sobuf) b < Const(0, width: 16),
          for (final b in qbuf) b < Const(0, width: 16),
          for (final b in kbuf) b < Const(0, width: 16),
          for (final b in vbuf) b < Const(0, width: 16),
          for (final b in qropebuf) b < Const(0, width: 16),
          dCtr < Const(0, width: 16),
          for (final b in attnOut) b < Const(0, width: 16),
          for (final b in obuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...loadWrites,
          ...collectWrites,
          ...ropeQWrites,
          ...attnOutWrites,
          ...obufWrites,
          Case(state, [
            CaseItem(Const(_load, width: 5), [
              If(input('x_en'), then: [hc < hc + one]),
              If(input('sq_en'), then: [sqc < sqc + one]),
              If(input('sk_en'), then: [skc < skc + one]),
              If(input('sv_en'), then: [svc < svc + one]),
              If(input('so_en'), then: [soc < soc + one]),
              If(
                start,
                then: [
                  posReg < input('pos').zeroExtend(16),
                  layerReg <
                      (hasLayer
                          ? input('layer').zeroExtend(16)
                          : Const(0, width: 16)),
                  j < Const(0, width: 16),
                  qcnt < Const(0, width: 16),
                  kcnt < Const(0, width: 16),
                  vcnt < Const(0, width: 16),
                  state < Const(_qnFeed, width: 5),
                ],
              ),
            ]),
            CaseItem(Const(_qnFeed, width: 5), [
              If(
                j.eq(Const(qnFeedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_qnStart, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_qnStart, width: 5), [
              state < Const(_qnRun, width: 5),
            ]),
            CaseItem(Const(_qnRun, width: 5), [
              If(
                qn.output('y_valid') & phase.eq(Const(0, width: 2)),
                then: [qcnt < qcnt + one],
              ),
              If(
                qn.output('y_valid') & phase.eq(Const(1, width: 2)),
                then: [kcnt < kcnt + one],
              ),
              If(
                qn.output('y_valid') & phase.eq(Const(2, width: 2)),
                then: [vcnt < vcnt + one],
              ),
              If(
                qn.output('done'),
                then: [
                  hr < Const(0, width: 16),
                  state < Const(_ropeQ, width: 5),
                ],
              ),
            ]),
            CaseItem(Const(_ropeQ, width: 5), [
              If(
                hr.eq(Const(nH - 1, width: 16)),
                then: [
                  hr < Const(0, width: 16),
                  dCtr < Const(0, width: 16),
                  state < Const(_ropeK, width: 5),
                ],
                orElse: [hr < hr + one],
              ),
            ]),
            // ROPEK writes the KV block RAM one lane per cycle: for each kv
            // head hr, stream lanes dCtr = 0..hd-1 (the write port above uses
            // hr/dCtr/posReg to form the address).
            CaseItem(Const(_ropeK, width: 5), [
              If(
                dCtr.eq(Const(hd - 1, width: 16)),
                then: [
                  dCtr < Const(0, width: 16),
                  If(
                    hr.eq(Const(nKV - 1, width: 16)),
                    then: [
                      qh < Const(0, width: 16),
                      j < Const(0, width: 16),
                      state < Const(_atFeedQ, width: 5),
                    ],
                    orElse: [hr < hr + one],
                  ),
                ],
                orElse: [dCtr < dCtr + one],
              ),
            ]),
            CaseItem(Const(_atFeedQ, width: 5), [
              If(
                j.eq(Const(hd - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_atFeedK, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            // Registered-read feed: j runs 0..lhd (one extra prime cycle at
            // j==0 while the first address propagates). K_en is gated off at
            // j==0, so exactly lhd keys land in attn_seq.
            CaseItem(Const(_atFeedK, width: 5), [
              If(
                j.eq(lhd),
                then: [
                  j < Const(0, width: 16),
                  state < Const(_atFeedV, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_atFeedV, width: 5), [
              If(
                j.eq(lhd),
                then: [
                  j < Const(0, width: 16),
                  aoc < Const(0, width: 16),
                  state < Const(_atStart, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_atStart, width: 5), [
              state < Const(_atRun, width: 5),
            ]),
            CaseItem(Const(_atRun, width: 5), [
              If(atn.output('o_valid'), then: [aoc < aoc + one]),
              If(
                atn.output('done'),
                then: [
                  If(
                    qh.eq(Const(nH - 1, width: 16)),
                    then: [
                      j < Const(0, width: 16),
                      state < Const(_opFeed, width: 5),
                    ],
                    orElse: [
                      qh < qh + one,
                      j < Const(0, width: 16),
                      state < Const(_atFeedQ, width: 5),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_opFeed, width: 5), [
              If(
                j.eq(Const(opFeedMax - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  opc < Const(0, width: 16),
                  state < Const(_opStart, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_opStart, width: 5), [
              state < Const(_opRun, width: 5),
            ]),
            CaseItem(Const(_opRun, width: 5), [
              If(ls.output('y_valid'), then: [opc < opc + one]),
              If(
                ls.output('done'),
                then: [
                  ec < Const(0, width: 16),
                  state < Const(_emit, width: 5),
                ],
              ),
            ]),
            CaseItem(Const(_emit, width: 5), [
              If(
                ec.eq(Const(H - 1, width: 16)),
                then: [state < Const(_fin, width: 5)],
                orElse: [ec < ec + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 5), [
              state < Const(_load, width: 5),
              hc < Const(0, width: 16),
              sqc < Const(0, width: 16),
              skc < Const(0, width: 16),
              svc < Const(0, width: 16),
              soc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
