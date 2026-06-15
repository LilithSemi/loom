library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_linear.dart';
import 'fp_residual.dart';
import 'fp_rmsnorm.dart';
import 'fp_rope.dart';
import 'fp_silu.dart';
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

/// LoomSeq: the whole-model forward pass through ONE shared compute engine, with
/// the model parameters BAKED IN (generated-constant / flash-resident), not
/// host-streamed.
///
/// Same behavior and numerics as [LoomForward] (embed -> N decoder layers ->
/// final norm -> lm_head -> argmax -> token), but instead of the 3+ engines the
/// brick stack instantiates (which blow the ECP5-25F DSP/LUT budget) it uses a
/// SINGLE [LoomFpLinear] reprogrammed per matmul plus one of each resident
/// nonlinear leaf, all sequenced by a master FSM. The eight matmul kinds
/// (q,k,v,o,gate,up,down,lm_head) already run sequentially in the schedule, so
/// time-sharing the one engine costs no throughput.
///
/// This is the model-SPECIFIC sequencer genip emits: the manifest's per-layer
/// gammas, weight-base offsets, and scale-table offsets become CONSTRUCTOR
/// CONSTANTS, so the device holds no host-streamed param store (a streamed
/// store would run ~70k FF, dominated by the scale tables, well over the 25F
/// FF budget):
///   * RMSNorm gammas are a baked constant ROM (Const-populated mux tree, no
///     flops), indexed by (normKind, layer, element).
///   * Weight bases and scale-table offsets are baked `Const`s muxed by
///     (matmul kind, layer).
///   * Per-row fp16 scales are FLASH-RESIDENT: per matmul the master FSM fetches
///     the `outDim` scales from `mem` at the baked `scale_base` (one fp16 per
///     32-bit word, low16) and streams them into the engine's `rs_en`/`rs_in`
///     before firing, mirroring [LoomFpLinearAccelerator]'s resident-scale fetch
///     FSM. Scales live in flash next to the weights (swappable, matched pair),
///     not in fabric flops.
///
/// The one engine's single Wishbone `mem` master is forwarded up, time-shared
/// with the embed row reader and the scale-fetch (all three phases are disjoint).
/// The KV cache is internal [HarborBram] block RAM, off the weight bus. The only
/// remaining runtime interface is token-in (`in_token`/`pos`, plus the
/// per-position RoPE cos/sin and `inv_n`/`eps`) and token-out (`token`).
class LoomSeq extends BridgeModule {
  // Master FSM states.
  static const int _load = 0;
  static const int _embed = 1;
  static const int _normAcc = 2;
  static const int _normComp = 3;
  static const int _normWait = 4;
  static const int _normNorm = 5;
  static const int _mmFeed = 6;
  static const int _mmSfetch = 7;
  static const int _mmFire = 8;
  static const int _mmRun = 9;
  static const int _ropeQ = 10;
  static const int _ropeK = 11;
  static const int _atFeedQ = 12;
  static const int _atFeedK = 13;
  static const int _atFeedV = 14;
  static const int _atStart = 15;
  static const int _atRun = 16;
  static const int _resid = 17;
  static const int _act = 18;
  static const int _fin = 19;

  // Matmul kinds.
  static const int _mmQ = 0;
  static const int _mmK = 1;
  static const int _mmV = 2;
  static const int _mmO = 3;
  static const int _mmG = 4;
  static const int _mmU = 5;
  static const int _mmD = 6;
  static const int _mmCls = 7;

  // Norm kinds.
  static const int _nkInput = 0;
  static const int _nkPost = 1;
  static const int _nkFinal = 2;

  late final WishboneInterface mem;

  LoomSeq({
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int intermediateSize,
    required int maxSeq,
    required int numLayers,
    required int vocab,
    // RMSNorm gammas: input-norm and post-attn-norm per layer (numLayers*H each),
    // plus the final norm gamma (H). fp16-encoded.
    required List<int> inputGamma,
    required List<int> postGamma,
    required List<int> finalGamma,
    // Flash byte offset of each matrix's tile-major int4 weights. Per layer, in
    // order [q,k,v,o,gate,up,down] (numLayers*7), plus lm_head + embed table.
    required List<int> weightBase,
    required int clsWeightBase,
    required int embedBase,
    // Flash byte offset of each matrix's per-row fp16 scale table (one fp16 per
    // 32-bit word). Same layout as [weightBase] (numLayers*7), plus lm_head.
    required List<int> scaleBase,
    required int clsScaleBase,
    int recipIterations = 4,
    String? name,
  }) : super('LoomSeq', name: name ?? 'loom_seq') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    if (inputGamma.length != numLayers * hidden ||
        postGamma.length != numLayers * hidden) {
      throw ArgumentError('input/post gamma length must be numLayers*hidden');
    }
    if (finalGamma.length != hidden) {
      throw ArgumentError('finalGamma length must be hidden');
    }
    if (weightBase.length != numLayers * 7 ||
        scaleBase.length != numLayers * 7) {
      throw ArgumentError('weightBase/scaleBase length must be numLayers*7');
    }
    const aw = 32;
    final H = hidden;
    final iSize = intermediateSize;
    final hd = headDim;
    final half = hd ~/ 2;
    final group = numHeads ~/ numKvHeads;
    final nH = numHeads;
    final nKV = numKvHeads;
    final qDim = nH * hd;
    final kvDim = nKV * hd;
    final posW = maxSeq.bitLength;
    final lenW = maxSeq.bitLength;
    final tokenW = vocab.bitLength;

    // Per-matmul-kind dims (index by mmKind 0..7).
    final inDimOf = [H, H, H, qDim, H, H, iSize, H];
    final outDimOf = [qDim, kvDim, kvDim, H, iSize, iSize, H, vocab];
    int ceil2(int n) => (n + 1) ~/ 2;
    final colTilesOf = [for (final d in inDimOf) ceil2(d)];
    final maxInDim = [H, qDim, iSize].reduce((a, b) => a > b ? a : b);

    // KV cache (layer-indexed block RAM).
    final perLayerCache = nKV * maxSeq * hd;
    final cacheLen = numLayers * perLayerCache;
    final awCache = (cacheLen - 1).bitLength.clamp(1, 32);

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('pos', PortDirection.input, width: posW);
    createPort('inv_n', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('in_token', PortDirection.input, width: tokenW);
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

    final state = Logic(name: 'state', width: 5);
    final mmKind = Logic(name: 'mm_kind', width: 3);
    final normKind = Logic(name: 'norm_kind', width: 2);
    final residKind = Logic(name: 'resid_kind');
    final lc = Logic(name: 'lc', width: 16); // layer index
    final tokReg = Logic(name: 'tok_reg', width: 16);
    final posReg = Logic(name: 'pos_reg', width: 16);
    final ei = Logic(name: 'ei', width: 16); // embed read index
    final k = Logic(name: 'k', width: 16); // norm element index
    final j = Logic(name: 'j', width: 16); // feed / replay index
    final rc = Logic(name: 'rc', width: 16); // matmul result collect index
    final hr = Logic(name: 'hr', width: 16); // rope head
    final dCtr = Logic(name: 'd_ctr', width: 16); // KV write lane
    final qh = Logic(name: 'qh', width: 16); // attn query head
    final aoc = Logic(name: 'aoc', width: 16); // attn out collect
    final ec = Logic(name: 'ec', width: 16); // residual / general index
    final ai = Logic(name: 'ai', width: 16); // act index
    final bestKey = Logic(name: 'best_key', width: 16);
    final bestIdx = Logic(name: 'best_idx', width: tokenW);
    final scaleAddr = Logic(
      name: 'scale_addr',
      width: aw,
    ); // flash scale word addr
    final sidx = Logic(name: 'sidx', width: 16); // scales fetched so far
    final tileIdx = Logic(
      name: 'tile_idx',
      width: 16,
    ); // lm_head row-tile index

    List<Logic> buf(String p, int n) => [
      for (var i = 0; i < n; i++) Logic(name: '$p$i', width: 16),
    ];
    final hbuf = buf('hb', H);
    final nbuf = buf('nb', H);
    final n2buf = buf('n2', H);
    final fbuf = buf('fb', H);
    final qbuf = buf('qb', qDim);
    final kbuf = buf('kb', kvDim);
    final vbuf = buf('vb', kvDim);
    final qropebuf = buf('qr', qDim);
    final attnOutBuf = buf('ao', qDim);
    final obuf = buf('ob', H);
    final dbuf = buf('db', H);
    // gate / up / silu*up (iSize-deep) live in block RAM (HarborBram), not flops:
    // at iSize=172 the three of them would cost ~8k FF as flip-flops.
    final awInter = (iSize - 1).bitLength.clamp(1, 32);

    Logic st(int s) => state.eq(Const(s, width: 5));
    Logic mul16(Logic a, int b) => (a * Const(b, width: 16)).getRange(0, 16);
    Logic mulC(Logic a, int b) =>
        (a.zeroExtend(32) * Const(b, width: 32)).getRange(0, 32);
    Logic mmC(List<int> vals) =>
        _sel(mmKind, [for (final v in vals) Const(v, width: 16)]);

    // RMSNorm gamma constant ROM: [input(numLayers*H)][post(numLayers*H)][final(H)].
    final gammaConsts = [
      for (final g in inputGamma) Const(g, width: 16),
      for (final g in postGamma) Const(g, width: 16),
      for (final g in finalGamma) Const(g, width: 16),
    ];
    final gammaIndex = _sel(normKind, [
      mul16(lc, H) + k,
      (Const(numLayers * H, width: 16) + mul16(lc, H) + k).getRange(0, 16),
      (Const(2 * numLayers * H, width: 16) + k).getRange(0, 16),
    ]);
    final gammaConst = _sel(gammaIndex, gammaConsts);

    // Baked weight-base and scale-base selects (per (matmul kind, layer)).
    final wbConsts = [for (final b in weightBase) Const(b, width: aw)];
    final sbConsts = [for (final b in scaleBase) Const(b, width: aw)];
    final kindLayerIdx = mul16(lc, 7) + mmKind.zeroExtend(16);
    final isCls = mmKind.eq(Const(_mmCls, width: 3));
    final weightBaseSel = mux(
      isCls,
      Const(clsWeightBase, width: aw),
      _sel(kindLayerIdx, wbConsts),
    );
    final scaleBaseSel = mux(
      isCls,
      Const(clsScaleBase, width: aw),
      _sel(kindLayerIdx, sbConsts),
    );

    final normActive =
        st(_normAcc) | st(_normComp) | st(_normWait) | st(_normNorm);

    final rms = LoomRmsNorm(recipIterations: recipIterations);
    rms.input('clk').srcConnection! <= clk;
    rms.input('reset').srcConnection! <= reset | ~normActive;
    rms.input('acc_en').srcConnection! <= st(_normAcc);
    rms.input('compute').srcConnection! <= st(_normComp);
    rms.input('norm_en').srcConnection! <= st(_normNorm);
    rms.input('x_in').srcConnection! <= _sel(k, hbuf);
    rms.input('gamma_in').srcConnection! <= gammaConst;
    rms.input('eps').srcConnection! <= input('eps');
    rms.input('inv_n').srcConnection! <= input('inv_n');

    // Sized to the small (attention/o/down) matmuls. The tall ones (gate, up,
    // lm_head) are ROW-TILED in the MM sub-FSM (tiles of engMaxRows). This keeps
    // the engine's result/scale/accumulator arrays small (sizing for lm_head's
    // 256 row-blocks would cost ~16k FF of accumulators alone).
    final maxNonTiledOut = [qDim, kvDim, H].reduce((a, b) => a > b ? a : b);
    final engRowBlocks = ceil2(maxNonTiledOut);
    final engMaxRows = engRowBlocks * 2;
    final numTilesOf = [
      for (final d in outDimOf) (d + engMaxRows - 1) ~/ engMaxRows,
    ];
    final maxTiles = numTilesOf.reduce((a, b) => a > b ? a : b);
    // Per-matmul byte stride between successive row-tiles of tile-major weights:
    // engRowBlocks row-blocks, each `wordsPerRow` 32-bit words.
    final rowStrideOf = [
      for (final c in colTilesOf) engRowBlocks * ceil2(c) * 4,
    ];

    final eng = LoomFpLinear(
      maxColTiles: ceil2(maxInDim),
      maxRowBlocks: engRowBlocks,
      recipIterations: recipIterations,
    );
    final engY = eng.output('y');
    final engYValid = eng.output('y_valid');
    final engDone = eng.output('done');
    final inDimSel = mmC(inDimOf);
    final outDimSel = mmC(outDimOf);
    // Global output-row base of the current row-tile (mux-of-constants, no mul).
    final tileBase = _sel(tileIdx, [
      for (var t = 0; t < maxTiles; t++) Const(t * engMaxRows, width: 16),
    ]);
    // Global result-row index while collecting (gate/up tile, so span tiles).
    final gRow = (tileBase + rc).getRange(0, awInter);

    // gate and up are written straight from the engine result stream (MM_RUN).
    // Read back in ACT to form silu(gate)*up -> interbuf. Interbuf is then read
    // by the engine as the `down` matmul activations. All three are single-write,
    // single-registered-read (readLatency 1) memories that yosys maps to DP16KD.
    final gStore = HarborBram(
      clk,
      width: 16,
      depth: iSize,
      wrEn: st(_mmRun) & engYValid & mmKind.eq(Const(_mmG, width: 3)),
      wrAddr: gRow,
      wrData: engY,
      rdAddr: ai.getRange(0, awInter),
      name: 'g_store',
    );
    final uStore = HarborBram(
      clk,
      width: 16,
      depth: iSize,
      wrEn: st(_mmRun) & engYValid & mmKind.eq(Const(_mmU, width: 3)),
      wrAddr: gRow,
      wrData: engY,
      rdAddr: ai.getRange(0, awInter),
      name: 'u_store',
    );
    // SwiGLU: silu(gate) * up. gate/up come from the registered BRAM read (so ACT
    // is primed: at ai>=1 the read holds index ai-1, whose product is written to
    // interbuf[ai-1]).
    final silu = LoomSiLU();
    silu.input('x').srcConnection! <= gStore.rdData;
    final hProd = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(silu.output('y')),
      FloatingPoint16()..gets(uStore.rdData),
    ).product.packed;
    final interStore = HarborBram(
      clk,
      width: 16,
      depth: iSize,
      wrEn: st(_act) & ~ai.eq(Const(0, width: 16)),
      wrAddr: (ai - one).getRange(0, awInter),
      wrData: hProd,
      rdAddr: j.getRange(0, awInter),
      name: 'inter_store',
    );

    // Per-matmul activation feed. Registered-read `interbuf` (down matmul) means
    // the engine feed is PRIMED: j runs 0..inDim, x_en is off at j==0, and the
    // value driven at cycle j is index j-1 (interbuf via BRAM latency 1. The flop
    // sources via a j-1 mux) so both align.
    final jm1 = (j - one).getRange(0, 16);
    final flopActSrc = _sel(mmKind, [
      _sel(jm1, nbuf), // Q
      _sel(jm1, nbuf), // K
      _sel(jm1, nbuf), // V
      _sel(jm1, attnOutBuf), // O
      _sel(jm1, n2buf), // G
      _sel(jm1, n2buf), // U
      Const(0, width: 16), // D (uses interStore.rdData)
      _sel(jm1, fbuf), // CLS
    ]);

    // lm_head (numTiles>1) tiles. Every other matmul is a single tile (tileIdx=0,
    // Tile offsets are mux-of-constant selects (NOT multiplies), so they cost a
    // few LUTs instead of a DSP each. (tileBase is defined next to the engine.)
    final tileRem = (outDimSel - tileBase).getRange(0, 16);
    final tileRows = mux(
      tileRem.gt(Const(engMaxRows, width: 16)),
      Const(engMaxRows, width: 16),
      tileRem,
    );
    // ceil(tileRows/2) = (tileRows+1) >> 1, as an unsigned bit-slice (avoid the
    // `>>` operator, which trips a yosys signedness assert on the emitted SV).
    final tileRowBlocks = (tileRows + one).getRange(1, 16).zeroExtend(16);
    final numTilesSel = mmC(numTilesOf);
    // Per-tile weight base: weightBaseSel + tileIdx * rowStride(mmKind).
    final tileWOff = _sel(mmKind, [
      for (final stride in rowStrideOf)
        _sel(tileIdx, [
          for (var t = 0; t < maxTiles; t++) Const(t * stride, width: aw),
        ]),
    ]);
    final tileSOff = _sel(tileIdx, [
      for (var t = 0; t < maxTiles; t++) Const(t * engMaxRows * 4, width: aw),
    ]);
    final weightBaseTile = (weightBaseSel + tileWOff).getRange(0, aw);
    final scaleBaseTile = (scaleBaseSel + tileSOff).getRange(0, aw);

    // Scale-fetch: read this tile's per-row fp16 scales from flash (one word
    // each) and push into the engine's rs_en/rs_in while it is still in _load.
    final scaleReading = st(_mmSfetch);
    final scaleAck = scaleReading & mem.ack;

    eng.input('clk').srcConnection! <= clk;
    eng.input('reset').srcConnection! <= reset;
    eng.input('start').srcConnection! <= st(_mmFire);
    eng.input('col_tiles').srcConnection! <= mmC(colTilesOf);
    eng.input('row_blocks').srcConnection! <= tileRowBlocks;
    eng.input('weight_base').srcConnection! <= weightBaseTile;
    eng.input('x_en').srcConnection! <=
        st(_mmFeed) & ~j.eq(Const(0, width: 16));
    eng.input('x_in').srcConnection! <=
        mux(mmKind.eq(Const(_mmD, width: 3)), interStore.rdData, flopActSrc);
    eng.input('rs_en').srcConnection! <= scaleAck;
    eng.input('rs_in').srcConnection! <= mem.datMiso.getRange(0, 16);

    // One shared rope unit is time-sliced across lanes instead of `half` parallel
    // units (which would need 4*half fp16 multipliers, 16 at hd=8, vs 4 here).
    // ROPEQ and ROPEK both iterate the lane counter dCtr 0..hd-1. For lane d the
    // paired index is pr = d mod half (d<half -> y1 is lane d. D>=half -> y2 is
    // lane d).
    final rope = LoomRope();
    final ropingK = st(_ropeK);
    final dLow = dCtr.lt(Const(half, width: 16)); // this lane is the low half
    final pr = mux(dLow, dCtr, (dCtr - Const(half, width: 16)).getRange(0, 16));
    final ropeBaseLo = mul16(hr, hd) + pr;
    final ropeBaseHi = mul16(hr, hd) + pr + Const(half, width: 16);
    rope.input('x1').srcConnection! <=
        mux(ropingK, _sel(ropeBaseLo, kbuf), _sel(ropeBaseLo, qbuf));
    rope.input('x2').srcConnection! <=
        mux(ropingK, _sel(ropeBaseHi, kbuf), _sel(ropeBaseHi, qbuf));
    rope.input('cos_in').srcConnection! <=
        mux(
          ropingK,
          _sel(pr, [for (var jj = 0; jj < half; jj++) input('cos_k$jj')]),
          _sel(pr, [for (var jj = 0; jj < half; jj++) input('cos_q$jj')]),
        );
    rope.input('sin_in').srcConnection! <=
        mux(
          ropingK,
          _sel(pr, [for (var jj = 0; jj < half; jj++) input('sin_k$jj')]),
          _sel(pr, [for (var jj = 0; jj < half; jj++) input('sin_q$jj')]),
        );
    // The roped value for the current lane dCtr.
    final ropeLane = mux(dLow, rope.output('y1'), rope.output('y2'));

    final layerBaseC = mulC(lc, perLayerCache);
    final kvReadAddr =
        (layerBaseC +
                _sel(qh, [
                  for (var h = 0; h < nH; h++)
                    Const((h ~/ group) * maxSeq * hd, width: 32),
                ]) +
                j.zeroExtend(32))
            .getRange(0, awCache);
    final kvWriteAddr =
        (layerBaseC +
                mulC(hr, maxSeq * hd) +
                mulC(posReg, hd) +
                dCtr.zeroExtend(32))
            .getRange(0, awCache);
    final kWrData = ropeLane;
    final vWrData = _sel(mul16(hr, hd) + dCtr, vbuf);
    final kStore = HarborBram(
      clk,
      width: 16,
      depth: cacheLen,
      wrEn: st(_ropeK),
      wrAddr: kvWriteAddr,
      wrData: kWrData,
      rdAddr: kvReadAddr,
      name: 'k_store',
    );
    final vStore = HarborBram(
      clk,
      width: 16,
      depth: cacheLen,
      wrEn: st(_ropeK),
      wrAddr: kvWriteAddr,
      wrData: vWrData,
      rdAddr: kvReadAddr,
      name: 'v_store',
    );

    final atn = LoomAttnSeq(headDim: hd, maxSeq: maxSeq);
    final lkeys = posReg + one;
    final lhd = mul16(lkeys, hd);
    atn.input('clk').srcConnection! <= clk;
    atn.input('reset').srcConnection! <= reset;
    atn.input('start').srcConnection! <= st(_atStart);
    atn.input('seq_len').srcConnection! <= lkeys.getRange(0, lenW);
    atn.input('q_en').srcConnection! <= st(_atFeedQ);
    atn.input('q_in').srcConnection! <= _sel(mul16(qh, hd) + j, qropebuf);
    atn.input('k_en').srcConnection! <=
        st(_atFeedK) & ~j.eq(Const(0, width: 16));
    atn.input('k_in').srcConnection! <= kStore.rdData;
    atn.input('v_en').srcConnection! <=
        st(_atFeedV) & ~j.eq(Const(0, width: 16));
    atn.input('v_in').srcConnection! <= vStore.rdData;

    // (SwiGLU silu/mult are wired above, next to the g/u/inter block RAMs.)

    final res = LoomFpResidual();
    res.input('a').srcConnection! <= _sel(ec, hbuf);
    res.input('b').srcConnection! <=
        mux(residKind, _sel(ec, dbuf), _sel(ec, obuf));

    final key = mux(
      engY[15],
      ~engY,
      engY | Const(0x8000, width: 16),
    ).named('argmax_key');

    final embActive = st(_embed);
    final embIdx = mul16(tokReg, H) + ei;
    final embAdr =
        (Const(embedBase, width: aw) +
        (embIdx.zeroExtend(32) * Const(4, width: 32)).getRange(0, 32));
    final embStb = st(_embed) & ei.lt(Const(H, width: 16));
    mem.cyc <=
        mux(
          embActive,
          embStb,
          mux(scaleReading, Const(1), eng.output('mem_CYC')),
        );
    mem.stb <=
        mux(
          embActive,
          embStb,
          mux(scaleReading, Const(1), eng.output('mem_STB')),
        );
    mem.we <=
        mux(
          embActive,
          Const(0),
          mux(scaleReading, Const(0), eng.output('mem_WE')),
        );
    mem.adr <=
        mux(
          embActive,
          embAdr,
          mux(scaleReading, scaleAddr, eng.output('mem_ADR')),
        );
    mem.datMosi <=
        mux(
          embActive,
          Const(0, width: 32),
          mux(scaleReading, Const(0, width: 32), eng.output('mem_DAT_MOSI')),
        );
    mem.sel <=
        mux(
          embActive,
          Const(0xF, width: mem.sel.width),
          mux(
            scaleReading,
            Const(0xF, width: mem.sel.width),
            eng.output('mem_SEL'),
          ),
        );
    eng.input('mem_ACK').srcConnection! <= mem.ack & ~embActive & ~scaleReading;
    eng.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;

    tokenP <= bestIdx;
    tokenValidP <= st(_fin);
    doneP <= st(_fin);
    busyP <= ~st(_load);

    final embedWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_embed) & mem.ack & ei.eq(Const(i, width: 16)),
          then: [hbuf[i] < mem.datMiso.getRange(0, 16)],
        ),
    ];

    // Norm output -> nbuf/n2buf/fbuf by normKind.
    final normWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_normNorm) &
              rms.output('y_valid') &
              normKind.eq(Const(_nkInput, width: 2)) &
              k.eq(Const(i, width: 16)),
          then: [nbuf[i] < rms.output('y')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_normNorm) &
              rms.output('y_valid') &
              normKind.eq(Const(_nkPost, width: 2)) &
              k.eq(Const(i, width: 16)),
          then: [n2buf[i] < rms.output('y')],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_normNorm) &
              rms.output('y_valid') &
              normKind.eq(Const(_nkFinal, width: 2)) &
              k.eq(Const(i, width: 16)),
          then: [fbuf[i] < rms.output('y')],
        ),
    ];

    // Engine result -> the right dest buffer by mmKind (argmax for CLS).
    final mmCollectWrites = <Conditional>[
      for (var i = 0; i < qDim; i++)
        If(
          st(_mmRun) &
              engYValid &
              mmKind.eq(Const(_mmQ, width: 3)) &
              rc.eq(Const(i, width: 16)),
          then: [qbuf[i] < engY],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_mmRun) &
              engYValid &
              mmKind.eq(Const(_mmK, width: 3)) &
              rc.eq(Const(i, width: 16)),
          then: [kbuf[i] < engY],
        ),
      for (var i = 0; i < kvDim; i++)
        If(
          st(_mmRun) &
              engYValid &
              mmKind.eq(Const(_mmV, width: 3)) &
              rc.eq(Const(i, width: 16)),
          then: [vbuf[i] < engY],
        ),
      for (var i = 0; i < H; i++)
        If(
          st(_mmRun) &
              engYValid &
              mmKind.eq(Const(_mmO, width: 3)) &
              rc.eq(Const(i, width: 16)),
          then: [obuf[i] < engY],
        ),
      // gate (G) and up (U) results go to gStore/uStore block RAM via their write
      // ports (wrEn/wrAddr=rc/wrData=engY), not flop buffers.
      for (var i = 0; i < H; i++)
        If(
          st(_mmRun) &
              engYValid &
              mmKind.eq(Const(_mmD, width: 3)) &
              rc.eq(Const(i, width: 16)),
          then: [dbuf[i] < engY],
        ),
      // lm_head streaming argmax over the GLOBAL row index (tileBase + rc), so it
      // works across row-tiles. Guard against padding rows past vocab.
      If(
        st(_mmRun) &
            engYValid &
            mmKind.eq(Const(_mmCls, width: 3)) &
            (tileBase + rc).lt(Const(vocab, width: 16)) &
            key.gt(bestKey),
        then: [bestKey < key, bestIdx < (tileBase + rc).getRange(0, tokenW)],
      ),
    ];

    // RoPE-Q -> qropebuf, one lane (dCtr) per cycle from the shared rope unit.
    final ropeQWrites = <Conditional>[
      for (var h = 0; h < nH; h++)
        for (var d = 0; d < hd; d++)
          If(
            st(_ropeQ) &
                hr.eq(Const(h, width: 16)) &
                dCtr.eq(Const(d, width: 16)),
            then: [qropebuf[h * hd + d] < ropeLane],
          ),
    ];

    // Attention head outputs -> attnOutBuf.
    final attnWrites = <Conditional>[
      for (var h = 0; h < nH; h++)
        for (var d = 0; d < hd; d++)
          If(
            st(_atRun) &
                atn.output('o_valid') &
                qh.eq(Const(h, width: 16)) &
                aoc.eq(Const(d, width: 16)),
            then: [attnOutBuf[h * hd + d] < atn.output('o')],
          ),
    ];

    // (SwiGLU product interbuf[ai-1] is written via interStore's BRAM write port.)

    // Residual -> hidden in place.
    final residWrites = <Conditional>[
      for (var i = 0; i < H; i++)
        If(
          st(_resid) & ec.eq(Const(i, width: 16)),
          then: [hbuf[i] < res.output('sum')],
        ),
    ];

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_load, width: 5),
          mmKind < Const(0, width: 3),
          normKind < Const(0, width: 2),
          residKind < Const(0),
          lc < Const(0, width: 16),
          tokReg < Const(0, width: 16),
          posReg < Const(0, width: 16),
          ei < Const(0, width: 16),
          k < Const(0, width: 16),
          j < Const(0, width: 16),
          rc < Const(0, width: 16),
          hr < Const(0, width: 16),
          dCtr < Const(0, width: 16),
          qh < Const(0, width: 16),
          aoc < Const(0, width: 16),
          ec < Const(0, width: 16),
          ai < Const(0, width: 16),
          bestKey < Const(0, width: 16),
          bestIdx < Const(0, width: tokenW),
          scaleAddr < Const(0, width: aw),
          sidx < Const(0, width: 16),
          tileIdx < Const(0, width: 16),
          for (final b in hbuf) b < Const(0, width: 16),
        ],
        orElse: [
          ...embedWrites,
          ...normWrites,
          ...mmCollectWrites,
          ...ropeQWrites,
          ...attnWrites,
          ...residWrites,
          Case(state, [
            CaseItem(Const(_load, width: 5), [
              If(
                start,
                then: [
                  tokReg < input('in_token').zeroExtend(16),
                  posReg < input('pos').zeroExtend(16),
                  ei < Const(0, width: 16),
                  state < Const(_embed, width: 5),
                ],
              ),
            ]),
            CaseItem(Const(_embed, width: 5), [
              If(
                mem.ack,
                then: [
                  If(
                    ei.eq(Const(H - 1, width: 16)),
                    then: [
                      lc < Const(0, width: 16),
                      normKind < Const(_nkInput, width: 2),
                      k < Const(0, width: 16),
                      state < Const(_normAcc, width: 5),
                    ],
                    orElse: [ei < ei + one],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_normAcc, width: 5), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_normComp, width: 5),
                ],
                orElse: [k < k + one],
              ),
            ]),
            CaseItem(Const(_normComp, width: 5), [
              state < Const(_normWait, width: 5),
            ]),
            CaseItem(Const(_normWait, width: 5), [
              If(
                rms.output('ready'),
                then: [
                  k < Const(0, width: 16),
                  state < Const(_normNorm, width: 5),
                ],
              ),
            ]),
            CaseItem(Const(_normNorm, width: 5), [
              If(
                k.eq(Const(H - 1, width: 16)),
                then: [
                  j < Const(0, width: 16),
                  tileIdx < Const(0, width: 16),
                  If(
                    normKind.eq(Const(_nkInput, width: 2)),
                    then: [
                      mmKind < Const(_mmQ, width: 3),
                      state < Const(_mmFeed, width: 5),
                    ],
                  ),
                  If(
                    normKind.eq(Const(_nkPost, width: 2)),
                    then: [
                      mmKind < Const(_mmG, width: 3),
                      state < Const(_mmFeed, width: 5),
                    ],
                  ),
                  If(
                    normKind.eq(Const(_nkFinal, width: 2)),
                    then: [
                      mmKind < Const(_mmCls, width: 3),
                      bestKey < Const(0, width: 16),
                      bestIdx < Const(0, width: tokenW),
                      state < Const(_mmFeed, width: 5),
                    ],
                  ),
                ],
                orElse: [k < k + one],
              ),
            ]),
            // Feed the activations (PRIMED: j runs 0..inDim, x_en off at j==0),
            // then fetch THIS TILE's scales from flash (at the tile's scale base).
            CaseItem(Const(_mmFeed, width: 5), [
              If(
                j.eq(inDimSel),
                then: [
                  j < Const(0, width: 16),
                  sidx < Const(0, width: 16),
                  scaleAddr < scaleBaseTile,
                  state < Const(_mmSfetch, width: 5),
                ],
                orElse: [j < j + one],
              ),
            ]),
            CaseItem(Const(_mmSfetch, width: 5), [
              If(
                scaleAck,
                then: [
                  sidx < sidx + one,
                  scaleAddr < scaleAddr + Const(4, width: aw),
                  If(
                    (sidx + one).gte(tileRows),
                    then: [
                      rc < Const(0, width: 16),
                      state < Const(_mmFire, width: 5),
                    ],
                  ),
                ],
              ),
            ]),
            CaseItem(Const(_mmFire, width: 5), [
              state < Const(_mmRun, width: 5),
            ]),
            CaseItem(Const(_mmRun, width: 5), [
              If(engYValid, then: [rc < rc + one]),
              If(
                engDone,
                // More row-tiles of THIS matmul remain -> run the next tile
                // (same mmKind, re-feed acts). Else finish and dispatch.
                then: [
                  If(
                    tileIdx.lt((numTilesSel - one).getRange(0, 16)),
                    then: [
                      tileIdx < tileIdx + one,
                      j < Const(0, width: 16),
                      state < Const(_mmFeed, width: 5),
                    ],
                    orElse: [
                      tileIdx < Const(0, width: 16),
                      If(
                        mmKind.eq(Const(_mmQ, width: 3)),
                        then: [
                          mmKind < Const(_mmK, width: 3),
                          j < Const(0, width: 16),
                          state < Const(_mmFeed, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmK, width: 3)),
                        then: [
                          mmKind < Const(_mmV, width: 3),
                          j < Const(0, width: 16),
                          state < Const(_mmFeed, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmV, width: 3)),
                        then: [
                          hr < Const(0, width: 16),
                          dCtr < Const(0, width: 16),
                          state < Const(_ropeQ, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmO, width: 3)),
                        then: [
                          ec < Const(0, width: 16),
                          residKind < Const(0),
                          state < Const(_resid, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmG, width: 3)),
                        then: [
                          mmKind < Const(_mmU, width: 3),
                          j < Const(0, width: 16),
                          state < Const(_mmFeed, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmU, width: 3)),
                        then: [
                          ai < Const(0, width: 16),
                          state < Const(_act, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmD, width: 3)),
                        then: [
                          ec < Const(0, width: 16),
                          residKind < Const(1),
                          state < Const(_resid, width: 5),
                        ],
                      ),
                      If(
                        mmKind.eq(Const(_mmCls, width: 3)),
                        then: [state < Const(_fin, width: 5)],
                      ),
                    ],
                  ),
                ],
              ),
            ]),
            // ROPEQ: iterate lanes dCtr 0..hd-1 per head (shared rope unit).
            CaseItem(Const(_ropeQ, width: 5), [
              If(
                dCtr.eq(Const(hd - 1, width: 16)),
                then: [
                  dCtr < Const(0, width: 16),
                  If(
                    hr.eq(Const(nH - 1, width: 16)),
                    then: [
                      hr < Const(0, width: 16),
                      state < Const(_ropeK, width: 5),
                    ],
                    orElse: [hr < hr + one],
                  ),
                ],
                orElse: [dCtr < dCtr + one],
              ),
            ]),
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
                      mmKind < Const(_mmO, width: 3),
                      tileIdx < Const(0, width: 16),
                      j < Const(0, width: 16),
                      state < Const(_mmFeed, width: 5),
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
            CaseItem(Const(_resid, width: 5), [
              If(
                ec.eq(Const(H - 1, width: 16)),
                then: [
                  If(
                    residKind.eq(Const(0)),
                    then: [
                      normKind < Const(_nkPost, width: 2),
                      k < Const(0, width: 16),
                      state < Const(_normAcc, width: 5),
                    ],
                    orElse: [
                      If(
                        lc.eq(Const(numLayers - 1, width: 16)),
                        then: [
                          normKind < Const(_nkFinal, width: 2),
                          k < Const(0, width: 16),
                          state < Const(_normAcc, width: 5),
                        ],
                        orElse: [
                          lc < lc + one,
                          normKind < Const(_nkInput, width: 2),
                          k < Const(0, width: 16),
                          state < Const(_normAcc, width: 5),
                        ],
                      ),
                    ],
                  ),
                ],
                orElse: [ec < ec + one],
              ),
            ]),
            // SwiGLU (PRIMED: ai runs 0..iSize. At ai>=1 the gate/up BRAM read
            // holds index ai-1, whose product interStore writes at ai-1).
            CaseItem(Const(_act, width: 5), [
              If(
                ai.eq(Const(iSize, width: 16)),
                then: [
                  mmKind < Const(_mmD, width: 3),
                  tileIdx < Const(0, width: 16),
                  j < Const(0, width: 16),
                  state < Const(_mmFeed, width: 5),
                ],
                orElse: [ai < ai + one],
              ),
            ]),
            CaseItem(Const(_fin, width: 5), [
              state < Const(_load, width: 5),
              lc < Const(0, width: 16),
            ]),
          ]),
        ],
      ),
    ]);
  }
}
