library;

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'loom_seq.dart';

/// Wishbone-slave CSR wrapper around [LoomSeq]: the thin-host token-pump front
/// end for the autonomous sequencer. Mirrors [LoomFpLinearAccelerator]'s dual-bus
/// shape:
///   - 'bus' : Wishbone SLAVE  - the host writes the per-token runtime inputs
///             (in_token, pos, inv_n, eps, and the RoPE cos/sin table), strobes
///             CONTROL.start, polls STATUS.done, and reads the emitted TOKEN.
///   - 'mem' : Wishbone MASTER - LoomSeq's weight-read master, forwarded up to the
///             weight store (SPI flash / SRAM) unchanged.
///
/// Unlike the fp accelerator, the host touches this ONCE PER TOKEN (~20 CSR
/// writes + a start + a token read), not once per matmul: LoomSeq runs the whole
/// forward pass on-chip.
///
/// CSR map (byte offsets, 32-bit words):
///   0x000 VERSION  RO 0x4C534551 ('LSEQ')
///   0x004 IN_TOKEN WO [tokenW-1:0]
///   0x008 POS      WO [posW-1:0]
///   0x00C INV_N    WO [15:0] fp16 1/hidden
///   0x010 EPS      WO [15:0] fp16
///   0x014 CONTROL  WO bit0 = start strobe (self-clearing)
///   0x018 STATUS   RO bit0 busy, bit1 done (sticky until next start)
///   0x01C TOKEN    RO [tokenW-1:0] the emitted next token (valid when done)
///   0x020.. ROPE   WO one fp16 per word, [cos_q,sin_q,cos_k,sin_k] x (headDim/2)
///                  in that interleaved order. 4*(headDim/2) words.
class LoomSeqCsr extends BridgeModule {
  static const int _version = 0x000;
  static const int _inToken = 0x004;
  static const int _pos = 0x008;
  static const int _invN = 0x00C;
  static const int _eps = 0x010;
  static const int _control = 0x014;
  static const int _status = 0x018;
  static const int _token = 0x01C;
  static const int _ropeBase = 0x020;

  static const int versionMagic = 0x4C534551; // 'LSEQ'

  late final BusSlavePort bus;
  late final WishboneInterface mem;

  LoomSeqCsr({
    required int baseAddress,
    required int hidden,
    required int numHeads,
    required int numKvHeads,
    required int headDim,
    required int intermediateSize,
    required int maxSeq,
    required int numLayers,
    required int vocab,
    required List<int> inputGamma,
    required List<int> postGamma,
    required List<int> finalGamma,
    required List<int> weightBase,
    required int clsWeightBase,
    required int embedBase,
    required List<int> scaleBase,
    required int clsScaleBase,
    int addressWidth = 32,
    int recipIterations = 4,
    String? name,
  }) : super('LoomSeqCsr', name: name ?? 'loom_seq_csr') {
    final aw = addressWidth;
    final half = headDim ~/ 2;
    final posW = maxSeq.bitLength;
    final tokenW = vocab.bitLength;
    // RoPE CSR ports in interleaved order [cos_q,sin_q,cos_k,sin_k] per frequency.
    final ropeNames = <String>[
      for (var j = 0; j < half; j++) ...[
        'cos_q$j',
        'sin_q$j',
        'cos_k$j',
        'sin_k$j',
      ],
    ];

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);

    bus = BusSlavePort.create(
      module: this,
      name: 'bus',
      protocol: BusProtocol.wishbone,
      addressWidth: aw,
      dataWidth: 32,
    );

    final memRef = addInterface(
      WishboneInterface(WishboneConfig(addressWidth: aw, dataWidth: 32)),
      name: 'mem',
      role: PairRole.provider,
    );
    mem = memRef.internalInterface!;

    final clk = input('clk');
    final reset = input('reset');

    final inTokReg = Logic(name: 'in_token_reg', width: tokenW);
    final posReg = Logic(name: 'pos_reg', width: posW);
    final invNReg = Logic(name: 'inv_n_reg', width: 16);
    final epsReg = Logic(name: 'eps_reg', width: 16);
    final ropeRegs = {
      for (final n in ropeNames) n: Logic(name: '${n}_reg', width: 16),
    };
    final startPulse = Logic(name: 'start_pulse');
    final tokenReg = Logic(name: 'token_reg', width: tokenW);
    final doneSticky = Logic(name: 'done_sticky');

    final seq = LoomSeq(
      hidden: hidden,
      numHeads: numHeads,
      numKvHeads: numKvHeads,
      headDim: headDim,
      intermediateSize: intermediateSize,
      maxSeq: maxSeq,
      numLayers: numLayers,
      vocab: vocab,
      inputGamma: inputGamma,
      postGamma: postGamma,
      finalGamma: finalGamma,
      weightBase: weightBase,
      clsWeightBase: clsWeightBase,
      embedBase: embedBase,
      scaleBase: scaleBase,
      clsScaleBase: clsScaleBase,
      recipIterations: recipIterations,
    );
    addSubModule(seq);
    seq.input('clk').srcConnection! <= clk;
    seq.input('reset').srcConnection! <= reset;
    seq.input('start').srcConnection! <= startPulse;
    seq.input('pos').srcConnection! <= posReg;
    seq.input('inv_n').srcConnection! <= invNReg;
    seq.input('eps').srcConnection! <= epsReg;
    seq.input('in_token').srcConnection! <= inTokReg;
    for (final n in ropeNames) {
      seq.input(n).srcConnection! <= ropeRegs[n]!;
    }

    // Forward LoomSeq's flattened Wishbone master to the 'mem' interface.
    seq.input('mem_ACK').srcConnection! <= mem.ack;
    seq.input('mem_DAT_MISO').srcConnection! <= mem.datMiso;
    mem.cyc <= seq.output('mem_CYC');
    mem.stb <= seq.output('mem_STB');
    mem.we <= seq.output('mem_WE');
    mem.adr <= seq.output('mem_ADR');
    mem.datMosi <= seq.output('mem_DAT_MOSI');
    mem.sel <= seq.output('mem_SEL');

    final busy = seq.output('busy');
    final done = seq.output('done');

    // Address decode helper (word offset, low 12 bits).
    Logic at(int off) => bus.addr.getRange(0, 12).eq(Const(off, width: 12));
    final writeAccept = bus.stb & ~bus.ack & bus.we;
    final newAccess = bus.stb & ~bus.ack;

    Sequential(clk, [
      If(
        reset,
        then: [
          bus.ack < Const(0),
          bus.dataOut < Const(0, width: 32),
          inTokReg < Const(0, width: tokenW),
          posReg < Const(0, width: posW),
          invNReg < Const(0, width: 16),
          epsReg < Const(0, width: 16),
          for (final n in ropeNames) ropeRegs[n]! < Const(0, width: 16),
          startPulse < Const(0),
          tokenReg < Const(0, width: tokenW),
          doneSticky < Const(0),
        ],
        orElse: [
          // start is a one-cycle strobe.
          startPulse < Const(0),
          bus.ack < Const(0),

          // done capture: latch the emitted token + set sticky done.
          If(
            done,
            then: [
              tokenReg < seq.output('token').getRange(0, tokenW),
              doneSticky < Const(1),
            ],
          ),

          // CSR writes.
          If(
            writeAccept & at(_inToken),
            then: [inTokReg < bus.dataIn.getRange(0, tokenW)],
          ),
          If(
            writeAccept & at(_pos),
            then: [posReg < bus.dataIn.getRange(0, posW)],
          ),
          If(
            writeAccept & at(_invN),
            then: [invNReg < bus.dataIn.getRange(0, 16)],
          ),
          If(
            writeAccept & at(_eps),
            then: [epsReg < bus.dataIn.getRange(0, 16)],
          ),
          If(
            writeAccept & at(_control) & bus.dataIn[0],
            then: [startPulse < Const(1), doneSticky < Const(0)],
          ),
          for (var i = 0; i < ropeNames.length; i++)
            If(
              writeAccept & at(_ropeBase + i * 4),
              then: [ropeRegs[ropeNames[i]]! < bus.dataIn.getRange(0, 16)],
            ),

          // Read response (single-cycle ack. Registered dataOut).
          If(
            newAccess,
            then: [
              bus.ack < Const(1),
              If(
                at(_version),
                then: [bus.dataOut < Const(versionMagic, width: 32)],
                orElse: [
                  If(
                    at(_status),
                    then: [
                      bus.dataOut < [doneSticky, busy].swizzle().zeroExtend(32),
                    ],
                    orElse: [
                      If(
                        at(_token),
                        then: [bus.dataOut < tokenReg.zeroExtend(32)],
                        orElse: [bus.dataOut < Const(0, width: 32)],
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    ]);
  }
}
