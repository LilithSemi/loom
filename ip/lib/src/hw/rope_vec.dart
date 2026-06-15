library;

import 'package:rohd_bridge/rohd_bridge.dart';

import 'fp_rope.dart';

/// Applies RoPE across a whole `headDim` vector using the HuggingFace half-split
/// pairing `(j, j+half)`, driving one combinational [LoomRope] per pair. The
/// per-pair cos/sin come precomputed (the sequencer reads them from the baked
/// freq_cis table - llama2.c ships these in the checkpoint - so no on-chip
/// angle generation is needed). Combinational: `y = rope(x, cos, sin)`.
///
/// Ports: in `x0..x{headDim-1}` [16], `cos0..cos{half-1}` [16],
/// `sin0..sin{half-1}` [16]. Out `y0..y{headDim-1}` [16]. All packed fp16.
class LoomRopeVec extends BridgeModule {
  LoomRopeVec({required int headDim, String? name})
    : super('LoomRopeVec', name: name ?? 'loom_rope_vec') {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even');
    }
    final half = headDim ~/ 2;

    for (var i = 0; i < headDim; i++) {
      createPort('x$i', PortDirection.input, width: 16);
    }
    for (var j = 0; j < half; j++) {
      createPort('cos$j', PortDirection.input, width: 16);
      createPort('sin$j', PortDirection.input, width: 16);
    }
    final y = [for (var i = 0; i < headDim; i++) addOutput('y$i', width: 16)];

    for (var j = 0; j < half; j++) {
      final rope = LoomRope(name: 'rope_$j');
      rope.input('x1').srcConnection! <= input('x$j');
      rope.input('x2').srcConnection! <= input('x${j + half}');
      rope.input('cos_in').srcConnection! <= input('cos$j');
      rope.input('sin_in').srcConnection! <= input('sin$j');
      y[j] <= rope.output('y1');
      y[j + half] <= rope.output('y2');
    }
  }
}
