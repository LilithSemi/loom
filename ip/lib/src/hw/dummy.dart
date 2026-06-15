import 'package:rohd/rohd.dart';

/// A trivial ROHD module: one input bit, one output bit (output = input).
///
/// Used to prove the ROHD toolchain (build + SystemVerilog emit) works
/// inside the loom package before building the real accelerator datapath.
class LoomDummy extends Module {
  Logic get out => output('out');

  LoomDummy(Logic inp, {super.name = 'LoomDummy'}) {
    inp = addInput('inp', inp);
    addOutput('out');

    out <= inp;
  }
}
