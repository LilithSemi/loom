library;

/// How Loom lowers a model into hardware.
enum LoomStrategy { overlay, spatial, hybrid }

/// For spatial mode: how baked weights are realized. `auto` lets Loom pick by
/// what fits the target.
enum LoomBaking { auto, folded, romBaked }

/// Numeric formats the datapath may instantiate.
enum LoomNumeric { int4, int8, fp8, mxfp8, fp16, bf16, fp32 }

/// Build target (selects DRAM availability, PDK, constraints downstream).
enum LoomTarget {
  sim,
  fpgaArtyS7,
  fpgaOrangeCrab,
  fpgaIcesugar,
  asicSky130,
  asicGf180,
}

/// Top-level Loom build knobs. Validated at build time.
class LoomConfig {
  final LoomStrategy strategy;
  final Set<LoomNumeric> numerics;
  final LoomTarget target;
  final LoomBaking baking;

  const LoomConfig({
    required this.strategy,
    required this.numerics,
    required this.target,
    this.baking = LoomBaking.auto,
  });

  /// Throws [ArgumentError] on an inconsistent configuration.
  void validate() {
    if (numerics.isEmpty) {
      throw ArgumentError.value(
        numerics,
        'numerics',
        'at least one numeric format must be enabled',
      );
    }
    if (strategy != LoomStrategy.spatial && baking != LoomBaking.auto) {
      throw ArgumentError.value(
        baking,
        'baking',
        'baking is only configurable for the spatial strategy',
      );
    }
  }
}
