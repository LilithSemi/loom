// LoomUsbBridge: a USB-device-to-Wishbone-master bridge.
//
// Composes harbor's PHY-less full-speed USB device + DFU DNLOAD engine
// (UsbEp0Engine) with a UsbDfuRamSink (a Wishbone MASTER that drains the
// engine's firmware-byte SINK stream through a dual-clock CDC FIFO and issues
// Wishbone single-byte writes at loadBase + offset).
//
// Net effect: bytes a host delivers over USB via a DFU DNLOAD become Wishbone
// writes at loadBase + offset on the master 'bus' interface. With loadBase set
// to the Loom accelerator's base, a crafted DFU image writes the accelerator's
// buffers / CSRs over USB.
//
// Spec grounding (read first, per reference-driven-rohd):
//   - USB 2.0 ch9 control transfers + DFU 1.1 DNLOAD are implemented inside
//     harbor's UsbEp0Engine (usb_dfu.dart, ~line 638). This module does NOT
//     reinvent the PHY/engine; it composes the existing, tested blocks.
//   - The Wishbone B4 master FSM + CDC live in harbor's UsbDfuRamSink
//     (usb_dfu.dart, ~line 1696). We instantiate it as-is with loadBase set.
//
// Clock domains
// -------------
//   usb_clk  : the 48 MHz USB domain. The engine runs here; the RamSink's USB
//              side (sink_* consumption + sink_ready back-pressure) runs here
//              too. The RamSink REQUIRES usb_clk be the same clock the engine
//              runs on (the handshake signals cross domains only inside its CDC
//              FIFO).
//   bus_clk  : the slower Wishbone-master domain. The RamSink's bus FSM +
//              Wishbone master provider live here.
// Resets are split (usb_reset / bus_reset) to match the RamSink's two-domain
// reset structure; a SoC may tie them together.
//
// Sink wiring
// -----------
// The engine's sink_data/sink_valid/dnload_done outputs feed the RamSink's
// matching inputs; alt_setting (the STABLE per-transfer target select) gates
// the RAM push inside the RamSink. The RamSink's sink_ready flows back to the
// engine's sink_ready input as the back-pressure handshake.
//
// Testbench sink injection
// ------------------------
// Driving raw dp/dm enumeration through the PHY is exercised by harbor's own
// engine tests and is DEFERRED here. To test the bridge's transport + master
// path, the engine's sink stream can be injected directly via the tb_sink_*
// top ports when [allowTestbenchSinkInjection] is true (the default): a mux
// selects the tb_* stream when tb_sink_inject_en is high, otherwise the live
// engine stream. The engine is still instantiated and wired (so the real USB
// path and the SV submodule are present); injection only overrides the bytes
// the RamSink consumes, exactly like harbor's usb_dfu_test injects the sink.

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

/// A USB-device-to-Wishbone-master bridge: UsbEp0Engine + UsbDfuRamSink.
///
/// Exposes:
///   - usb_clk / usb_reset (48 MHz USB domain) and bus_clk / bus_reset (the
///     Wishbone-master domain).
///   - the raw USB pins: dp / dm (in), dp_out / dm_out / oe / usb_pullup (out).
///   - the Wishbone MASTER interface 'bus' (PairRole.provider), to be connected
///     downstream to the accelerator's Wishbone slave.
///   - observability: image_ready / bytes_written / sink_ready / overflow, plus
///     the engine status (dev_addr / configured / alt_setting / dfu_state).
///   - testbench sink-injection ports tb_sink_* (see class docs).
class LoomUsbBridge extends BridgeModule {
  /// RAM load address that image byte 0 is written to (the accelerator base).
  final int loadBase;

  /// Wishbone address bus width.
  final int busAddressWidth;

  /// Wishbone data bus width (8, 16, 32 or 64).
  final int busDataWidth;

  /// CDC FIFO depth (power of two) inside the RamSink.
  final int fifoDepth;

  /// When true, expose tb_sink_* ports that can override the engine's sink
  /// stream into the RamSink (for testing the transport + master path without
  /// raw dp/dm enumeration). Defaults to true.
  final bool allowTestbenchSinkInjection;

  /// When true, the bridge exposes a single `clk` / `reset` pair instead of the
  /// split usb_clk/usb_reset/bus_clk/bus_reset pairs, and ties BOTH internal
  /// domains (the USB engine domain and the Wishbone-master domain) to it.
  ///
  /// This is the shape harbor's [HarborSoC.addMaster] expects: it wires
  /// `master.input('clk')` and `master.input('reset')` unconditionally, so a
  /// master with split clock ports cannot be added directly. In single-clock
  /// mode the whole bridge runs in one domain (the SoC's 48 MHz USB-rate
  /// clock), which is electrically fine because the RamSink's CDC FIFO simply
  /// degenerates to a same-clock FIFO when usb_clk == bus_clk. Defaults to
  /// false so the existing split-clock testbench path is unchanged.
  final bool singleClock;

  LoomUsbBridge({
    this.loadBase = 0x0,
    this.busAddressWidth = 32,
    this.busDataWidth = 32,
    this.fifoDepth = 128,
    this.allowTestbenchSinkInjection = true,
    this.singleClock = false,
    String? name,
  }) : super('LoomUsbBridge', name: name ?? 'loom_usb_bridge') {
    _validateConfig();

    // ---- Clocks / resets. ----
    final Logic usbClk;
    final Logic usbReset;
    final Logic busClk;
    final Logic busReset;
    if (singleClock) {
      // One domain for the whole bridge (harbor SoC master shape).
      createPort('clk', PortDirection.input);
      createPort('reset', PortDirection.input);
      usbClk = busClk = input('clk');
      usbReset = busReset = input('reset');
    } else {
      createPort('usb_clk', PortDirection.input);
      createPort('usb_reset', PortDirection.input);
      createPort('bus_clk', PortDirection.input);
      createPort('bus_reset', PortDirection.input);
      usbClk = input('usb_clk');
      usbReset = input('usb_reset');
      busClk = input('bus_clk');
      busReset = input('bus_reset');
    }

    // ---- Raw USB pins. ----
    createPort('dp', PortDirection.input);
    createPort('dm', PortDirection.input);
    addOutput('dp_out');
    addOutput('dm_out');
    addOutput('oe');
    addOutput('usb_pullup');

    // ------------------------------------------------------------------
    // Sub-module 1: the PHY-less full-speed USB device + DFU DNLOAD engine.
    // ------------------------------------------------------------------
    final engine = UsbEp0Engine(name: 'usb_ep0');
    addSubModule(engine);
    engine.input('clk').srcConnection! <= usbClk;
    engine.input('reset').srcConnection! <= usbReset;
    engine.input('dp').srcConnection! <= input('dp');
    engine.input('dm').srcConnection! <= input('dm');

    // Engine line drivers + pull-up out to the top.
    output('dp_out') <= engine.output('dp_out');
    output('dm_out') <= engine.output('dm_out');
    output('oe') <= engine.output('oe');
    output('usb_pullup') <= engine.output('usb_pullup');

    // Engine status passthroughs (observability).
    addOutput('dev_addr', width: 7) <= engine.output('dev_addr');
    addOutput('configured') <= engine.output('configured');
    addOutput('eng_alt_setting', width: 8) <= engine.output('alt_setting');
    addOutput('dfu_state', width: 4) <= engine.output('dfu_state');

    // ------------------------------------------------------------------
    // Sub-module 2: the Wishbone MASTER sink (drains the engine stream into
    // RAM at loadBase + offset). Instantiated as-is with loadBase set.
    // ------------------------------------------------------------------
    final sink = UsbDfuRamSink(
      name: 'usb_dfu_ram_sink',
      loadBase: loadBase,
      busAddressWidth: busAddressWidth,
      busDataWidth: busDataWidth,
      fifoDepth: fifoDepth,
    );
    addSubModule(sink);

    // ---- Sink stream source: engine outputs, optionally overridden by the
    // testbench injection ports. ----
    final engSinkData = engine.output('sink_data');
    final engSinkValid = engine.output('sink_valid');
    final engDnloadDone = engine.output('dnload_done');
    final engAltSetting = engine.output('alt_setting');

    Logic sinkDataSrc = engSinkData;
    Logic sinkValidSrc = engSinkValid;
    Logic dnloadDoneSrc = engDnloadDone;
    Logic altSettingSrc = engAltSetting;

    if (allowTestbenchSinkInjection) {
      createPort('tb_sink_inject_en', PortDirection.input);
      createPort('tb_sink_data', PortDirection.input, width: 8);
      createPort('tb_sink_valid', PortDirection.input);
      createPort('tb_dnload_done', PortDirection.input);
      createPort('tb_alt_setting', PortDirection.input, width: 8);

      final inj = input('tb_sink_inject_en');
      sinkDataSrc = mux(inj, input('tb_sink_data'), engSinkData);
      sinkValidSrc = mux(inj, input('tb_sink_valid'), engSinkValid);
      dnloadDoneSrc = mux(inj, input('tb_dnload_done'), engDnloadDone);
      altSettingSrc = mux(inj, input('tb_alt_setting'), engAltSetting);
    }

    // ---- Wire the sink stream + clocks/resets into the RamSink. ----
    sink.input('usb_clk').srcConnection! <= usbClk;
    sink.input('usb_reset').srcConnection! <= usbReset;
    sink.input('bus_clk').srcConnection! <= busClk;
    sink.input('bus_reset').srcConnection! <= busReset;

    sink.input('sink_data').srcConnection! <= sinkDataSrc;
    sink.input('sink_valid').srcConnection! <= sinkValidSrc;
    sink.input('dnload_done').srcConnection! <= dnloadDoneSrc;
    sink.input('alt_setting').srcConnection! <= altSettingSrc;
    // image_target is observability-only inside the RamSink; tie to RAM (0).
    sink.input('image_target').srcConnection! <= Const(0, width: 8);

    // ---- Back-pressure: RamSink sink_ready -> engine sink_ready. ----
    engine.input('sink_ready').srcConnection! <= sink.output('sink_ready');

    // ---- Expose the Wishbone MASTER interface 'bus' at the top, pulled up
    // from the RamSink's master interface (preserves provider/master role). ----
    pullUpInterface(sink.interface('bus'), newIntfName: 'bus');

    // ---- Observability passthroughs (bus / USB domain). ----
    addOutput('image_ready') <= sink.output('image_ready');
    addOutput('entry_addr', width: busAddressWidth) <=
        sink.output('entry_addr');
    addOutput('bytes_written', width: 32) <= sink.output('bytes_written');
    addOutput('sink_ready') <= sink.output('sink_ready');
    addOutput('overflow') <= sink.output('overflow');
  }

  void _validateConfig() {
    if (![8, 16, 32, 64].contains(busDataWidth)) {
      throw ArgumentError(
        'busDataWidth must be one of [8,16,32,64], got '
        '$busDataWidth',
      );
    }
    if (busAddressWidth < 1 || busAddressWidth > 64) {
      throw ArgumentError('busAddressWidth out of range: $busAddressWidth');
    }
    if (loadBase < 0 || loadBase >= (BigInt.one << busAddressWidth).toInt()) {
      throw ArgumentError(
        'loadBase 0x${loadBase.toRadixString(16)} does not '
        'fit in $busAddressWidth address bits',
      );
    }
    if (fifoDepth < 2 || (fifoDepth & (fifoDepth - 1)) != 0) {
      throw ArgumentError('fifoDepth must be a power of two >= 2');
    }
  }

  /// Build-time config + structural validation. Throws on a bad config.
  void validate() => _validateConfig();
}
