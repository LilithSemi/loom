// LoomUartBridge: a UART-driven Wishbone command bridge for the Loom
// accelerator.
//
// This is a RELIABLE alternative transport to the bit-banged USB path. A host
// drives the accelerator over a simple, robust 8N1 UART (default 115200 baud
// from a 48 MHz clock). It uses the EXACT SAME command protocol the USB path
// uses, by REUSING loom's proven command parser + Wishbone master,
// [LoomUsbCmdEngine] (in usb_device.dart): that engine already parses the
// {opcode:u8, addr:u32 LE, len:u16 LE} header, performs byte-granular Wishbone
// WRITES (opcode 0x01) and READS (opcode 0x02), and emits READ responses on a
// byte stream. The only net-new logic here is the UART itself (a TX + RX
// bit-level engine) and the glue that:
//   * feeds RX bytes into the command engine's cmd_data/cmd_valid stream, and
//   * drains the command engine's resp_data/resp_valid stream out the UART TX.
//
// UART bit-level engine
// The TX and RX shift engines model harbor's PROVEN [HarborUart] bit-level
// logic (same start/8-data-LSB-first/stop framing, same two-flop RX
// synchronizer, same mid-bit sampling), but exposed as a simple byte-stream
// peripheral (tx_data/tx_valid/tx_ready, rx_data/rx_valid) instead of a 16550
// register-mapped CPU slave. We need a UART that the command-engine glue can
// pump bytes through, not a CPU-driven CSR block, so a small clean engine is
// the right fit (and keeps the bit-level logic auditable against harbor's).
//
// Baud: divisor = round(clkFreq / baud). At 48 MHz, 115200 baud the divisor is
// 417 (48e6/115200 = 416.67). Each bit lasts `divisor` clocks.
//
// Clocking
// Single clock domain (the SoC runs one 48 MHz domain, exactly like the USB
// build): the UART, the command engine and the Wishbone master all run on
// `clk`. The asynchronous RX pad passes through a two-flop synchronizer before
// the sampler looks at it.

import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'usb_device.dart';

// Configuration

/// Immutable configuration for [LoomUart] / [LoomUartBridge].
class LoomUartConfig {
  /// Wishbone master address bus width.
  final int busAddressWidth;

  /// Wishbone master data bus width (8, 16, 32 or 64).
  final int busDataWidth;

  /// Input clock frequency in Hz (48 MHz on the OrangeCrab).
  final int clockFrequency;

  /// Serial baud rate (115200 8N1 by default).
  final int baudRate;

  const LoomUartConfig({
    this.busAddressWidth = 12,
    this.busDataWidth = 32,
    this.clockFrequency = 48000000,
    this.baudRate = 115200,
  });

  /// Clocks-per-bit divisor: round(clockFrequency / baudRate). At 48 MHz /
  /// 115200 this is 417.
  int get baudDivisor => ((clockFrequency + baudRate ~/ 2) ~/ baudRate);

  /// Bit width needed to hold the baud divisor counter.
  int get baudCounterWidth {
    final d = baudDivisor;
    return d <= 1 ? 1 : (d - 1).bitLength;
  }

  /// Validates the configuration. Throws [ArgumentError] on failure.
  void validate() {
    if (![8, 16, 32, 64].contains(busDataWidth)) {
      throw ArgumentError(
        'LoomUartConfig.busDataWidth must be one of [8,16,32,64], got '
        '$busDataWidth',
      );
    }
    if (busAddressWidth < 1 || busAddressWidth > 64) {
      throw ArgumentError(
        'LoomUartConfig.busAddressWidth out of range (1..64), got '
        '$busAddressWidth',
      );
    }
    if (clockFrequency <= 0) {
      throw ArgumentError(
        'LoomUartConfig.clockFrequency must be > 0, got $clockFrequency',
      );
    }
    if (baudRate <= 0) {
      throw ArgumentError('LoomUartConfig.baudRate must be > 0, got $baudRate');
    }
    if (baudDivisor < 2) {
      throw ArgumentError(
        'LoomUartConfig baud divisor (clockFrequency/baudRate = $baudDivisor) '
        'must be >= 2',
      );
    }
  }
}

// LoomUart: a byte-stream UART (TX + RX), 8N1.

/// A clean 8N1 UART with byte-stream handshakes.
///
/// Ports:
///   in:  clk, reset
///   in:  rx                      - async serial in pad (idle high)
///   out: tx                      - serial out pad (idle high)
///   in:  tx_data[8], tx_valid    - a byte to transmit is offered when tx_valid
///   out: tx_ready                - high when the TX holding register is free;
///                                  a byte is accepted on (tx_valid & tx_ready)
///   out: rx_data[8], rx_valid    - a received byte is offered (one-cycle pulse)
///
/// Framing: start bit (0), 8 data bits LSB-first, stop bit (1). Each bit lasts
/// [LoomUartConfig.baudDivisor] clocks. The RX pad is synchronized through two
/// flops, the start bit is validated at mid-bit, and data bits are sampled at
/// each bit center, mirroring harbor's proven [HarborUart] receiver.
class LoomUart extends BridgeModule {
  final LoomUartConfig config;

  LoomUart({required this.config, String? name})
    : super('LoomUart', name: name ?? 'loom_uart') {
    config.validate();

    final divisor = config.baudDivisor;
    final cw = config.baudCounterWidth;

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('rx', PortDirection.input);
    addOutput('tx');

    createPort('tx_data', PortDirection.input, width: 8);
    createPort('tx_valid', PortDirection.input);
    addOutput('tx_ready');
    // One-cycle pulse: high exactly on the cycle a tx byte is accepted (the
    // cycle tx_valid & tx_ready and the engine should advance). This is the
    // correct producer->UART accept event. Unlike the tx_ready LEVEL (which
    // stays high for one extra cycle before tx_busy rises), it fires once per
    // byte, so a producer pacing off it never double-advances.
    addOutput('tx_accept');

    addOutput('rx_data', width: 8);
    addOutput('rx_valid');

    final clk = input('clk');
    final reset = input('reset');
    final txData = input('tx_data');
    final txValid = input('tx_valid');
    final rxIn = input('rx');

    // txShift[0] is driven onto the line. Loaded with {stop=1, data[7:0],
    // start=0} so the start bit goes out first. txCount counts the 10 shifted
    // bits (start + 8 data + stop). A dedicated bit-period counter (txBaud)
    // times each bit.
    final txBusy = Logic(name: 'tx_busy');
    final txShift = Logic(name: 'tx_shift', width: 10);
    final txCount = Logic(name: 'tx_count', width: 4);
    final txBaud = Logic(name: 'tx_baud', width: cw);

    final rxSyncA = Logic(name: 'rx_sync_a');
    final rxSyncB = Logic(name: 'rx_sync_b');
    final rxBusy = Logic(name: 'rx_busy');
    final rxBits = Logic(name: 'rx_bits', width: 4);
    final rxBaud = Logic(name: 'rx_baud', width: cw);
    final rxShift = Logic(name: 'rx_shift', width: 8);
    final rxDataReg = Logic(name: 'rx_data_reg', width: 8);
    final rxValidReg = Logic(name: 'rx_valid_reg');

    // tx_ready: the holding register is free whenever the TX engine is idle.
    final txReadyLocal = (~txBusy).named('tx_ready');
    final txAccept = (txReadyLocal & txValid).named('tx_accept');

    Sequential(clk, [
      If(
        reset,
        then: [
          txBusy < Const(0),
          txShift < Const(0x3FF, width: 10), // idle-high
          txCount < Const(0, width: 4),
          txBaud < Const(0, width: cw),
          rxSyncA < Const(1), // idle-high line
          rxSyncB < Const(1),
          rxBusy < Const(0),
          rxBits < Const(0, width: 4),
          rxBaud < Const(0, width: cw),
          rxShift < Const(0, width: 8),
          rxDataReg < Const(0, width: 8),
          rxValidReg < Const(0),
        ],
        orElse: [
          // rx_valid is a one-cycle pulse.
          rxValidReg < Const(0),

          If(
            txBusy,
            then: [
              If(
                txBaud.eq(Const(0, width: cw)),
                then: [
                  txBaud < Const(divisor - 1, width: cw),
                  txShift < (txShift >> Const(1, width: 10)),
                  txCount < (txCount + Const(1, width: 4)),
                  If(
                    txCount.eq(Const(9, width: 4)),
                    then: [txBusy < Const(0), txCount < Const(0, width: 4)],
                  ),
                ],
                orElse: [txBaud < (txBaud - Const(1, width: cw))],
              ),
            ],
            orElse: [
              // Idle: accept a new byte to transmit. Frame = {stop, data, start}
              // shifted out LSB (start) first.
              If(
                txAccept,
                then: [
                  txShift <
                      [
                        Const(1, width: 1),
                        txData,
                        Const(0, width: 1),
                      ].swizzle(),
                  txBusy < Const(1),
                  txCount < Const(0, width: 4),
                  txBaud < Const(divisor - 1, width: cw),
                ],
              ),
            ],
          ),

          rxSyncA < rxIn,
          rxSyncB < rxSyncA,
          If(
            ~rxBusy,
            then: [
              // Hunt for the falling edge that starts a frame.
              If(
                ~rxSyncB,
                then: [
                  rxBusy < Const(1),
                  rxBits < Const(0, width: 4),
                  // Half a bit period: land at the middle of the start bit.
                  rxBaud < Const(divisor ~/ 2, width: cw),
                ],
              ),
            ],
            orElse: [
              If(
                rxBaud.eq(Const(0, width: cw)),
                then: [
                  rxBaud < Const(divisor - 1, width: cw),
                  If(
                    rxBits.eq(Const(0, width: 4)),
                    then: [
                      // Mid-start: still low => a real start bit, else a glitch.
                      If(
                        ~rxSyncB,
                        then: [rxBits < Const(1, width: 4)],
                        orElse: [rxBusy < Const(0)],
                      ),
                    ],
                    orElse: [
                      If(
                        rxBits.lte(Const(8, width: 4)),
                        then: [
                          // Data bits 1..8: insert at the top, shift down, so
                          // the first (LSB-first) bit lands at bit 0.
                          rxShift < [rxSyncB, rxShift.getRange(1, 8)].swizzle(),
                          rxBits < (rxBits + Const(1, width: 4)),
                        ],
                        orElse: [
                          // Stop position: a high line frames a valid byte.
                          If(
                            rxSyncB,
                            then: [rxDataReg < rxShift, rxValidReg < Const(1)],
                          ),
                          rxBusy < Const(0),
                        ],
                      ),
                    ],
                  ),
                ],
                orElse: [rxBaud < (rxBaud - Const(1, width: cw))],
              ),
            ],
          ),
        ],
      ),
    ]);

    // TX line: LSB of shift register when busy, else idle high.
    final txLine = Logic(name: 'tx_line');
    Combinational([
      If(txBusy, then: [txLine < txShift[0]], orElse: [txLine < Const(1)]),
    ]);

    output('tx') <= txLine;
    output('tx_ready') <= txReadyLocal;
    output('tx_accept') <= txAccept;
    output('rx_data') <= rxDataReg;
    output('rx_valid') <= rxValidReg;
  }
}

// LoomUartBridge: UART <-> command engine <-> Wishbone master.

/// A UART-driven Wishbone command bridge.
///
/// Composes [LoomUart] (the serial line engine) with [LoomUsbCmdEngine] (loom's
/// PROVEN command parser + Wishbone master, reused verbatim from the USB path).
/// UART RX bytes become the command-engine's cmd stream. The command-engine's
/// response bytes become UART TX bytes. The exposed `bus` interface is a
/// Wishbone MASTER to be wired to the accelerator slave.
///
/// Ports:
///   in:  clk, reset
///   in:  rx                  - serial in pad
///   out: tx                  - serial out pad
///   out: Wishbone MASTER 'bus'
///   out: busy                - command in flight (observability)
///
/// RX -> command engine
/// The UART produces a one-cycle rx_valid pulse per received byte. The command
/// engine's cmd_data/cmd_valid handshake is gated by cmd_ready (its OUT-NAK
/// gate). A full UART byte takes ~417 clocks at 115200, and the command engine
/// asserts cmd_ready within a couple of cycles of being in a header/WR_DATA
/// state, so the engine is almost always ready by the time the next byte
/// arrives. To be safe against the rare case where the engine is momentarily
/// busy (e.g. waiting on a bus ack) when a byte lands, the bridge latches the
/// received byte and re-offers it until the engine accepts it (a one-deep RX
/// holding register).
///
/// command engine -> TX
/// The command engine offers response bytes on resp_data/resp_valid, paced by
/// resp_ready. The bridge drives resp_ready from the UART's tx_ready and hands
/// each accepted response byte to the UART TX. Because resp_valid is held until
/// taken, a response byte waits for the UART to finish the previous frame.
class LoomUartBridge extends BridgeModule {
  final LoomUartConfig config;

  LoomUartBridge({required this.config, String? name})
    : super('LoomUartBridge', name: name ?? 'loom_uart_bridge') {
    config.validate();

    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('rx', PortDirection.input);
    addOutput('tx');
    addOutput('busy');

    final clk = input('clk');
    final reset = input('reset');

    final uart = LoomUart(config: config, name: 'bridge_uart');
    addSubModule(uart);
    uart.input('clk').srcConnection! <= clk;
    uart.input('reset').srcConnection! <= reset;
    uart.input('rx').srcConnection! <= input('rx');
    output('tx') <= uart.output('tx');

    final engine = LoomUsbCmdEngine(
      config: LoomUsbDeviceConfig(
        busAddressWidth: config.busAddressWidth,
        busDataWidth: config.busDataWidth,
      ),
      name: 'bridge_cmd_engine',
    );
    addSubModule(engine);
    engine.input('clk').srcConnection! <= clk;
    engine.input('reset').srcConnection! <= reset;
    // No new-command preempt on the UART path: the host streams a whole command
    // and reads its whole response in order (no partial-read re-issue like the
    // USB test does), so tie cmd_start low.
    engine.input('cmd_start').srcConnection! <= Const(0);

    // rx_valid is a one-cycle pulse. The command engine may not have cmd_ready
    // high that exact cycle, so latch the byte and re-offer it until accepted.
    final rxHoldData = Logic(name: 'rx_hold_data', width: 8);
    final rxHoldFull = Logic(name: 'rx_hold_full');

    final cmdReady = engine.output('cmd_ready');
    // A held byte is consumed when the engine accepts it.
    final rxConsumed = (rxHoldFull & cmdReady).named('rx_consumed');

    Sequential(clk, [
      If(
        reset,
        then: [rxHoldData < Const(0, width: 8), rxHoldFull < Const(0)],
        orElse: [
          // Capture a freshly received byte. (At 115200, bytes are >400 clocks
          // apart and the engine drains the prior byte within a few cycles, so
          // the holding register is effectively always empty when rx_valid
          // pulses. This latch just removes the timing assumption.)
          If(
            uart.output('rx_valid'),
            then: [rxHoldData < uart.output('rx_data'), rxHoldFull < Const(1)],
            orElse: [
              // Clear once the engine has taken the held byte.
              If(rxConsumed, then: [rxHoldFull < Const(0)]),
            ],
          ),
        ],
      ),
    ]);

    engine.input('cmd_data').srcConnection! <= rxHoldData;
    engine.input('cmd_valid').srcConnection! <= rxHoldFull;

    // The command engine holds resp_valid until the byte is taken, and advances
    // its read pointer on the cycle resp_ready is high. We MUST pulse resp_ready
    // exactly once per byte. The UART's tx_ready is a LEVEL that stays high for
    // one extra cycle after a byte is latched (until tx_busy rises next cycle),
    // so pacing resp_ready off tx_ready would let the engine advance TWICE and
    // skip a byte. Instead:
    //   * offer the response byte to the UART whenever resp_valid & tx_ready
    //     (tx_valid = resp_valid & tx_ready), and
    //   * advance the engine only on the UART's one-cycle tx_accept pulse
    //     (resp_ready = tx_accept).
    final respValid = engine.output('resp_valid');
    final txReady = uart.output('tx_ready');
    final txAccept = uart.output('tx_accept');
    final offerByte = (respValid & txReady).named('offer_byte');

    engine.input('resp_ready').srcConnection! <= txAccept;
    uart.input('tx_data').srcConnection! <= engine.output('resp_data');
    uart.input('tx_valid').srcConnection! <= offerByte;

    output('busy') <= engine.output('busy');

    // Expose the Wishbone MASTER interface 'bus' at the top.
    pullUpInterface(engine.interface('bus'), newIntfName: 'bus');
  }

  /// Build-time config validation.
  void validate() => config.validate();
}
