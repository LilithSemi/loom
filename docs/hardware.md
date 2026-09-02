# Hardware: targets, boards and transports

Loom is not tied to one board. The SoC comes from
[Harbor](https://git.lilithsemi.com/LilithSemi/harbor), which builds for
Lattice iCE40, Lattice ECP5 and Xilinx 7-series parts. `loom-genip` selects
the part with `--board` or `--target`, and the pins with `--pin`. To move a
design to a different FPGA, change those flags. Do not change the code.

This page gives the targets, the parts that do not move with the target, and
the numbers of the reference board.

## Select a target

### A board from the catalog (`--board`)

Harbor's catalog holds the part, the package, the oscillator, the pin sites
and the program command:

| Board | FPGA | Oscillator | Flow |
| ----- | ---- | ---------- | ---- |
| `orangecrab-25f` | ECP5 LFE5U-25F, CSFBGA285 | 48 MHz | yosys, nextpnr-ecp5, ecppack |
| `ulx3s-85f` | ECP5 LFE5U-85F, CABGA381 | 25 MHz | yosys, nextpnr-ecp5, ecppack |
| `arty-s7-50` | Spartan-7 XC7S50, csga324 | 100 MHz | yosys, openXC7 |

`orangecrab-25f` is the default, and it is the board that Loom runs on today.
The catalog gives an error that lists the known names when you ask for a board
that it does not hold.

### A part that the catalog does not hold (`--target`)

```
loom-genip --soc fp --target ecp5:25f:CSFBGA285 \
  --pin clk=A9 --pin uart_tx=N17 --pin uart_rx=M18 --model model -o out
```

The value is `vendor:device:package`. `--target` takes the `ecp5` and `ice40`
vendors, and it overrides `--board`. It assumes an input clock of 48 MHz, so
use `--board` for a board with a different oscillator, or the timing
constraint will be wrong. Give each pin with `--pin`. Use a Xilinx part
through `--board`, because `--target` has no Xilinx vendor name.

### Pins (`--pin`)

`--pin name=site` adds or replaces one pin. Repeat the flag for more pins. It
wins over the site in the catalog, and Loom adds the `LVCMOS33` IO type to it.
These are the signals that the SoC drives:

| Signal | When |
| ------ | ---- |
| `clk` | Always. |
| `uart_tx`, `uart_rx` | With the UART transport, which is every `stream` and `fp` build. |
| `spi_cs_n`, `spi_io[0]` to `spi_io[3]` | With `--fp-flash`. |
| `sdram_*` | With `--ddr`. The DDR board entry gives these, so `--pin` is for a board that Loom does not hold yet. |

## What moves with the target, and what does not

| Part | Portability |
| ---- | ----------- |
| The SoC RTL: the fabric, the accelerator and the bridges | Portable. Harbor gives the vendor primitives. |
| The fabric clock (`--fp-mhz`) | Portable. Harbor builds a PLL for each vendor. |
| The build flow | Portable. The Makefile holds `nextpnr-ice40` and `icepack` for iCE40, `nextpnr-ecp5` and `ecppack` for ECP5, and the openXC7 flow for Spartan-7. A Vivado target gets the synthesis step only. |
| Weights in the config flash (`--fp-flash`) | The board must give the flash pins. Only `orangecrab-25f` catalogs them today, so another board needs `--pin`. Harbor drives the flash clock through `USRMCLK` on ECP5 and through `STARTUPE2` on Xilinx, because that ball has no user pad. |
| Weights in DDR3 (`--ddr`) | Portable to a board with a DDR entry. `loom-genip` reads the DRAM part configuration and the pad sites from the board entry, and Harbor builds the ECP5 or the Xilinx PHY for the target. `orangecrab-25f` and `arty-s7-50` have entries. A board with no entry gives an error that lists the known names, and `--ddr` needs `--board`. |
| The `overlay` board shims and the link diagnostics | Not portable. `LoomTop.sv`, `LoomUartTop.sv` and the diagnostic tops hold OrangeCrab pins for the USB pads, the LED and the reset button. |
| The flash offsets and the 14 MiB budget | The offsets are the same everywhere, but the free space follows the size of the flash on the board. |

The part must also be large enough. The `fp` SoC fills much of an ECP5 25F, so
a small iCE40 builds the flow but does not hold the accelerator of a model.

## Add the DDR3 of a board

`ip/lib/src/hw/ddr_boards.dart` holds `LoomDdrBoard.byName`, which is keyed by
the name of the board in Harbor's catalog. Each entry gives the DRAM part
configuration, and the pad sites when Harbor's catalog does not hold them. This
is the same rule as River's `DdrBoard.byName`.

To add a board:

1. Give the entry a `HarborDdrConfig` for the DRAM part on the board.
2. Leave `pins` empty when Harbor's board catalog holds the `sdram_*` sites.
   Give the sites in the entry when it does not.
3. Add the entry to `byName` under the catalog name of the board.

Then `--ddr --board <name>` builds. `loom-genip` gives an error that lists the
known names for a board with no entry, so a build cannot take the pad sites of
a different board.

## The reference board: OrangeCrab r0.2

The default target `--board orangecrab-25f` is an
[OrangeCrab](https://github.com/GregDavil/OrangeCrab) r0.2. The numbers in
these docs come from this board:

| Part | Detail |
| ---- | ------ |
| FPGA | Lattice ECP5 `LFE5U-25F-8MG285C` in a CSFBGA285 package. |
| Clock | A 48 MHz oscillator on pin A9. |
| USB | Full speed. D+ on N1, D- on M2, and a 1k5 pull-up enable on N2. |
| UART | TX on N17 and RX on M18, on the feather header. A DirtyJTAG CDC bridge gives the host `/dev/ttyACM0`. |
| Config flash | 16 MiB of SPI flash of the W25Q128 class, through the ECP5 `USRMCLK` macro, in quad IO mode. It holds the bitstream and the resident weight image. |
| DDR3 | An MT41K64M16, which gives 128 MB. It is an optional weight store on SSTL135 pads. |
| Other | An active-low reset button on V17, and the green channel of the RGB LED on M3. |

For the `overlay` builds, `loom-genip` also writes the board shims
`LoomTop.sv` and `LoomUartTop.sv`. They fold the split USB pads of the SoC
into true bidirectional pads, they invert the active-low reset and hold it
after power-on, and they blink the LED at approximately 2.9 Hz. The blink
shows that the fabric runs with no host attached. Write a shim for your own
board to use the `overlay` SoC on it.

## Memory map

The `stream` and `fp` SoCs use a 32-bit Wishbone fabric. The map does not
change with the target:

| Base | Store |
| ---- | ----- |
| `0x00000000` | On-chip SRAM scratchpad for activations and results. |
| `0x00010000` | The accelerator CSR window. |
| `0x20000000` | The weight store: resident SPI flash or DDR3. |
| `0x30000000` | The BRAM hot-weight cache, from `--fp-bram-cache-kb`. |

The `overlay` SoC uses a 12-bit fabric and puts the accelerator at `0x000` in
a 2 KiB window. That window covers all of its registers.

### Config-flash layout (`--soc fp --fp-flash`)

```
0x00000000  bitstream
0x00200000  weights.bin        (int4, tile-major)
  ....      scales_flash.bin   (fp16 scales, one in each 32-bit word,
                                on the next 64 KiB boundary above the weights)
0x01000000  the end of the 16 MiB flash on the OrangeCrab
```

The weights start at `0x200000` on every board. The space above them follows
the size of the flash. The OrangeCrab gives approximately 14 MiB. See
[flashing.md](flashing.md#write-the-weight-image) for the commands and for the
way to find the scale offset.

## Clocking and baud

| SoC | Clock |
| --- | ----- |
| `overlay` | Directly from the board oscillator. The USB transport needs the whole SoC in that one domain. |
| `stream` | From the oscillator. With `--ddr` it runs from the PLL at `--fp-mhz`. |
| `fp` | From the PLL. The default is 30 MHz (`--fp-mhz`). |

The oscillator of the OrangeCrab runs at 48 MHz. The placed `fp` design closes
at approximately 37 MHz on its ECP5 25F. The default of 30 MHz keeps
approximately 19% of margin, which is correct on silicon and includes the
flash reads. It also makes the default baud rate of 1,500,000 an exact divisor
of 20. Drop to 24 MHz when a board is not stable. Read the report of the
place-and-route tool to find the limit on a different part.

The baud rate must be the same in the bitstream and in the runtime. Set it
with `--fp-baud` when you generate the IP, and with `--baud` when you run the
CLI. Keep the baud an exact divisor of the fabric clock. 1.5 Mbaud is the
reliable ceiling of the DirtyJTAG CDC bridge on the OrangeCrab, because 2
Mbaud corrupts the data. A different bridge has a different ceiling. The
transport is the bottleneck of the system, so the baud rate is the main speed
control.

`--ddr-read-tap` sets the static DDR3 read tap. Sweep it on the board to
centre the read eye. The default is 40.

## Transports

| Transport | What it is |
| --------- | ---------- |
| `usb` | A bit-banged full-speed USB device. It uses vendor class `0xFF`, the ids `0x1209:0x10C0` and two bulk endpoints. Its EP1 command engine is a Wishbone master, so the bytes that the host streams become bus writes. |
| `uart` | An 8N1 UART command bridge, and the bus master. It runs at 115200 for the `overlay` and `stream` SoCs, and at 1.5 Mbaud for `fp`. |
| `uartprobe` | A link diagnostic. It sends `0x55` forever at 115200. |
| `uartecho` | A link diagnostic. It sends back each byte that it receives. |
| `uartechoswap` | The same loopback with TX and RX crossed, at tx=N17 and rx=M18. |

The `overlay` SoC takes the USB transport or the UART transport. The `stream`
and `fp` SoCs take the UART only.

Write a diagnostic bitstream when a board misbehaves. If the echo comes back
byte for byte, the wires and the bridge are good, and the fault is in the
command engine. The three diagnostic tops hold OrangeCrab pins. See
[debugging.md](debugging.md).

## Bring-up sequence

1. Write the bitstream, and the weight image at `0x200000` for an `fp` build.
   See [flashing.md](flashing.md).
2. Run `loom-cli probe`. All of its counters must read zero. For an `overlay`
   build, run `loom-cli version` in place of this step.
3. Run `loom-cli generate -m <modeldir> -P "Once upon a time"`. Add `--stats`
   for the round trips and the times.

If `probe` is clean but generation is not, look at the compute path or at the
weight image. Do not look at the transport. [debugging.md](debugging.md) holds
the symptom list.
