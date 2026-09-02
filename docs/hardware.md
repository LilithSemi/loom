# Hardware: boards, memory map and transports

## The reference board: OrangeCrab r0.2

The default target is `--board orangecrab-25f`, an
[OrangeCrab](https://github.com/GregDavil/OrangeCrab) r0.2:

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
shows that the fabric runs with no host attached.

### Other targets

To retarget, change a flag. Do not change the code:

```
loom-genip --soc fp --board arty-s7-50 ...        # another catalog board
loom-genip --soc fp --target ecp5:25f:CSFBGA285 \ # or any part
  --pin clk=A9 --pin uart_tx=N17 --pin uart_rx=M18 ...
```

`--board` reads the vendor, the device, the package, the oscillator frequency,
the pin sites and the program command out of Harbor's board catalog.
`--target` with repeated `--pin name=site` is the way in for a part that the
catalog does not hold. Loom knows the `ecp5` and `ice40` vendors. Give a
signal that the catalog does not hold, such as a DDR3 site, with `--pin`.

## Memory map

The `stream` and `fp` SoCs use a 32-bit Wishbone fabric:

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
0x01000000  the end of the 16 MiB flash
```

Approximately 14 MiB is free for the weights. See
[flashing.md](flashing.md#write-the-weight-image) for the commands and for the
way to find the scale offset.

## Clocking and baud

| SoC | Clock |
| --- | ----- |
| `overlay` | Directly from the 48 MHz oscillator. The USB transport needs the whole SoC in that one domain. |
| `stream` | From the oscillator. With `--ddr` it runs from the PLL at `--fp-mhz`. |
| `fp` | From the PLL. The default is 30 MHz (`--fp-mhz`). |

The placed `fp` design closes at approximately 37 MHz. The default 30 MHz
keeps approximately 19% of margin, which is correct on silicon and includes
the flash reads. It also makes the default baud rate of 1,500,000 an exact
divisor of 20. Drop to 24 MHz when a board is not stable.

The baud rate must be the same in the bitstream and in the runtime. Set it
with `--fp-baud` when you generate the IP, and with `--baud` when you run the
CLI. 1.5 Mbaud is the reliable ceiling of the DirtyJTAG CDC bridge on this
board, because 2 Mbaud corrupts the data. The transport is the bottleneck of
the system, so the baud rate is the main speed control.

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
command engine. See [debugging.md](debugging.md).

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
