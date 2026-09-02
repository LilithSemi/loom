# Building and flashing the bitstream

`loom-genip` writes a full build flow into the output directory. The flow
follows the vendor of the target, so the tool names below change with the
part. The examples use the ECP5 flow of the reference board. See
[hardware.md](hardware.md#select-a-target) for the other targets.

## What the compiler writes

| File | Purpose |
| ---- | ------- |
| `rtl/*.sv`, `filelist.f` | The synthesizable SystemVerilog of the SoC. |
| `synth.tcl` | The yosys script for the vendor: `synth_ecp5`, `synth_ice40` or `synth_xilinx`. |
| `Makefile` | The synthesis, place-and-route, pack and program steps. |
| `<Top>.lpf`, `<Top>.pcf` or `<Top>.xdc` | The pin constraints of the vendor. |
| `<Top>.dts`, `<Top>.svd` | The device tree and the register description. |
| `<Top>.dot`, `<Top>.mermaid.md` | Diagrams of the SoC hierarchy. |
| `board/LoomTop.sv`, `board/LoomUartTop.sv` | Board shims for the `overlay` builds. |

The model artifacts (`weights.bin`, `scales_flash.bin`, `glue.bin`,
`loom.json`, `tokenizer.bin`) go into the same directory. See
[genip.md](genip.md).

## Build the bitstream

The dev shells do not hold the FPGA tools. You must have yosys and the tools
of your vendor on `PATH`:

| Vendor | Place and route | Pack | Result |
| ------ | --------------- | ---- | ------ |
| ECP5 | `nextpnr-ecp5` | `ecppack` | `<Top>.bit` |
| iCE40 | `nextpnr-ice40` | `icepack` | `<Top>.bin` |
| Xilinx 7-series | `nextpnr-xilinx` | `fasm2frames` and `xc7frames2bit` | `<Top>.bit` |

```
cd out
make            # synth, place-and-route, then pack
```

The steps also run one at a time:

```
make synth      # yosys      -> <Top>.json
make pnr        # nextpnr    -> the vendor's intermediate file
make pack       # the packer -> the bitstream
```

The openXC7 flow reads `CHIPDB`, `XRAY_DB`, `PART`, `SEED`, `THREADS` and
`PLACER` as Make variables, so you can point it at your own database and tune
the place-and-route run.

Read the report at the end of `make pnr`. It gives the utilization and the
maximum frequency of the placed design. The frequency must stay above the
fabric clock that you asked for with `--fp-mhz`. See
[hardware.md](hardware.md#clocking-and-baud).

## Write the bitstream

The board catalog supplies the program command, and `loom-genip` writes it
into the Makefile:

```
make prog       # openFPGALoader -c dirtyJtag <Top>.bit  (orangecrab-25f)
```

`make prog` loads the FPGA over JTAG. The bitstream goes away at power-off. To
keep it, write it into the config flash. On an ECP5 board:

```
ecpprog out/LoomFpSoC.bit
```

A target from `--target` has no program command, because the catalog holds
that field. Use the tool of your board.

The green LED blinks at approximately 2.9 Hz on the `overlay` builds of the
OrangeCrab when the fabric runs. Use it as a liveness sign with no host
attached.

## Write the weight image

The `fp` SoC with `--fp-flash` reads the weights out of the config flash. The
flash holds the bitstream at offset 0, the weights at `0x200000`, and the
scales above the weights on a 64 KiB boundary. The offsets are the same on
every board.

```
ecpprog -o 0x200000 out/weights.bin
ecpprog -o <scale offset> out/scales_flash.bin
```

Read the scale offset out of the manifest. `loom.json` gives
`scale_flash_base` as a bus address, and the weight store starts at
`0x20000000`:

```
scale offset = scale_flash_base - 0x20000000
```

For a model with no BRAM cache the value is `0x200000` plus the size of
`weights.bin`, rounded up to the next 64 KiB.

A model built with `--fp-bram-cache-kb` also writes `weights_bram.bin` and
`scales_bram.bin`. Do not flash those two files. The host writes them into the
on-chip cache when `loom-cli generate` starts.

## Flash budget

The free space is the size of the flash of the board, less the `0x200000` that
the bitstream holds. The 16 MiB flash of the OrangeCrab leaves approximately
14 MiB for the weights and the scales.

`loom-genip` prints the size of each image. It checks the sizes against the
16 MiB flash of the reference board: it gives a warning above 14 MiB of
weights, and it stops with an error when the weights and the scales together
overrun 16 MiB. A model that does not fit needs DDR3, a board with a larger
flash, or a smaller checkpoint. See
[models.md](models.md#choosing-a-model).

## Next

- Check the board: [debugging.md](debugging.md).
- Generate text: [runtime.md](runtime.md).
