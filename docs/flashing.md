# Building and flashing the bitstream

`loom-genip` writes a full ECP5 flow into the output directory. This page shows
how to make the bitstream and how to write it, and the weights, to the board.

## What the compiler writes

| File | Purpose |
| ---- | ------- |
| `rtl/*.sv`, `filelist.f` | The synthesizable SystemVerilog of the SoC. |
| `synth.tcl` | The yosys script. |
| `Makefile` | The synthesis, place-and-route and pack steps. |
| `<Top>.lpf` | The pin constraints. |
| `<Top>.dts`, `<Top>.svd` | The device tree and the register description. |
| `<Top>.dot`, `<Top>.mermaid.md` | Diagrams of the SoC hierarchy. |
| `board/LoomTop.sv`, `board/LoomUartTop.sv` | Board shims for the `overlay` builds. |

The model artifacts (`weights.bin`, `scales_flash.bin`, `glue.bin`,
`loom.json`, `tokenizer.bin`) go into the same directory. See
[genip.md](genip.md).

## Build the bitstream

The dev shells do not hold the FPGA tools. You must have yosys, nextpnr-ecp5
and ecppack on `PATH`, and `openFPGALoader` or `ecpprog` to write the board.

```
cd out
make            # synth, place-and-route, then pack
```

The steps also run one at a time:

```
make synth      # yosys      -> <Top>.json
make pnr        # nextpnr    -> <Top>.config
make pack       # ecppack    -> <Top>.bit
```

Read the nextpnr report at the end of `make pnr`. It gives the utilization and
the maximum frequency of the placed design. The frequency must stay above the
fabric clock that you asked for with `--fp-mhz`. See
[hardware.md](hardware.md#clocking-and-baud).

## Write the bitstream

The board catalog supplies the program command, and `loom-genip` writes it into
the Makefile:

```
make prog       # openFPGALoader -c dirtyJtag <Top>.bit
```

`make prog` loads the FPGA over JTAG. The bitstream goes away at power-off. To
keep it, write it into the config flash:

```
ecpprog out/LoomFpSoC.bit
```

The green LED blinks at approximately 2.9 Hz on the `overlay` builds when the
fabric runs. Use it as a liveness sign with no host attached.

## Write the weight image

The `fp` SoC with `--fp-flash` reads the weights out of the config flash. The
flash holds the bitstream at offset 0, the weights at `0x200000`, and the
scales above the weights on a 64 KiB boundary.

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

The config flash of the OrangeCrab holds 16 MiB. The bitstream sits below
`0x200000`, so approximately 14 MiB stays free for the weights and the scales.
`loom-genip` prints the size of each image. A model that does not fit needs
DDR3 or a smaller checkpoint. See [models.md](models.md#choosing-a-model).

## Retarget to another board

The flow follows the board catalog, so a different board is a flag change:

```
loom-genip --soc fp --board arty-s7-50 --model model -o out
```

The Makefile that `loom-genip` writes follows the vendor of the target. Use
`--target` and `--pin` for a part that the catalog does not hold. See
[hardware.md](hardware.md#other-targets).

## Next

- Check the board: [debugging.md](debugging.md).
- Generate text: [runtime.md](runtime.md).
