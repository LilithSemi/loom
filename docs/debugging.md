# Debugging

This page tells you how to find the layer that is broken. Work from the link,
to the compute, to the model. Each step below assumes that the step before it
is clean.

## Know which command speaks to which bitstream

The accelerator sits at a different bus address in each SoC variant, so the
`loom-cli` commands do not all work everywhere.

| Command | Address it reads | Works with |
| ------- | ---------------- | ---------- |
| `version`, `info`, `matmul` | `0x000` | `overlay` bitstreams. The `stream` and `fp` SoCs put their CSRs at `0x10000`. |
| `probe` | `0x10000` and `0x20200000` | `fp` bitstreams with `--fp-flash`. |
| `generate`, `serve` | `csr_base` from `loom.json` | `fp` bitstreams. |

A `loom-cli version` that reads `0x00000000` against an `fp` bitstream is not a
fault. That address holds the SRAM scratchpad in the `fp` memory map. Use
`probe` for an `fp` board.

## Prove the link first

```
loom-cli probe
```

`probe` prints five lines. Read them in order.

| Line | A bad count means |
| ---- | ----------------- |
| single-word VERSION reads | The read path corrupts, or the baud does not match. |
| burst VERSION reads | The burst framing breaks, but single reads are good. |
| single write-readback | The CSR write path corrupts. |
| burst write-readback | The multi-register write path corrupts. |
| flash-read consistency | The flash reads are not stable. Lower `--fp-mhz`. |

All five must read `0/N bad`. The flash line must read `0 words differ`.

## Diagnostic bitstreams

When `probe` cannot open the link at all, build a diagnostic bitstream. These
hold no accelerator, so they isolate the wires and the USB bridge.

| Transport | What it does | What it proves |
| --------- | ------------ | -------------- |
| `--transport uartprobe` | Sends `0x55` forever at 115200. | The device-to-host leg. |
| `--transport uartecho` | Sends back each byte it receives. | Both legs and the bridge. |
| `--transport uartechoswap` | The same loopback with TX and RX crossed. | A crossover mistake on the host side. |

If the echo comes back byte for byte, the wires are good. Then look at the
command engine, not at the cable.

## Symptoms

**No device at all.** The CLI prints `could not open uart transport`. The
board is not enumerated. Check the cable and check that `/dev/ttyACM0` exists.
Give the port with `-p` when the name is different.

**`generate` prints `FAIL: VERSION ...`.** The runtime read the fp VERSION CSR
and did not find the `'LOOM'` magic. Either the board holds a different
bitstream, or the weights are not in flash, or the baud does not match. The
baud must be the same at both ends: `--fp-baud` when you generate the IP, and
`--baud` when you run the CLI.

**The link is clean, but the tokens are wrong.** Suspect the weight image.
Check that you flashed `weights.bin` at `0x200000` and `scales_flash.bin` at
the offset that `loom.json` gives. See
[flashing.md](flashing.md#write-the-weight-image).

**The output repeats one phrase.** Small models loop. The runtime stops on a
repeat window of 8 tokens. Change it with `--rep-window N`, or turn it off with
`--rep-window 0`.

**Generation stops with `DeviceTimeout`.** The device did not set its done
status in time. The runtime reads STATUS up to 1,000,000 times for one
matmul. Raise the limit with `--poll-timeout N`, or add a delay between two
polls with `--poll-delay-us N`. A timeout that returns at the same matmul each
time points at the compute path, not at the link.

**Tokens are correct but slow.** This is expected. The UART is the bottleneck.
Run `--stats` to see the round trips and the time of each token. Raise
`--fp-baud`, or move the largest matrices on-chip with `--fp-bram-cache-kb`.

**The design does not meet timing.** Read the nextpnr report. Lower
`--fp-mhz`, but keep the baud an exact divisor of the fabric clock. 24 MHz
gives more margin than the default 30 MHz.

**DDR3 reads are unstable.** Sweep `--ddr-read-tap` from 0 to 127 and keep the
centre of the range that works.

## Debug without a board

`sim.zig` runs the full generate path in process. Use it to separate a model
problem from a hardware problem. If the simulator gives good tokens and the
board does not, the fault is in the bitstream, the transport or the weight
image.

The Python binding opens the simulator with `transport="sim"`, and the C ABI
opens it with `loom_device_open_sim`. See [runtime.md](runtime.md#bindings).

## Check the manifest

`loom.json` says where each matrix lives. Read it when a matrix looks wrong:

- `matrices[].store` is `flash` or `bram`.
- `matrices[].weight_offset` is the byte offset in that store.
- `csr_base`, `flash_weight_base` and `scale_flash_base` are bus addresses.

`loom-genip` also prints the split of hot and cold matrices when you build with
a BRAM cache. Compare that list against the manifest.
