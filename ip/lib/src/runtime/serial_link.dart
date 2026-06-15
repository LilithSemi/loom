import 'dart:io';
import 'dart:typed_data';

import 'loom_device.dart';

/// A [LoomLink] over a serial character device (e.g. /dev/ttyACM0 from the
/// OrangeCrab's DirtyJTAG CDC), speaking the SAME command frames the
/// LoomUartBridge / LoomUsbDevice command engine parses (and the Zig runtime's
/// protocol.zig encodes):
///
///   WRITE (0x01): [op] [addr:u32 LE] [len:u16 LE] [data x len]
///   READ  (0x02): [op] [addr:u32 LE] [len:u16 LE]  -> device returns len bytes
///
/// Configure the port first (raw, matching baud), e.g.:
///   stty -F /dev/ttyACM0 115200 raw -echo -echoe -echok
///
/// The framing matches protocol.zig; host orchestration is validated against
/// [EmulatedLoomDevice], which meets the same contract.
class SerialLoomLink extends LoomLink {
  // Dart's FileMode has no O_RDWR, so open the tty twice: one fd to write
  // command frames, one to read responses. Both reference the same terminal.
  final RandomAccessFile _wr;
  final RandomAccessFile _rd;
  final Duration readTimeout;

  SerialLoomLink._(this._wr, this._rd, this.readTimeout, this.writeChunkPause);

  /// Open [path]. Configures the tty (raw, 115200, VMIN=0/VTIME so reads return
  /// promptly) and drains any stale RX (the DirtyJTAG CDC emits a spurious byte
  /// on open, which otherwise desyncs every framed read).
  factory SerialLoomLink(
    String path, {
    int baud = 115200,
    Duration readTimeout = const Duration(seconds: 5),
    Duration writeChunkPause = const Duration(milliseconds: 3),
  }) {
    // Own the termios setup so callers don't have to remember stty. min 0 /
    // time 2 (VTIME=0.2s) makes readSync return available bytes without blocking
    // forever, which the timeout loop below relies on.
    Process.runSync('stty', [
      '-F',
      path,
      '$baud',
      'raw',
      '-echo',
      '-echoe',
      '-echok',
      '-icrnl',
      '-ixon',
      'min',
      '0',
      'time',
      '2',
    ]);
    final rd = File(path).openSync(mode: FileMode.read);
    final wr = File(
      path,
    ).openSync(mode: FileMode.write); // O_TRUNC no-op on tty
    final link = SerialLoomLink._(wr, rd, readTimeout, writeChunkPause);
    link._drain();
    return link;
  }

  /// Flush any pending RX so framed reads stay aligned.
  void _drain() {
    for (var i = 0; i < 8; i++) {
      if (_rd.readSync(256).isEmpty) break;
    }
  }

  void close() {
    _wr.closeSync();
    _rd.closeSync();
  }

  void _frame(int op, int addr, int len, [List<int>? data]) {
    final hdr = Uint8List(7);
    hdr[0] = op;
    hdr[1] = addr & 0xFF;
    hdr[2] = (addr >> 8) & 0xFF;
    hdr[3] = (addr >> 16) & 0xFF;
    hdr[4] = (addr >> 24) & 0xFF;
    hdr[5] = len & 0xFF;
    hdr[6] = (len >> 8) & 0xFF;
    _wr.writeFromSync(hdr);
    if (data != null && data.isNotEmpty) {
      _wr.writeFromSync(data is Uint8List ? data : Uint8List.fromList(data));
    }
    // No flushSync: it's invalid (EINVAL) on a character device. WriteFromSync
    // already issues the write(2) syscall directly.
  }

  /// Max data bytes per WRITE frame. A single oversized frame overruns the
  /// device-side command buffer and desyncs the bridge (a ~8KB weight image
  /// wedges it; <=2KB is fine), so big writes are split into several frames at
  /// incrementing addresses. Many small WRITE frames are fine (CSR pushes do it).
  static const int maxWriteChunk = 1024;

  /// Inter-chunk pause on large writes. The UART bridge has no flow control, so
  /// back-to-back multi-KB weight writes overrun the device RX (it is issuing a
  /// Wishbone write per word to SRAM, slower than the wire). A short pause per
  /// chunk lets it drain. Without it, weight images >~2KB desync the bridge.
  final Duration writeChunkPause;

  @override
  void write(int addr, List<int> bytes) {
    // A large multi-chunk write (weight image) desyncs the bridge on its own
    // (device-side buffer limit, not rate: images >~2KB wedge regardless of
    // chunk size/pauses). Read the chunk back after each frame: the READ forces
    // the command engine to finish the pending writes and reply, which keeps the
    // stream frame-aligned. Only for genuine multi-chunk writes. CSR pushes stay
    // plain (reading back actPush/scalePush would consume the wrong value).
    final barrier = bytes.length > maxWriteChunk;
    var off = 0;
    while (off < bytes.length) {
      final n = (bytes.length - off) < maxWriteChunk
          ? (bytes.length - off)
          : maxWriteChunk;
      _frame(0x01, addr + off, n, bytes.sublist(off, off + n));
      if (barrier) {
        read(addr + off, 4); // resync barrier + readback
      }
      off += n;
    }
  }

  @override
  List<int> read(int addr, int len) {
    _frame(0x02, addr, len);
    final out = Uint8List(len);
    var got = 0;
    final deadline = DateTime.now().add(readTimeout);
    while (got < len) {
      final chunk = _rd.readSync(len - got);
      if (chunk.isEmpty) {
        if (DateTime.now().isAfter(deadline)) {
          throw TimeoutException(
            'serial read of $len bytes at '
            '0x${addr.toRadixString(16)} timed out (got $got)',
          );
        }
        continue;
      }
      out.setRange(got, got + chunk.length, chunk);
      got += chunk.length;
    }
    return out;
  }
}

class TimeoutException implements Exception {
  final String message;
  TimeoutException(this.message);
  @override
  String toString() => 'TimeoutException: $message';
}
