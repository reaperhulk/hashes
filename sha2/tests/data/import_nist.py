#!/usr/bin/env python3
"""Reproduce the SHA-224/SHA-256 NIST fixtures from the pinned official ZIP.

Usage: python3 import_nist.py /path/to/shabytetestvectors.zip
The URL and archive checksum are recorded in NIST.md. No downloads or external
Python packages are needed. The output uses blobby 0.4 without deduplication.
"""

import hashlib
from pathlib import Path
import re
import sys
import zipfile

ARCHIVE_SHA256 = "929ef80b7b3418aca026643f6f248815913b60e01741a44bba9e118067f4c9b8"


def vlq(value):
    encoded = [value & 0x7f]
    while value > 0x7f:
        value = (value >> 7) - 1
        encoded.append(0x80 | (value & 0x7f))
    return bytes(reversed(encoded))


def write_blobs(name, blobs):
    data = vlq(len(blobs)) + vlq(0)  # item count, empty deduplication table
    data += b"".join(vlq(len(blob) << 1) + blob for blob in blobs)
    Path(__file__).with_name(name + ".blb").write_bytes(data)
    print(f"{name}: {len(blobs)} blobs, {len(data)} bytes")


def import_vectors(archive):
    raw = Path(archive).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA256, "Unexpected NIST archive"
    with zipfile.ZipFile(archive) as z:
        for bits in (224, 256):
            algorithm = getattr(hashlib, f"sha{bits}")
            for kind, count in (("Short", 65), ("Long", 64)):
                text = z.read(f"shabytetestvectors/SHA{bits}{kind}Msg.rsp").decode("ascii")
                records = re.findall(r"Len = (\d+)\s+Msg = ([0-9a-f]+)\s+MD = ([0-9a-f]+)", text)
                assert len(records) == count == text.count("MD =")
                blobs = []
                for length, message, digest in records:
                    length = int(length)
                    assert length % 8 == 0
                    message, digest = bytes.fromhex(message), bytes.fromhex(digest)
                    # CAVS represents an empty input as Len=0, Msg=00.
                    if length == 0:
                        assert message == b"\0"
                        message = b""
                    assert len(message) == length // 8
                    assert algorithm(message).digest() == digest
                    blobs.extend((message, digest))
                if kind == "Long":
                    # Retain one 100-block known-answer test without shipping
                    # all 64 large inputs in every published crate download.
                    longest = max(range(0, len(blobs), 2), key=lambda i: len(blobs[i]))
                    blobs = blobs[longest:longest + 2]
                    assert len(blobs[0]) == 6400
                write_blobs(f"sha{bits}_nist_{kind.lower()}", blobs)

            text = z.read(f"shabytetestvectors/SHA{bits}Monte.rsp").decode("ascii")
            seed = bytes.fromhex(re.search(r"Seed = ([0-9a-f]+)", text)[1])
            records = re.findall(r"COUNT = (\d+)\s+MD = ([0-9a-f]+)", text)
            assert len(records) == 100 == text.count("MD =")
            blobs = [seed]
            for index, (count, digest) in enumerate(records):
                assert int(count) == index
                a = b = c = seed
                for _ in range(1000):
                    a, b, c = b, c, algorithm(a + b + c).digest()
                seed = c
                expected = bytes.fromhex(digest)
                assert seed == expected
                blobs.append(expected)
            write_blobs(f"sha{bits}_nist_monte", blobs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import_vectors(sys.argv[1])
