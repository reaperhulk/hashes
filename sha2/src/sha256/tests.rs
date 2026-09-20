use super::soft;

#[cfg(all(target_arch = "x86_64", unix))]
pub(super) mod guard;

pub(super) struct Rng(pub(super) u32);

impl Rng {
    pub(super) fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    pub(super) fn fill(&mut self, bytes: &mut [u8]) {
        for byte in bytes {
            *byte = self.next() as u8;
        }
    }
}

pub(super) fn blocks(input: &[u8]) -> &[[u8; 64]] {
    assert_eq!(input.len() % 64, 0);
    // SAFETY: [u8; 64] has alignment 1; the slice contains only complete blocks.
    unsafe { core::slice::from_raw_parts(input.as_ptr().cast(), input.len() / 64) }
}

// Independent padding plus the scalar compressor provides an oracle for both
// public hashers, including buffering and SHA-224's different IV/truncation.
fn soft_digest(input: &[u8], mut state: [u32; 8]) -> [u8; 32] {
    let full = input.len() / 64 * 64;
    soft::compress(&mut state, blocks(&input[..full]));
    let remainder = &input[full..];
    let mut padding = [0u8; 128];
    padding[..remainder.len()].copy_from_slice(remainder);
    padding[remainder.len()] = 0x80;
    let len = if remainder.len() < 56 { 64 } else { 128 };
    padding[len - 8..len].copy_from_slice(&((input.len() as u64) * 8).to_be_bytes());
    soft::compress(&mut state, blocks(&padding[..len]));
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[test]
fn public_api_streaming_boundaries_and_random() {
    use crate::{
        Digest, Sha224, Sha256,
        consts::{H256_224, H256_256},
    };

    // Exercise the selected backend through the public API.
    let mut rng = Rng(0x6d2b_79f5);
    let mut data = [0u8; 16 * 1024 + 63];
    let lengths = [
        511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097, 16383, 16384,
    ];
    let chunks = [
        1, 2, 3, 31, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 255, 256, 257, 1024,
    ];
    for case in 0..1536 {
        let len = if case <= 257 {
            case
        } else if case % 2 == 0 {
            lengths[rng.next() as usize % lengths.len()]
        } else {
            rng.next() as usize % (16 * 1024 + 1)
        };
        let offset = rng.next() as usize % 64;
        let input = &mut data[offset..offset + len];
        rng.fill(input);
        let expected224 = soft_digest(input, H256_224);
        let expected256 = soft_digest(input, H256_256);
        assert_eq!(
            Sha224::digest(&*input)[..],
            expected224[..28],
            "case={case}"
        );
        assert_eq!(Sha256::digest(&*input)[..], expected256, "case={case}");
        let mut h224 = Sha224::new();
        let mut h256 = Sha256::new();
        let mut pos = 0;
        while pos < len {
            h224.update([]);
            h256.update([]);
            let chunk = if rng.next() & 1 == 0 {
                chunks[rng.next() as usize % chunks.len()]
            } else {
                1 + rng.next() as usize % 2048
            };
            let end = (pos + chunk).min(len);
            h224.update(&input[pos..end]);
            h256.update(&input[pos..end]);
            pos = end;
        }
        assert_eq!(
            h224.finalize()[..],
            expected224[..28],
            "chunked case={case}"
        );
        assert_eq!(h256.finalize()[..], expected256, "chunked case={case}");
    }
}
