//! SHA-256 with AVX2 message scheduling and BMI2 scalar rounds.
//!
//! The two 128-bit lanes schedule consecutive blocks in parallel.
//! Compression remains sequential:
//! the second block uses the chaining state produced by the first.
//! Scalar rounds defer the final Sigma0 addition until the next round.
//! LLVM owns the stack frame and callee-saved registers; the assembly never
//! adjusts the stack pointer.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_arch = "x86_64"))]
compile_error!("SHA-256 x86-avx2 backend requires x86_64");

use crate::consts::K32;
use core::arch::{asm, x86_64::*};

// Two copies of each group of four constants, one for each 128-bit lane.
// The zero group terminates the three iterations generating W16..W63.
#[repr(C, align(32))]
struct Constants([[u32; 8]; 17]);
static K: Constants = {
    let mut k = [[0u32; 8]; 17];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 8 {
            k[i][j] = K32[4 * i + j % 4];
            j += 1;
        }
        i += 1;
    }
    Constants(k)
};

// Keep b^c and a copy of f across rounds. Delay the final Sigma0 addition
// until the next round, overlapping it with work on e. t3 holds that value;
// it starts at zero and must be added to a after the last round.
// t0 is Sigma1 scratch; t1/t2 alternate between a^b and temporary storage;
// t4 retains f until Ch is computed, then becomes rotation scratch.
// Eight optional vector instructions interleave scheduling with the scalar work.
macro_rules! round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident; $tmp:ident, $bc:ident; $off:expr) => {
        round!($a, $b, $c, $d, $e, $f, $g, $h; $tmp, $bc; $off; "", "", "", "", "", "", "", "")
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident; $tmp:ident, $bc:ident; $off:expr; $v0:expr, $v1:expr, $v2:expr, $v3:expr, $v4:expr, $v5:expr, $v6:expr, $v7:expr) => {
        concat!(
            $v0,
            concat!("add {", stringify!($h), ":e}, dword ptr [{wk} + ", stringify!($off), "]\n"),
            concat!("xor {t4:e}, {", stringify!($g), ":e}\n"),
            concat!("rorx {t0:e}, {", stringify!($e), ":e}, 25\n"),
            $v1,
            concat!("rorx {", stringify!($tmp), ":e}, {", stringify!($e), ":e}, 11\n"),
            concat!("add {", stringify!($a), ":e}, {t3:e}\n"),
            concat!("and {t4:e}, {", stringify!($e), ":e}\n"),
            $v2,
            concat!("xor {t4:e}, {", stringify!($g), ":e}\n"),
            concat!("xor {t0:e}, {", stringify!($tmp), ":e}\n"),
            concat!("rorx {t3:e}, {", stringify!($e), ":e}, 6\n"),
            $v3,
            concat!("add {", stringify!($h), ":e}, {t4:e}\n"),
            "xor {t0:e}, {t3:e}\n",
            concat!("mov {", stringify!($tmp), ":e}, {", stringify!($a), ":e}\n"),
            $v4,
            concat!("rorx {t4:e}, {", stringify!($a), ":e}, 22\n"),
            concat!("add {", stringify!($h), ":e}, {t0:e}\n"),
            concat!("xor {", stringify!($tmp), ":e}, {", stringify!($b), ":e}\n"),
            $v5,
            concat!("rorx {t3:e}, {", stringify!($a), ":e}, 13\n"),
            concat!("rorx {t0:e}, {", stringify!($a), ":e}, 2\n"),
            concat!("add {", stringify!($d), ":e}, {", stringify!($h), ":e}\n"),
            $v6,
            concat!("and {", stringify!($bc), ":e}, {", stringify!($tmp), ":e}\n"),
            "xor {t3:e}, {t4:e}\n",
            concat!("xor {", stringify!($bc), ":e}, {", stringify!($b), ":e}\n"),
            $v7,
            "xor {t3:e}, {t0:e}\n",
            concat!("add {", stringify!($h), ":e}, {", stringify!($bc), ":e}\n"),
            concat!("mov {t4:e}, {", stringify!($e), ":e}\n"),
        )
    };
}

macro_rules! rounds4 {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident; $off:expr) => {
        concat!(
            round!($a, $b, $c, $d, $e, $f, $g, $h; t1, t2; $off + 0),
            round!($h, $a, $b, $c, $d, $e, $f, $g; t2, t1; $off + 4),
            round!($g, $h, $a, $b, $c, $d, $e, $f; t1, t2; $off + 8),
            round!($f, $g, $h, $a, $b, $c, $d, $e; t2, t1; $off + 12),
        )
    };
}

// Advance the 16-word window by four words per lane. W16/W17 depend on
// W14/W15, then W18/W19 depend on W16/W17. Duplicating each u32 in a u64
// lets 64-bit shifts implement sigma1's rotations; low/high discard the
// unwanted halves. Interleave scheduling with groups of three scalar instructions.
macro_rules! schedule_rounds4 {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident; $w0:ident, $w1:ident, $w2:ident, $w3:ident; $off:expr) => {
        concat!(
            round!($a, $b, $c, $d, $e, $f, $g, $h; t1, t2; $off + 0;
                concat!("vpalignr {v0}, {", stringify!($w1), "}, {", stringify!($w0), "}, 4\n"),
                concat!("vpalignr {v1}, {", stringify!($w3), "}, {", stringify!($w2), "}, 4\n"),
                concat!("vpaddd {", stringify!($w0), "}, {", stringify!($w0), "}, {v1}\n"),
                "vpsrld {v1}, {v0}, 7\n",
                "vpslld {v2}, {v0}, 25\n",
                "vpxor {v1}, {v1}, {v2}\n",
                "vpsrld {v2}, {v0}, 18\n",
                "vpxor {v1}, {v1}, {v2}\n"
            ),
            round!($h, $a, $b, $c, $d, $e, $f, $g; t2, t1; $off + 4;
                "vpslld {v2}, {v0}, 14\n",
                "vpxor {v1}, {v1}, {v2}\n",
                "vpsrld {v0}, {v0}, 3\n",
                "vpxor {v0}, {v0}, {v1}\n",
                concat!("vpaddd {", stringify!($w0), "}, {", stringify!($w0), "}, {v0}\n"),
                concat!("vpshufd {v0}, {", stringify!($w3), "}, 250\n"),
                "vpsrlq {v1}, {v0}, 17\n",
                "vpsrlq {v2}, {v0}, 19\n"
            ),
            round!($g, $h, $a, $b, $c, $d, $e, $f; t1, t2; $off + 8;
                "vpxor {v1}, {v1}, {v2}\n",
                "vpsrld {v0}, {v0}, 10\n",
                "vpxor {v1}, {v1}, {v0}\n",
                "vpshufb {v1}, {v1}, {low}\n",
                concat!("vpaddd {", stringify!($w0), "}, {", stringify!($w0), "}, {v1}\n"),
                concat!("vpshufd {v0}, {", stringify!($w0), "}, 80\n"),
                "vpsrlq {v1}, {v0}, 17\n",
                "vpsrlq {v2}, {v0}, 19\n"
            ),
            round!($f, $g, $h, $a, $b, $c, $d, $e; t2, t1; $off + 12;
                "vpxor {v1}, {v1}, {v2}\n",
                "vpsrld {v0}, {v0}, 10\n",
                "vpxor {v1}, {v1}, {v0}\n",
                "vpshufb {v1}, {v1}, {high}\n",
                concat!("vpaddd {", stringify!($w0), "}, {", stringify!($w0), "}, {v1}\n"),
                "",
                "",
                ""
            ),
            "vmovq {t0}, {vkp}\n",
            concat!("vpaddd {v0}, {", stringify!($w0), "}, ymmword ptr [{t0} + ", stringify!($off), "]\n"),
            concat!("vmovdqu ymmword ptr [{wk} + 128 + ", stringify!($off), "], {v0}\n"),
        )
    };
}

#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn compress_pair(state: &mut [u32; 8], w: [__m256i; 4], wk: *mut __m256i, len: usize) {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    let low = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,
    ));
    let high = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 8, 9, 10, 11,
    ));
    // SAFETY: the caller provides 544 writable bytes, len is 1 or 2, and the
    // required CPU features are available. The first 512 bytes hold W+K;
    // the last 32 save the chaining state for feed-forward. Every load follows
    // its store, and all constant loads stay within K.
    // All modified registers are operands. vkp keeps a pointer outside the
    // fourteen general registers used by the state, scalar scratch, and wk.
    asm!(
        "vmovq {pair_len}, {t0}",
        "mov dword ptr [{wk} + 512], {a:e}",
        "mov dword ptr [{wk} + 516], {b:e}",
        "mov dword ptr [{wk} + 520], {c:e}",
        "mov dword ptr [{wk} + 524], {d:e}",
        "mov dword ptr [{wk} + 528], {e:e}",
        "mov dword ptr [{wk} + 532], {f:e}",
        "mov dword ptr [{wk} + 536], {g:e}",
        "mov dword ptr [{wk} + 540], {h:e}",
        "mov {t4:e}, {f:e}",
        "mov {t2:e}, {b:e}",
        "xor {t2:e}, {c:e}",
        "xor {t3:e}, {t3:e}",
        "vpaddd {v0}, {w0}, ymmword ptr [rip + {constants} + 0]",
        "vmovdqu ymmword ptr [{wk} + 0], {v0}",
        "vpaddd {v0}, {w1}, ymmword ptr [rip + {constants} + 32]",
        "vmovdqu ymmword ptr [{wk} + 32], {v0}",
        "vpaddd {v0}, {w2}, ymmword ptr [rip + {constants} + 64]",
        "vmovdqu ymmword ptr [{wk} + 64], {v0}",
        "vpaddd {v0}, {w3}, ymmword ptr [rip + {constants} + 96]",
        "vmovdqu ymmword ptr [{wk} + 96], {v0}",
        "lea {t0}, [rip + {constants} + 128]",
        "vmovq {vkp}, {t0}",
        ".p2align 5",
        "2:",
        schedule_rounds4!(a, b, c, d, e, f, g, h; w0, w1, w2, w3; 0),
        schedule_rounds4!(e, f, g, h, a, b, c, d; w1, w2, w3, w0; 32),
        schedule_rounds4!(a, b, c, d, e, f, g, h; w2, w3, w0, w1; 64),
        schedule_rounds4!(e, f, g, h, a, b, c, d; w3, w0, w1, w2; 96),
        "add {wk}, 128",
        "vmovq {t0}, {vkp}",
        "add {t0}, 128",
        "vmovq {vkp}, {t0}",
        "cmp dword ptr [{t0}], 0",
        "jne 2b",
        rounds4!(a, b, c, d, e, f, g, h; 0),
        rounds4!(e, f, g, h, a, b, c, d; 32),
        rounds4!(a, b, c, d, e, f, g, h; 64),
        rounds4!(e, f, g, h, a, b, c, d; 96),
        "add {a:e}, {t3:e}",
        // wk advanced 384 bytes. Feed forward from the saved state at +512.
        "add {a:e}, dword ptr [{wk} + 128]",
        "add {b:e}, dword ptr [{wk} + 132]",
        "add {c:e}, dword ptr [{wk} + 136]",
        "add {d:e}, dword ptr [{wk} + 140]",
        "add {e:e}, dword ptr [{wk} + 144]",
        "add {f:e}, dword ptr [{wk} + 148]",
        "add {g:e}, dword ptr [{wk} + 152]",
        "add {h:e}, dword ptr [{wk} + 156]",
        "vmovq {t0}, {pair_len}",
        "cmp {t0}, 2",
        "jne 4f",
        // Keep the new state in registers, and save it for the second feed-forward.
        "mov dword ptr [{wk} + 128], {a:e}",
        "mov dword ptr [{wk} + 132], {b:e}",
        "mov dword ptr [{wk} + 136], {c:e}",
        "mov dword ptr [{wk} + 140], {d:e}",
        "mov dword ptr [{wk} + 144], {e:e}",
        "mov dword ptr [{wk} + 148], {f:e}",
        "mov dword ptr [{wk} + 152], {g:e}",
        "mov dword ptr [{wk} + 156], {h:e}",
        "sub {wk}, 384",
        "mov {t4:e}, {f:e}",
        "mov {t2:e}, {b:e}",
        "xor {t2:e}, {c:e}",
        "xor {t3:e}, {t3:e}",
        "lea {t0}, [{wk} + 512]",
        "vmovq {vkp}, {t0}",
        ".p2align 5",
        "3:",
        rounds4!(a, b, c, d, e, f, g, h; 16),
        rounds4!(e, f, g, h, a, b, c, d; 48),
        "add {wk}, 64",
        "vmovq {t0}, {vkp}",
        "cmp {wk}, {t0}",
        "jne 3b",
        "add {a:e}, {t3:e}",
        // wk now points just past the schedule, at the saved chaining state.
        "add {a:e}, dword ptr [{wk} + 0]",
        "add {b:e}, dword ptr [{wk} + 4]",
        "add {c:e}, dword ptr [{wk} + 8]",
        "add {d:e}, dword ptr [{wk} + 12]",
        "add {e:e}, dword ptr [{wk} + 16]",
        "add {f:e}, dword ptr [{wk} + 20]",
        "add {g:e}, dword ptr [{wk} + 24]",
        "add {h:e}, dword ptr [{wk} + 28]",
        "4:",
        a = inout(reg) a, b = inout(reg) b, c = inout(reg) c, d = inout(reg) d,
        e = inout(reg) e, f = inout(reg) f, g = inout(reg) g, h = inout(reg) h,
        t0 = inout(reg) len => _, t1 = out(reg) _, t2 = out(reg) _,
        t3 = out(reg) _, t4 = out(reg) _,
        wk = inout(reg) wk => _,
        constants = sym K, vkp = out(xmm_reg) _, pair_len = out(xmm_reg) _,
        w0 = inout(ymm_reg) w[0] => _, w1 = inout(ymm_reg) w[1] => _,
        w2 = inout(ymm_reg) w[2] => _, w3 = inout(ymm_reg) w[3] => _,
        v0 = out(ymm_reg) _, v1 = out(ymm_reg) _, v2 = out(ymm_reg) _,
        low = in(ymm_reg) low, high = in(ymm_reg) high,
        options(nostack),
    );
    *state = [a, b, c, d, e, f, g, h];
}

// Inlining into a caller with a realigned stack and frame pointer can reserve
// an extra register, leaving too few for the assembly's fourteen GPR operands.
#[inline(never)]
#[target_feature(enable = "avx2,bmi2")]
pub(super) unsafe fn compress(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    let shuffle = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    ));
    for pair in blocks.chunks(2) {
        let p0 = pair[0].as_ptr();
        // Duplicate an odd tail without reading past the input.
        let p1 = pair.get(1).unwrap_or(&pair[0]).as_ptr();
        let mut w = [_mm256_setzero_si256(); 4];
        for i in 0..4 {
            w[i] = _mm256_shuffle_epi8(
                _mm256_set_m128i(
                    _mm_loadu_si128(p1.add(16 * i).cast()),
                    _mm_loadu_si128(p0.add(16 * i).cast()),
                ),
                shuffle,
            );
        }
        // Sixteen schedule vectors and one vector-sized slot for the state.
        let mut wk = core::mem::MaybeUninit::<[__m256i; 17]>::uninit();
        compress_pair(state, w, wk.as_mut_ptr().cast(), pair.len());
    }
}

#[cfg(test)]
mod tests;
