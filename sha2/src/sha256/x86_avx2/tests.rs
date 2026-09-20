use super::*;
use crate::sha256::{
    soft,
    tests::{Rng, blocks},
};

cpufeatures::new!(avx2_cpuid, "avx2", "bmi2");

#[test]
fn matches_soft() {
    if !avx2_cpuid::get() {
        return;
    }
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut input = [0u8; 64 * 17 + 63];
    for byte in &mut input {
        *byte = next() as u8;
    }
    // Exercise unaligned loads, empty input, odd tails, and multiple block pairs.
    for offset in 0..64 {
        for len in 0..=17 {
            // SAFETY: the slice contains `len` complete blocks; [u8; 64] has alignment 1.
            let blocks = unsafe {
                core::slice::from_raw_parts(input[offset..].as_ptr().cast::<[u8; 64]>(), len)
            };
            let initial = core::array::from_fn(|_| next());
            let mut expected = initial;
            soft::compress(&mut expected, blocks);
            let mut actual = initial;
            // SAFETY: the required features were detected above.
            unsafe { compress(&mut actual, blocks) };
            assert_eq!(actual, expected, "offset={offset}, blocks={len}");
            for split in 0..=len {
                let mut split_state = initial;
                // SAFETY: the required features were detected above.
                unsafe {
                    compress(&mut split_state, &blocks[..split]);
                    compress(&mut split_state, &blocks[split..]);
                }
                assert_eq!(split_state, expected, "blocks={len}, split={split}");
            }
        }
    }
}

#[repr(C)]
struct CheckedState {
    before: [u32; 8],
    state: [u32; 8],
    after: [u32; 8],
}

impl CheckedState {
    fn new(state: [u32; 8]) -> Self {
        Self {
            before: [0xa5a5_a5a5; 8],
            state,
            after: [0x5a5a_5a5a; 8],
        }
    }

    fn check(&self) {
        // SAFETY: both fields are initialized and aligned. Volatile reads ensure
        // the compiler checks the memory, rather than substituting the canaries.
        unsafe {
            assert_eq!(core::ptr::read_volatile(&self.before), [0xa5a5_a5a5; 8]);
            assert_eq!(core::ptr::read_volatile(&self.after), [0x5a5a_5a5a; 8]);
        }
    }
}

#[test]
fn compression_patterns_and_random() {
    if !avx2_cpuid::get() {
        return;
    }
    let mut rng = Rng(0x91e1_0da5);
    let mut input = [0u8; 64 * 257 + 63];
    for case in 0..2048 {
        let len = if case <= 257 {
            case
        } else {
            rng.next() as usize % 258
        };
        let offset = rng.next() as usize % 64;
        let input = &mut input[offset..offset + 64 * len];
        match case % 8 {
            0 => input.fill(0),
            1 => input.fill(0xff),
            2 => input.fill(0xaa),
            3 => input.fill(0x55),
            4 => input.iter_mut().enumerate().for_each(|(i, b)| *b = i as u8),
            5 => input
                .iter_mut()
                .enumerate()
                .for_each(|(i, b)| *b = 1 << (i % 8)),
            _ => rng.fill(input),
        }
        let initial = match case % 4 {
            0 => [0; 8],
            1 => [u32::MAX; 8],
            _ => core::array::from_fn(|_| rng.next()),
        };
        let blocks = blocks(input);
        let mut expected = initial;
        soft::compress(&mut expected, blocks);
        let mut actual = CheckedState::new(initial);
        // SAFETY: AVX2 and BMI2 were detected above.
        unsafe { compress(&mut actual.state, blocks) };
        actual.check();
        assert_eq!(
            actual.state, expected,
            "case={case}, offset={offset}, blocks={len}"
        );

        let mut split = CheckedState::new(initial);
        let mut pos = 0;
        while pos < len {
            let end = (pos + 1 + rng.next() as usize % 33).min(len);
            // SAFETY: AVX2 and BMI2 were detected above.
            unsafe {
                compress(&mut split.state, &[]);
                compress(&mut split.state, &blocks[pos..end]);
            }
            split.check();
            pos = end;
        }
        assert_eq!(split.state, expected, "split case={case}, blocks={len}");
    }
}

#[cfg(unix)]
#[test]
fn compression_state_at_guard_pages() {
    use crate::sha256::tests::guard::GuardedPage;

    if !avx2_cpuid::get() {
        return;
    }
    let mut rng = Rng(0x8b4f_a103);
    let mut input_page = GuardedPage::new();
    rng.fill(input_page.bytes_mut());
    input_page.make_read_only();
    let data = input_page.bytes();
    let mut state_page = GuardedPage::new();
    for len in (0..=4096).step_by(64) {
        for input in [&data[..len], &data[data.len() - len..]] {
            let initial = core::array::from_fn(|_| rng.next());
            let mut expected = initial;
            soft::compress(&mut expected, blocks(input));
            // Every valid state alignment within a cache line, plus the last
            // 32 bytes before the trailing guard page.
            for offset in (0..64).step_by(4).chain([state_page.len - 32]) {
                state_page.bytes_mut().fill(0xa5);
                // SAFETY: all offsets are u32-aligned with 32 writable bytes.
                // CPU features were checked above; the input has full blocks.
                unsafe {
                    let state = state_page.ptr().add(offset).cast::<[u32; 8]>();
                    state.write(initial);
                    compress(&mut *state, blocks(input));
                    assert_eq!(state.read(), expected, "len={len}, state offset={offset}");
                    // Detect writes outside the state even on its mapped side.
                    // Volatile reads prevent substitution of the fill value.
                    for i in (0..offset).chain(offset + 32..state_page.len) {
                        assert_eq!(
                            state_page.ptr().add(i).read_volatile(),
                            0xa5,
                            "canary byte={i}"
                        );
                    }
                }
            }
        }
    }
}
