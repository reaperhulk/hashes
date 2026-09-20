#[derive(Debug, Eq, PartialEq)]
pub(super) enum Backend {
    ShaNi,
    Avx2,
    Soft,
}

// Each probe covers all required features: SHA + SSE4.1, or AVX2 + BMI2.
// Defer the second probe so SHA-NI hosts only check the preferred backend.
#[inline(always)]
pub(super) fn select(sha_ni: bool, avx2_bmi2: impl FnOnce() -> bool) -> Backend {
    if sha_ni {
        Backend::ShaNi
    } else if avx2_bmi2() {
        Backend::Avx2
    } else {
        Backend::Soft
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, select};

    #[test]
    fn backend_priority() {
        for (sha_ni, avx2_bmi2, expected) in [
            (true, true, Backend::ShaNi),
            (true, false, Backend::ShaNi),
            (false, true, Backend::Avx2),
            (false, false, Backend::Soft),
        ] {
            assert_eq!(
                select(sha_ni, || avx2_bmi2),
                expected,
                "SHA-NI={sha_ni}, AVX2/BMI2={avx2_bmi2}"
            );
        }
    }

    #[test]
    fn sha_ni_skips_avx2_probe() {
        assert_eq!(
            select(true, || panic!("AVX2 probe must not run on SHA-NI hosts")),
            Backend::ShaNi
        );
    }
}
