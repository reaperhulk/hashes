//! NIST CAVS byte-oriented SHA-224/SHA-256 vectors (see data/NIST.md).
use digest::{Digest, Output, dev::fixed_reset_test, new_test};
use sha2::{Sha224, Sha256};

new_test!(sha224_nist_short, Sha224, fixed_reset_test);
new_test!(sha224_nist_long, Sha224, fixed_reset_test);
new_test!(sha256_nist_short, Sha256, fixed_reset_test);
new_test!(sha256_nist_long, Sha256, fixed_reset_test);

// SHAVS Monte Carlo: MD0 = MD1 = MD2 = Seed, then
// MDi = SHA(MD[i-3] || MD[i-2] || MD[i-1]), for i=3..=1002.
// Check MD1002 and use it as the seed of the next of 100 outer iterations.
fn monte_carlo<D: Digest>(vectors: &[&[u8]]) {
    assert_eq!(vectors.len(), 101); // Initial seed plus 100 expected digests.
    let mut seed = Output::<D>::default();
    seed.copy_from_slice(vectors[0]);
    for (count, expected) in vectors[1..].iter().enumerate() {
        let mut a = seed.clone();
        let mut b = seed.clone();
        let mut c = seed.clone();
        for _ in 0..1000 {
            let mut h = D::new();
            h.update(&a);
            h.update(&b);
            h.update(&c);
            a = b;
            b = c;
            c = h.finalize();
        }
        seed = c;
        assert_eq!(&seed[..], *expected, "Monte Carlo COUNT={count}");
    }
}

#[test]
fn sha224_nist_monte() {
    monte_carlo::<Sha224>(digest::dev::blobby::parse_into_slice!(include_bytes!(
        "data/sha224_nist_monte.blb"
    )));
}

#[test]
fn sha256_nist_monte() {
    monte_carlo::<Sha256>(digest::dev::blobby::parse_into_slice!(include_bytes!(
        "data/sha256_nist_monte.blb"
    )));
}
