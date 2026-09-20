//! Native memory-access checks: digest equality alone cannot detect overreads.
use super::{Rng, soft_digest};
use crate::{
    Digest, Sha224, Sha256,
    consts::{H256_224, H256_256},
};

pub(in crate::sha256) struct GuardedPage {
    base: *mut libc::c_void,
    pub(in crate::sha256) len: usize,
    read_only: bool,
}

impl GuardedPage {
    pub(in crate::sha256) fn new() -> Self {
        // SAFETY: sysconf takes no pointers. mmap reserves three inaccessible
        // pages; only the middle page is subsequently made readable/writable.
        unsafe {
            let len = libc::sysconf(libc::_SC_PAGESIZE);
            assert!(len >= 4096 && len % 64 == 0);
            let len = len as usize;
            let base = libc::mmap(
                core::ptr::null_mut(),
                3 * len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert_ne!(base, libc::MAP_FAILED);
            let mut page = Self {
                base,
                len,
                read_only: false,
            };
            assert_eq!(
                libc::mprotect(page.ptr().cast(), len, libc::PROT_READ | libc::PROT_WRITE),
                0
            );
            page.bytes_mut().fill(0xa5);
            page
        }
    }

    pub(in crate::sha256) fn ptr(&self) -> *mut u8 {
        // SAFETY: the mapping covers three pages; this is the middle one.
        unsafe { self.base.cast::<u8>().add(self.len) }
    }

    pub(in crate::sha256) fn bytes(&self) -> &[u8] {
        // SAFETY: the middle page is initialized and readable until Drop.
        unsafe { core::slice::from_raw_parts(self.ptr(), self.len) }
    }

    pub(in crate::sha256) fn bytes_mut(&mut self) -> &mut [u8] {
        assert!(!self.read_only);
        // SAFETY: &mut self excludes other borrows; the page is writable.
        unsafe { core::slice::from_raw_parts_mut(self.ptr(), self.len) }
    }

    pub(in crate::sha256) fn make_read_only(&mut self) {
        // SAFETY: the pointer/length describe exactly the mapped middle page.
        assert_eq!(
            unsafe { libc::mprotect(self.ptr().cast(), self.len, libc::PROT_READ) },
            0
        );
        self.read_only = true;
    }
}

impl Drop for GuardedPage {
    fn drop(&mut self) {
        // SAFETY: this owns the complete mapping, and no borrows survive Drop.
        assert_eq!(unsafe { libc::munmap(self.base, 3 * self.len) }, 0);
    }
}

#[test]
fn public_api_read_only_input_at_guard_pages() {
    let mut page = GuardedPage::new();
    Rng(0x23b1_9e75).fill(page.bytes_mut());
    page.make_read_only();
    let data = page.bytes();
    // Every length includes padding boundaries and every input alignment. The
    // empty suffix points at the inaccessible page and must not be read.
    for len in 0..=4096 {
        for input in [&data[..len], &data[data.len() - len..]] {
            assert_eq!(
                Sha224::digest(input)[..],
                soft_digest(input, H256_224)[..28],
                "len={len}"
            );
            assert_eq!(
                Sha256::digest(input)[..],
                soft_digest(input, H256_256),
                "len={len}"
            );
        }
    }
}
