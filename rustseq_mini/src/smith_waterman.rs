// core alignment logic

use std::arch::x86_64::*;

// smith-waterman algorithm with vectorized scoring
pub fn align(seq1: &str, seq2: &str) -> i32 {
    let bytes1 = seq1.as_bytes();
    let bytes2 = seq2.as_bytes();
    let len = bytes1.len().min(bytes2.len());

    let mut score = 0;

    unsafe {
        let chunk = 16; // 16x u8 fits in __m128i

        let mut i = 0;
        while i + chunk <= len {
            
        }
    }
}