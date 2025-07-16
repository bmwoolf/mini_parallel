// core alignment logic
// +2 if the letters match, -1 if they dont, per letter postion

use std::arch::x86_64::*;

// example of SIMD inner loop
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
            // loads 16 bytes into a 128 bit register (1 byte per 1 letter)
            let a = _mm_loadu_si128(bytes1[i..].as_ptr() as *const __m128i);
            // loads 16 bytes into a 128 bit register
            let b = _mm_loadu_si128(bytes2[i..].as_ptr() as *const __m128i);

            let cmp = _mm_cmpeq_epi8(a, b);
            // if match, +2
            let match_score = _mm_and_si128(cmp, _mm_set1_epi8(2));
            // if mismatch, -1
            let mismatch_mask = _mm_andnot_si128(cmp, _mm_set1_epi8(1));

            let diff = _mm_add_epi8(match_score, mismatch_mask);

            // horizontal sum
            let mut tmp = [0i8; 16];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, diff);

            score += tmp.iter().map(|x| *x as i32).sum::<i32>();

            i += chunk;
        }

        // scalar fallback for remaining bases
        for (a, b) in bytes1[i..].iter().zip(&bytes2[i..]) {
            score += if a == b { 2 } else { -1 };
        }
    }
    
    score
}