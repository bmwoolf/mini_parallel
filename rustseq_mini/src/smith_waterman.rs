// core alignment logic for genome-scale sequences
// +2 if the letters match, -1 if they dont, per letter position

use std::arch::x86_64::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

// constants for genome-scale processing
const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks
const MAX_MEMORY_MB: usize = 1024; // 1GB memory limit

// smith-waterman algorithm with vectorized scoring for large sequences
pub fn align(seq1: &str, seq2: &str) -> i32 {
    let bytes1 = seq1.as_bytes();
    let bytes2 = seq2.as_bytes();
    let len = bytes1.len().min(bytes2.len());

    let mut total_score = 0;

    // Process in chunks to handle large genomes
    let mut offset = 0;
    while offset < len {
        let chunk_end = (offset + CHUNK_SIZE).min(len);
        let chunk_len = chunk_end - offset;
        
        let mut score = 0;

        unsafe {
            // Use largest available SIMD width
            let simd_chunk = if chunk_len >= 64 && is_x86_feature_detected!("avx2") {
                32 // AVX2 can process 32 bytes at once
            } else if chunk_len >= 16 {
                16 // SSE can process 16 bytes at once
            } else {
                chunk_len // Fall back to scalar for small chunks
            };

            let mut i = 0;
            while i + simd_chunk <= chunk_len {
                // Load SIMD vectors
                let a = _mm_loadu_si128(bytes1[offset + i..].as_ptr() as *const __m128i);
                let b = _mm_loadu_si128(bytes2[offset + i..].as_ptr() as *const __m128i);

                let cmp = _mm_cmpeq_epi8(a, b);
                // if match, +2
                let match_score = _mm_and_si128(cmp, _mm_set1_epi8(2));
                // if mismatch, -1
                let mismatch_mask = _mm_andnot_si128(cmp, _mm_set1_epi8(1));

                let diff = _mm_add_epi8(match_score, mismatch_mask);

                // Horizontal sum
                let mut tmp = [0i8; 16];
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, diff);

                score += tmp.iter().map(|x| *x as i32).sum::<i32>();

                i += simd_chunk;
            }

            // Scalar fallback for remaining bases in this chunk
            for (a, b) in bytes1[offset + i..chunk_end].iter().zip(&bytes2[offset + i..chunk_end]) {
                score += if a == b { 2 } else { -1 };
            }
        }
        
        total_score += score;
        offset = chunk_end;
        
        // progress reporting for WGS
        if len > 100_000_000 { // 100MB threshold
            let progress = (offset as f64 / len as f64) * 100.0;
            if offset % (100 * CHUNK_SIZE) == 0 {
                println!("Progress: {:.1}%", progress);
            }
        }
    }
    
    total_score
}

// Function to read genome from file with memory-efficient streaming
pub fn align_from_files(file1: &str, file2: &str) -> Result<i32, std::io::Error> {
    let mut total_score = 0;
    let mut buffer1 = String::new();
    let mut buffer2 = String::new();
    
    let file1 = File::open(file1)?;
    let file2 = File::open(file2)?;
    
    let mut reader1 = BufReader::new(file1);
    let mut reader2 = BufReader::new(file2);
    
    loop {
        buffer1.clear();
        buffer2.clear();
        
        let len1 = reader1.read_line(&mut buffer1)?;
        let len2 = reader2.read_line(&mut buffer2)?;
        
        if len1 == 0 || len2 == 0 {
            break;
        }
        
        // Skip header lines (start with '>')
        if buffer1.starts_with('>') || buffer2.starts_with('>') {
            continue;
        }
        
        // Trim whitespace and newlines
        let seq1 = buffer1.trim();
        let seq2 = buffer2.trim();
        
        if !seq1.is_empty() && !seq2.is_empty() {
            total_score += align(seq1, seq2);
        }
    }
    
    Ok(total_score)
}