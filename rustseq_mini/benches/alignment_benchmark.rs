use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustseq_mini::smith_waterman;

fn generate_test_sequence(size: usize) -> String {
    let bases = ['A', 'C', 'G', 'T'];
    (0..size)
        .map(|_| bases[fastrand::u8(0..4) as usize])
        .collect()
}

fn alignment_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");
    
    // Test different sequence sizes
    for size in [100, 1000, 10000, 100000] {
        let seq1 = generate_test_sequence(size);
        let seq2 = generate_test_sequence(size);
        
        group.bench_function(&format!("cpu_{}", size), |b| {
            b.iter(|| smith_waterman::align(black_box(&seq1), black_box(&seq2)))
        });
    }
    
    group.finish();
}

criterion_group!(benches, alignment_benchmark);
criterion_main!(benches); 