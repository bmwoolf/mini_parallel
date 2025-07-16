use needletail::parse_fastx_file;
use dashmap::DashMap;
use rayon::prelude::*;
use fxhash::hash64;

pub fn count_kmers(filename: &str, k: usize) {
    let reader = parse_fastx_file(filename).expect("Invalid FASTQ/FASTA file");
    let counts = DashMap::new();

    reader.into_iter().par_bridge().for_each(|record| {
        let seqrec = record.expect("Parse error");
        let seq = seqrec.seq();

        for i in 0..=seq.len().saturating_sub(k) {
            let kmer = &seq[i..i + k];
            let hash = hash64(kmer);
            counts.entry(hash).and_modify(|e| *e += 1).or_insert(1);
        }
    });

    for item in counts.iter() {
        println!("{}\t{}", item.key(), item.value());
    }
}
