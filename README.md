# STATUS: PAUSED
it works, but keeps maxing out my gpu. after debugging memory and storage limits, i am done. i am building a gpu tool for cloud providers since i have so many projects beyond this. but learned a lot about gpus and actual hpc techniques, albeit introductory and small on 1 gpu

# mini_parallel
mini core bioinformatics algorithms 

## Smith-Waterman
DNA sequence alignment using SIMD instructions. Compares two DNA sequences and scores how well the letters line up. Match gets +2 and Mismatch gets -1.

A = A (+2)  
T = T (+2)  
C = T (-1)  
G = G (+2)  
T = G (-1)  
...

## For my WGS:
Direct alignment: compare to average reference genome  
Complementary alignment: find what % of genome is not perfectly complementary (boooo)

## Setup for WGS Processing

### Environment Configuration
Create a `.env` file in the project root with your WGS data configuration:

```bash
# WGS Data Configuration
WGS_DATA_DIR=/path/to/your/wgs/data
WGS_SAMPLE_ID=your-sample-id
WGS_LANES=8
WGS_READS_PER_LANE=2

# GPU Configuration
GPU_CHUNK_SIZE_READS=10000
GPU_CHUNK_SIZE_BASES=1000000
```

### Usage
```bash
# Test WGS file reading
cargo run -- --test-wgs --gpu

# Process full WGS dataset
cargo run -- --full-wgs --gpu

# Run with Nsight Systems
nsys profile -t opencl,cuda,osrt --output wgs_profile ./target/release/rustseq_mini --full-wgs --gpu
```

### File Naming Convention
The aligner expects files named: `{SAMPLE_ID}_L{LANE:03}_R{READ}_001.fastq.gz`
- Example: `SAMPLE_001_L001_R1_001.fastq.gz`