// CUDA kernel for Smith-Waterman sequence alignment
// Optimized for RTX 4070 (Ada Lovelace architecture, SM 8.9)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for RTX 4070 optimization
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_BLOCKS 65535

// Shared memory for block-level reduction
__shared__ int shared_scores[THREADS_PER_BLOCK];

// Main Smith-Waterman alignment kernel
extern "C" __global__ void smith_waterman_kernel(
    const unsigned char* __restrict__ seq1,
    const unsigned char* __restrict__ seq2,
    int* __restrict__ result,
    const int sequence_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_score = 0;
    
    // Process multiple elements per thread for better occupancy
    const int elements_per_thread = 4;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int i = idx; i < sequence_length; i += total_threads) {
        // Load sequences with coalesced memory access
        unsigned char a = seq1[i];
        unsigned char b = seq2[i];
        
        // Smith-Waterman scoring: +2 for match, -1 for mismatch
        local_score += (a == b) ? 2 : -1;
    }
    
    // Store local score in shared memory
    shared_scores[threadIdx.x] = local_score;
    __syncthreads();
    
    // Block-level reduction using warp shuffle
    cg::thread_block block = cg::this_thread_block();
    
    // Reduce within warps first
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(0xffffffff, local_score, offset);
    }
    
    // Write warp results to shared memory
    if (threadIdx.x % WARP_SIZE == 0) {
        shared_scores[threadIdx.x / WARP_SIZE] = local_score;
    }
    __syncthreads();
    
    // Final reduction across warps (only first thread in block)
    if (threadIdx.x == 0) {
        int block_score = 0;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = 0; i < num_warps; i++) {
            block_score += shared_scores[i];
        }
        
        // Atomic add to global result
        atomicAdd(result, block_score);
    }
}

// Optimized kernel for large sequences with memory coalescing
extern "C" __global__ void smith_waterman_large_kernel(
    const unsigned char* __restrict__ seq1,
    const unsigned char* __restrict__ seq2,
    int* __restrict__ result,
    const int sequence_length,
    const int chunk_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_score = 0;
    
    // Process chunks for better memory locality
    for (int chunk_start = 0; chunk_start < sequence_length; chunk_start += chunk_size) {
        int chunk_end = min(chunk_start + chunk_size, sequence_length);
        
        for (int i = chunk_start + tid; i < chunk_end; i += gridDim.x * blockDim.x) {
            unsigned char a = seq1[i];
            unsigned char b = seq2[i];
            local_score += (a == b) ? 2 : -1;
        }
    }
    
    // Block-level reduction
    shared_scores[threadIdx.x] = local_score;
    __syncthreads();
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(0xffffffff, local_score, offset);
    }
    
    if (threadIdx.x % WARP_SIZE == 0) {
        shared_scores[threadIdx.x / WARP_SIZE] = local_score;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        int block_score = 0;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = 0; i < num_warps; i++) {
            block_score += shared_scores[i];
        }
        
        atomicAdd(result, block_score);
    }
}

// Kernel for 16-file batch processing
extern "C" __global__ void smith_waterman_batch_kernel(
    const unsigned char* __restrict__ sequences,
    int* __restrict__ results,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ sequence_offsets,
    const int num_pairs
) {
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;
    
    int seq1_offset = sequence_offsets[pair_idx * 2];
    int seq2_offset = sequence_offsets[pair_idx * 2 + 1];
    int seq1_len = sequence_lengths[pair_idx * 2];
    int seq2_len = sequence_lengths[pair_idx * 2 + 1];
    int min_len = min(seq1_len, seq2_len);
    
    int tid = threadIdx.x;
    int local_score = 0;
    
    // Process sequence pair
    for (int i = tid; i < min_len; i += blockDim.x) {
        unsigned char a = sequences[seq1_offset + i];
        unsigned char b = sequences[seq2_offset + i];
        local_score += (a == b) ? 2 : -1;
    }
    
    // Block reduction
    shared_scores[tid] = local_score;
    __syncthreads();
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(0xffffffff, local_score, offset);
    }
    
    if (tid % WARP_SIZE == 0) {
        shared_scores[tid / WARP_SIZE] = local_score;
    }
    __syncthreads();
    
    if (tid == 0) {
        int block_score = 0;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = 0; i < num_warps; i++) {
            block_score += shared_scores[i];
        }
        
        results[pair_idx] = block_score;
    }
}

// Memory management helpers
extern "C" __device__ void* allocate_shared_memory(size_t size) {
    extern __shared__ char shared_memory[];
    return shared_memory;
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0) 