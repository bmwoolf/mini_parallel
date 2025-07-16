// OpenCL kernel for Smith-Waterman sequence alignment
// Optimized for parallel processing of DNA sequences

// Scoring parameters
#define MATCH_SCORE 2
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -2

// Smith-Waterman alignment kernel
// Each work item processes a portion of the alignment matrix
__kernel void smith_waterman_align(
    __global const uchar* seq1,
    __global const uchar* seq2,
    __global int* result,
    uint length
) {
    uint gid = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    
    // Shared memory for local computation
    __local int local_scores[256]; // Assuming max work group size of 256
    
    // Each work group processes a chunk of the alignment
    uint chunk_size = (length + get_num_groups(0) - 1) / get_num_groups(0);
    uint start_pos = group_id * chunk_size;
    uint end_pos = min(start_pos + chunk_size, length);
    
    if (start_pos >= length) {
        return;
    }
    
    // Local variables for this work item's computation
    int max_score = 0;
    int current_score = 0;
    
    // Process assigned portion of the sequence
    for (uint i = start_pos + local_id; i < end_pos; i += group_size) {
        if (i < length) {
            // Simple scoring for this position
            int score = 0;
            if (seq1[i] == seq2[i]) {
                score = MATCH_SCORE;
            } else {
                score = MISMATCH_PENALTY;
            }
            
            // Update current score (simplified Smith-Waterman)
            current_score = max(current_score + score, 0);
            max_score = max(max_score, current_score);
        }
    }
    
    // Store local result
    local_scores[local_id] = max_score;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction to find maximum score in work group
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_scores[local_id] = max(local_scores[local_id], local_scores[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result if this is the first work item in the group
    if (local_id == 0) {
        atomic_max(result, local_scores[0]);
    }
}

// Alternative kernel for more detailed Smith-Waterman implementation
__kernel void smith_waterman_detailed(
    __global const uchar* seq1,
    __global const uchar* seq2,
    __global int* result,
    uint len1,
    uint len2
) {
    uint gid = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    
    // Each work group processes a row of the alignment matrix
    uint row = get_group_id(0);
    
    if (row >= len1) {
        return;
    }
    
    // Shared memory for current row
    __local int row_scores[256];
    __local int prev_row_scores[256];
    
    // Initialize first row
    if (row == 0) {
        for (uint j = local_id; j < len2; j += group_size) {
            row_scores[j] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Process each row
    for (uint i = 0; i < len1; i++) {
        // Copy current row to previous
        if (local_id < len2) {
            prev_row_scores[local_id] = row_scores[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute new row
        for (uint j = local_id; j < len2; j += group_size) {
            int match_score = (seq1[i] == seq2[j]) ? MATCH_SCORE : MISMATCH_PENALTY;
            
            int diag = (i > 0 && j > 0) ? prev_row_scores[j - 1] : 0;
            int left = (j > 0) ? row_scores[j - 1] : 0;
            int up = (i > 0) ? prev_row_scores[j] : 0;
            
            int score1 = diag + match_score;
            int score2 = left + GAP_PENALTY;
            int score3 = up + GAP_PENALTY;
            int score4 = 0;
            
            row_scores[j] = max(max(max(score1, score2), score3), score4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Find maximum score in final row
    int local_max = 0;
    for (uint j = local_id; j < len2; j += group_size) {
        local_max = max(local_max, row_scores[j]);
    }
    
    // Reduction within work group
    __local int local_maxima[256];
    local_maxima[local_id] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_maxima[local_id] = max(local_maxima[local_id], local_maxima[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Update global result
    if (local_id == 0) {
        atomic_max(result, local_maxima[0]);
    }
} 