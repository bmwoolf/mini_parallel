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
Alignment: compare to average reference genome
Complementary: find what % of genome is not perfectly complementary (boooo)