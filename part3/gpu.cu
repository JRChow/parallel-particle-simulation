#include "common.h"
#include <cuda.h>
#include <iostream>
#include <cmath>

using namespace std;

//////////////////////////////////////////////////// Useful Macros ////////////////////////////////////////////////////

#define MAX_PTS_PER_BIN 16   // Max capacity of each bin
#define NUM_THREADS_PER_BLK 256  // Number of threads per block
#define BIN_SIZE 0.02  // Length of bin side

// Error checking wrapper macro
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// Indexing bin to retrieve particle
#define IDX(bin_idx, pt_idx) bin_idx * MAX_PTS_PER_BIN + pt_idx

/////////////////////////////////////////////////// Global Variables ///////////////////////////////////////////////////

int Num_Bins_Per_Side;  // Number of bins per side
int Total_Num_Bins;  // Total number of bins
int Num_Blocks_By_Pt;  // Number of blocks (for particle iteration)
int Num_Blocks_By_Bin;  // Number of blocks (for bin iteration)

int* Bins;  // Bins containing particle indices
int* Bin_Sizes;  // Actual size of each bin

/////////////////////////////////////////////////// Helper Functions ///////////////////////////////////////////////////

// Apply force on the particle based on neighbor's position
__device__ void apply_force(particle_t* particle, particle_t* neighbor) {
    // Calculate Distance
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

// For a particle, make it interact with all particles in a neighboring bin.
__device__ void interact_with_neighbor(particle_t* parts, int self_pt_idx,
                                       int* Bins, int* Bin_Sizes, int Num_Bins_Per_Side,
                                       int nei_bin_row, int nei_bin_col) {
    // Check if the neighbor is valid (within bound)
    if (nei_bin_row < 0 || nei_bin_row >= Num_Bins_Per_Side ||
        nei_bin_col < 0 || nei_bin_col >= Num_Bins_Per_Side)
        return;

    // Interact with all particles in the neighbor bin
    int bin_idx = nei_bin_row * Num_Bins_Per_Side + nei_bin_col;
    int num_nei_pts = Bin_Sizes[bin_idx];
    for (int i = 0; i < num_nei_pts; ++i) {
        int nei_pt_idx = Bins[IDX(bin_idx, i)];
        apply_force(&parts[self_pt_idx], &parts[nei_pt_idx]);
    }
}

// Assert-style handler
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/////////////////////////////////////////////////////// Kernels ///////////////////////////////////////////////////////

// Set all bin sizes to 0
__global__ void reset_bin_sizes(int* Bin_Sizes, int Total_Num_Bins) {
    // Calculate thread/bin index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Total_Num_Bins)
        return;

    Bin_Sizes[idx] = 0;  // "Clear" bin
}

// Associate each particle with its corresponding bin
__global__ void rebinning(particle_t* parts, int num_parts,
                          int Num_Bins_Per_Side, int* Bins, int* Bin_Sizes) {
    // Calculate thread/particle index
    int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pt_idx >= num_parts)
        return;

//    // Exhaust all acceleration to prepare for force calculation
//    parts[pt_idx].ax = parts[pt_idx].ay = 0;

    // Determine which bin to put the particle
    particle_t& pt = parts[pt_idx];
    const int bin_row = floor(pt.x / BIN_SIZE);
    const int bin_col = floor(pt.y / BIN_SIZE);
    const int bin_idx = bin_row * Num_Bins_Per_Side + bin_col;

    // Increment bin size atomically
    int old_bin_size = atomicAdd(&Bin_Sizes[bin_idx], 1);
    // Store particle index in bin
    Bins[IDX(bin_idx, old_bin_size)] = pt_idx;
}

// Calculate forces bin-by-bin
__global__ void compute_forces_gpu(particle_t* parts, int* Bins, int* Bin_Sizes, int Num_Bins_Per_Side) {
    // Get thread/bin index
    int bin_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bin_idx >= Num_Bins_Per_Side * Num_Bins_Per_Side)
        return;

    // Calculate row index & column index of this bin
    int row = bin_idx / Num_Bins_Per_Side;
    int col = bin_idx % Num_Bins_Per_Side;

    // For each particle in this bin
    int my_pts_cnt = Bin_Sizes[bin_idx];
    for (int i = 0; i < my_pts_cnt; ++i) {
        int pt_idx = Bins[IDX(bin_idx, i)];
        // Exhaust all acceleration of this particle
        parts[pt_idx].ax = parts[pt_idx].ay = 0;
        // Interact with all 9 valid neighbor bins
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row    , col    );  // Self
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row - 1, col    );  // Top
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row + 1, col    );  // Bottom
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row    , col - 1);  // Left
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row    , col + 1);  // Right
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row - 1, col - 1);  // Top left
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row - 1, col + 1);  // Top right
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row + 1, col - 1);  // Bottom left
        interact_with_neighbor(parts, pt_idx, Bins, Bin_Sizes, Num_Bins_Per_Side, row + 1, col + 1);  // Bottom right
    }
}

// Move each particle
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    // Get thread (particle) ID
    int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pt_idx >= num_parts)
        return;

    // Get particle reference
    particle_t& p = particles[pt_idx];

    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

//////////////////////////////////////////////////// Key Functions ////////////////////////////////////////////////////

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Calculate number of bins
    Num_Bins_Per_Side = ceil(size / BIN_SIZE);
    Total_Num_Bins = Num_Bins_Per_Side * Num_Bins_Per_Side;
    // Calculate number of blocks by particle and by bin (ceiling division)
    Num_Blocks_By_Pt = (num_parts + NUM_THREADS_PER_BLK - 1) / NUM_THREADS_PER_BLK;
    Num_Blocks_By_Bin = (Total_Num_Bins + NUM_THREADS_PER_BLK - 1) / NUM_THREADS_PER_BLK;

    // Allocate memory to bins
    gpuErrorCheck( cudaMalloc(&Bins, Total_Num_Bins * MAX_PTS_PER_BIN * sizeof(int)) );
    gpuErrorCheck( cudaMalloc(&Bin_Sizes, Total_Num_Bins * sizeof(int)) );
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Clearing bins (each thread handles a bin)
    reset_bin_sizes<<<Num_Blocks_By_Bin, NUM_THREADS_PER_BLK>>>(Bin_Sizes, Total_Num_Bins);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Assigning particles to bins (each thread handles a particle)
    rebinning<<<Num_Blocks_By_Pt, NUM_THREADS_PER_BLK>>>(parts, num_parts,
                                                         Num_Bins_Per_Side, Bins, Bin_Sizes);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Compute interaction forces (each thread handles a bin)
    compute_forces_gpu<<<Num_Blocks_By_Bin, NUM_THREADS_PER_BLK>>>(parts, Bins, Bin_Sizes, Num_Bins_Per_Side);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Move particles (each thread handles a particle)
    move_gpu<<<Num_Blocks_By_Pt, NUM_THREADS_PER_BLK>>>(parts, num_parts, size);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );
}