#include "common.h"
#include <cuda.h>

using namespace std;

#define MAX_PER_BIN 16
#define NUM_THREADS 256

int d;
__device__ int* bins;  // indexes of particles instead
__device__ int* bin_sizes;
__device__ int* neighbor_bins;
__device__ int* neighbor_bin_sizes;

////////////////////////////////////////// Global Variables //////////////////////////////////////////

int part_blks;  // Number of blocks
int bin_blks;


__device__ void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* parts, int d) {
    // Get thread (particle) ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d * d)
        return;
    
    for (int p = 0; p < neighbor_bin_sizes[idx]; p++) {
        const int neighbor_idx = neighbor_bins[9 * idx + p];
        for (int m = 0; m < bin_sizes[idx]; m++) {
            for (int q = 0; q < bin_sizes[neighbor_idx]; q++) {
                apply_force(parts[bins[idx * MAX_PER_BIN + m]], parts[bins[neighbor_idx * MAX_PER_BIN + q]]);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

__global__ void get_neighbor_bin_indexes(int d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d * d)
        return;
    
    int* neighbors = neighbor_bins + 9 * idx;
    int* neighbor_sizes = neighbor_bin_sizes + idx;

    neighbors[0] = idx;
    *neighbor_sizes = 1;

    if (idx % d != 0) {
        if (idx > d) {
            neighbors[*neighbor_sizes] = idx - d - 1;
            (*neighbor_sizes)++;
        }
        neighbors[*neighbor_sizes] = idx - 1;
        (*neighbor_sizes)++;

        if (idx <= d * (d-1)) {
            neighbors[*neighbor_sizes] = idx + d - 1;
            (*neighbor_sizes)++;
        }
    }

    if (idx % d != d-1) {
        if (idx >= d - 1) {
            neighbors[*neighbor_sizes] = idx - d + 1;
            (*neighbor_sizes)++;
        }
        neighbors[*neighbor_sizes] = idx + 1;
        (*neighbor_sizes)++;

        if (idx < d * (d-1) - 1) {
            neighbors[*neighbor_sizes] = idx + d + 1;
            (*neighbor_sizes)++;
        }
    }

    if (idx >= d) {
        neighbors[*neighbor_sizes] = idx - d;
        (*neighbor_sizes)++;
    }

    if (idx < d * (d-1)) {
        neighbors[*neighbor_sizes] = idx + d;
        (*neighbor_sizes)++;
    }
}

__global__ void reset_bin_sizes(int d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d * d)
        return;

    bin_sizes[idx] = 0;
}

__global__ void rebinning(particle_t* parts, double size, int num_parts, int d) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_parts)
        return;

    const int X = floor(parts[i].x * d / size);
    const int Y = floor(parts[i].y * d / size);
    const int idx = X + d * Y;
    parts[i].ax = parts[i].ay = 0;

    int id = atomicAdd(bin_sizes + idx, 1);
    bins[idx * MAX_PER_BIN + id] = i;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    part_blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    bin_blks = (d * d + NUM_THREADS - 1) / NUM_THREADS;

    d = floor(size / cutoff);

    // cudaMallocManaged(&bins, d * d * MAX_PER_BIN * sizeof(int));
    // cudaMallocManaged(&bin_sizes, d * d * sizeof(int));
    // cudaMallocManaged(&neighbor_bins, 9 * d * d * sizeof(int));
    // cudaMallocManaged(&neighbor_bin_sizes, d * d * sizeof(int));

    cudaMalloc(&bins, d * d * MAX_PER_BIN * sizeof(int));
    cudaMalloc(&bin_sizes, d * d * sizeof(int));
    cudaMalloc(&neighbor_bins, 9 * d * d * sizeof(int));
    cudaMalloc(&neighbor_bin_sizes, d * d * sizeof(int));
    
    get_neighbor_bin_indexes<<<bin_blks, NUM_THREADS>>>(d);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    reset_bin_sizes<<<bin_blks, NUM_THREADS>>>(d);
    cudaDeviceSynchronize();

    rebinning<<<part_blks, NUM_THREADS>>>(parts, size, num_parts, d);
    cudaDeviceSynchronize();

    // Compute forces
    compute_forces_gpu<<<bin_blks, NUM_THREADS>>>(parts, d);
    cudaDeviceSynchronize();

    // Move particles
    move_gpu<<<part_blks, NUM_THREADS>>>(parts, num_parts, size);
}
