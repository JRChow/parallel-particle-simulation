#include "common.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cmath>

using namespace std;

////////////////////////////////////////// Global Variables //////////////////////////////////////////

// Length of a bin's side
#define BIN_SIZE 0.01

// Number of bins per side
int BinCnt;
// Matrix of sets containing particles
unordered_set<particle_t *> *Bins;  // <- Need to protect it against race conditions
// Each bin has a lock
omp_lock_t *Locks;

////////////////////////////////////////// Function Declarations //////////////////////////////////////////

void init_simulation(particle_t *parts, int num_parts, double size);

void simulate_one_step(particle_t *parts, int num_parts, double size);

inline void calculate_bin_forces(int row, int col);

inline void interact_with_neighbor(particle_t *pt, int neiRow, int neiCol);

inline void apply_force(particle_t &particle, particle_t &neighbor);

void move(particle_t &p, double size);

////////////////////////////////////////// Key Functions //////////////////////////////////////////

// You can use this space to initialize static, global data objects
// that you may need. This function will be called once before the
// algorithm begins. Do not do any particle simulation here
void init_simulation(particle_t *parts, int num_parts, double size) {
    // Calculate number of bins and initialize bins
    BinCnt = ceil(size / BIN_SIZE);
    int N = BinCnt * BinCnt;  // Total number of bins
    Bins = new unordered_set<particle_t *>[N];

    // Initialize locks
    Locks = new omp_lock_t[N];
    for (int i = 0; i < N; i++) {
        omp_init_lock(&Locks[i]);
    }

    // Fill in particles into corresponding bins
#pragma omp parallel for  // Parallelize because initialization is also timed
    for (int i = 0; i < num_parts; i++) {
        particle_t &pt = parts[i];
        int row = floor(pt.x / BIN_SIZE);
        int col = floor(pt.y / BIN_SIZE);
        int idx = row * BinCnt + col;

        omp_set_lock(&Locks[idx]);
        Bins[idx].insert(&pt);
        omp_unset_lock(&Locks[idx]);
    }
}

// Note: Outside this function is a "#pragma omp parallel"
void simulate_one_step(particle_t *parts, int num_parts, double size) {
    // Compute forces in each bin
#pragma omp for collapse(2)  // Bin-level parallelism (likely necessary because it's the top-level)
    for (int i = 0; i < BinCnt; i++) {
        for (int j = 0; j < BinCnt; j++) {
            calculate_bin_forces(i, j);  // Each iteration is independent of previous ones
        }
    }

    // Move Particles
#pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}

////////////////////////////////////////// Helper Functions //////////////////////////////////////////

// Helper function to calculate 9-by-9 bins
inline void calculate_bin_forces(int row, int col) {
    auto &bin = Bins[row * BinCnt + col];
    // For each particle in the input bin
    for (auto &pt : Bins[row * BinCnt + col]) {
        pt->ax = pt->ay = 0;
        // Interact with all valid neighboring bins
        interact_with_neighbor(pt, row, col);          // Self
        interact_with_neighbor(pt, row - 1, col);      // Top
        interact_with_neighbor(pt, row + 1, col);      // Bottom
        interact_with_neighbor(pt, row, col - 1);      // Left
        interact_with_neighbor(pt, row, col + 1);      // Right
        interact_with_neighbor(pt, row - 1, col - 1);  // Top left
        interact_with_neighbor(pt, row - 1, col + 1);  // Top right
        interact_with_neighbor(pt, row + 1, col - 1);  // Bottom left
        interact_with_neighbor(pt, row + 1, col + 1);  // Bottom right
    }
}

// For a particle, make it interact with all particles in a neighboring bin.
inline void interact_with_neighbor(particle_t *pt, int neiRow, int neiCol) {
    // Check if the neighbor is valid (within bound)
    if (neiRow < 0 || neiRow >= BinCnt ||
        neiCol < 0 || neiCol >= BinCnt)
        return;
    // Interact with all particles in a valid neighbor
    // Parallelization is probably not helpful here
    for (auto &neiPts : Bins[neiRow * BinCnt + neiCol]) {
        apply_force(*pt, *neiPts);
    }
}

// Apply the force from neighbor to particle
inline void apply_force(particle_t &particle, particle_t &neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t &p, double size) {
    // Old bin of the particle before moving
    int oldRow = floor(p.x / BIN_SIZE);
    int oldCol = floor(p.y / BIN_SIZE);
    int oldIdx = oldRow * BinCnt + oldCol;

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

    // Move the particle to a new bin if necessary
    int newRow = floor(p.x / BIN_SIZE);
    int newCol = floor(p.y / BIN_SIZE);
    int newIdx = newRow * BinCnt + newCol;
    if (newIdx != oldIdx) {

        omp_set_lock(&Locks[oldIdx]);
        Bins[oldIdx].erase(&p);
        omp_unset_lock(&Locks[oldIdx]);

        omp_set_lock(&Locks[newIdx]);
        Bins[newIdx].insert(&p);
        omp_unset_lock(&Locks[newIdx]);

    }
}
