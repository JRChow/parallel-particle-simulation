#include "common.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cmath>

using namespace std;

// Length of a bin's side
#define BIN_SIZE 0.02

// Number of bins per side
int binCnt;
// Matrix of sets containing particles
unordered_set<particle_t *> *bins;
// 8 neighbors and self
const int dirs[9][2] = {{-1, -1},
                        {-1, 0},
                        {-1, 1},
                        {0,  -1},
                        {0,  0},
                        {0,  1},
                        {1,  -1},
                        {1,  0},
                        {1,  1}};

// Apply the force from neighbor to particle
void apply_force(particle_t &particle, particle_t &neighbor) {
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

// Add the particle to its new bin (if necessary).
inline void add_particle_to_right_bin(particle_t &pt) {
    int row = floor(pt.x / BIN_SIZE);
    int col = floor(pt.y / BIN_SIZE);
    bins[row * binCnt + col].insert(&pt);
}

// Integrate the ODE
void move(particle_t &p, double size) {
    // Erase particle from old bin
    int oldRow = floor(p.x / BIN_SIZE);
    int oldCol = floor(p.y / BIN_SIZE);
    bins[oldRow * binCnt + oldCol].erase(&p);

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

    // Add particle to new bin
    add_particle_to_right_bin(p);
}

// You can use this space to initialize static, global data objects
// that you may need. This function will be called once before the
// algorithm begins. Do not do any particle simulation here
void init_simulation(particle_t *parts, int num_parts, double size) {
    // Calculate number of bins and initialize bins
    binCnt = ceil(size / BIN_SIZE);
    bins = new unordered_set<particle_t *>[binCnt * binCnt];

    // Fill in particles into corresponding bins
    for (int i = 0; i < num_parts; i++) {
        add_particle_to_right_bin(parts[i]);
    }
}

// Helper function to calculate 9-by-9 bins
void calculate_bin_forces(int row, int col) {
    // For each particle in the input bin
    for (auto &pt : bins[row * binCnt + col]) {
        pt->ax = pt->ay = 0;
        // Iterate over all valid neighboring bins
        for (auto const &dir : dirs) {
            int neiRow = row + dir[0];
            int neiCol = col + dir[1];
            if (neiRow >= 0 && neiCol >= 0 &&
                neiRow < binCnt && neiCol < binCnt) {
                // Iterate over all particles in a neighbor
                for (auto &neiPts : bins[neiRow * binCnt + neiCol]) {
                    apply_force(*pt, *neiPts);
                }
            }
        }
    }
}

void simulate_one_step(particle_t *parts, int num_parts, double size) {
    // Compute forces
    for (int i = 0; i < binCnt; i++) {
        for (int j = 0; j < binCnt; j++) {
            calculate_bin_forces(i, j);
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}

