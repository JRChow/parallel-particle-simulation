#include "common.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

#define BIN_CNT 1  // TODO: tune

// Global 2D array of sets containing particles
unordered_set<particle_t *> binMat[BIN_CNT][BIN_CNT];
double binSize;  // Size of one bin
// 8 neighbors and self
int dirs[9][2] = {{-1, -1},
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
void add_particle_to_right_bin(particle_t pt) {
    int row = int(pt.x / binSize);
    int col = int(pt.y / binSize);
    binMat[row][col].insert(&pt);
}

// Integrate the ODE
// TODO: update points' bins
void move(particle_t &p, double size) {
    // Erase particle from old bin
    int oldRow = int(p.x / binSize);
    int oldCol = int(p.y / binSize);
    binMat[oldRow][oldCol].erase(&p);

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
    // Initialize bin size
    binSize = double(size / BIN_CNT);

    // Fill in particles into corresponding bins
    for (int i = 0; i < num_parts; i++) {
        add_particle_to_right_bin(parts[i]);
    }
}

// Helper function to calculate 9-by-9 bins
void calculate_bin_forces(int row, int col) {
    unordered_set < particle_t * > pts = binMat[row][col];
    // For each particle in the input bin
    for (auto pt : pts) {
        // Iterate over all valid neighboring bins
        for (auto dir : dirs) {
            int neiRow = row + dir[0];
            int neiCol = col + dir[1];
            if (min(neiRow, neiCol) >= 0 &&
                max(neiRow, neiCol) < BIN_CNT) {
                // Iterate over all particles in a neighbor
                for (auto neiPts : binMat[neiRow][neiCol]) {
                    apply_force(*pt, *neiPts);
                }
            }
        }
    }
}

void simulate_one_step(particle_t *parts, int num_parts, double size) {
    // Compute forces
    for (int i = 0; i < BIN_CNT; i++) {
        for (int j = 0; j < BIN_CNT; j++) {
            calculate_bin_forces(i, j);
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}

