#include "common.h"
#include <cmath>
#include <vector>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
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
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dtt;
    p.vy += p.ay * dtt;
    p.x += p.vx * dtt;
    p.y += p.vy * dtt;

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


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    //Build Bins
    int bincnt = 200;
    double binsize = double(size / bincnt);
    std::vector<particle_t> bins[bincnt][bincnt];

    for (int i = 0; i < num_parts; ++i) {
        int j = int(parts[i].x / binsize);
        int k = int(parts[i].y / binsize);
        bins[j][k].push_back(parts[i]);
    }

    // Add neighbor bins loops:
    for (int a = 0; a < bincnt; ++a) {
        for (int b = 0; b < bincnt; ++b) {
            for (int i = 0; i < bins[a][b].size(); ++i) {
                bins[a][b][i].ax = bins[a][b][i].ay = 0;
                int dirs[9][2] = {{-1, -1},
                                  {-1, 0},
                                  {-1, 1},
                                  {0,  -1},
                                  {0,  0},
                                  {0,  1},
                                  {1,  -1},
                                  {1,  0},
                                  {1,  1}};
                for (int d = 0; d < 9; d++) {
                    int row = a + dirs[d][0];
                    int col = b + dirs[d][1];
                    if (row >= 0 && row < bincnt && col >= 0 && col < bincnt) {
                        for (int j = 0; j < bins[row][col].size(); ++j) {
                            apply_force(bins[a][b][i], bins[a][b][j]);
                        }
                    }
                }
            }
        }
    }
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
    /**

    // Compute Forces
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        for (int j = 0; j < num_parts; ++j) {
            apply_force(parts[i], parts[j]);
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
     **/
}

