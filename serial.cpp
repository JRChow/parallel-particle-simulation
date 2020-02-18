#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_set>
using namespace std;

#define BIN_SIZE 0.01
int BinCnt;
unordered_set<particle_t *> *Bins;
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

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    BinCnt = ceil(size/BIN_SIZE);
    Bins = new unordered_set<particle_t *>[BinCnt*BinCnt];
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    //Put particles into Bins
    for (int i = 0; i<num_parts; ++i){
        int rowidx = int(parts[i].x/BIN_SIZE);
        int colidx = int(parts[i].y/BIN_SIZE);
        parts[i].ax=parts[i].ay=0;
        Bins[rowidx*BinCnt+colidx].insert(&parts[i]);
    }

    for (int a = 0; a<BinCnt; ++a){
        for (int b = 0; b<BinCnt; ++b){
            for (auto& pt: Bins[a*BinCnt+b]){
                for (auto const &dir : dirs){
                    int row = a+dir[0];
                    int col = b+dir[1];
                    if (row<0 || col<0 || row>= BinCnt || col>=BinCnt)
                        continue; 
                    for (auto& neipts : Bins[row*BinCnt+col]) {
                        apply_force(*pt, *neipts);
                    }
                }
            }
        }
    }


    // for (int i=0; i<num_parts; ++i){
    //     parts[i].ax = parts[i].ay=0;
    //     for (int j=0; j<num_parts; ++j){
    //         apply_force(parts[i], parts[j]);
    //     }
    // }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }

    // Clear Particles in Bins
    for (int a = 0; a<BinCnt; ++a){
        for (int b = 0; b<BinCnt; ++b){
            Bins[a*BinCnt+b].clear();
        }
    }
}
