#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <omp.h>
#include <chrono>
using namespace std;

#define BIN_SIZE 0.05
int BinCnt;
// unordered_set<int> *Bins;
vector<particle_t *> *Bins;
omp_lock_t *Locks;
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
double t1=0;
double t2=0;
double t3=0;
double t4=0;
std::chrono::time_point<std::chrono::steady_clock> start1;
std::chrono::time_point<std::chrono::steady_clock> start2;
std::chrono::time_point<std::chrono::steady_clock> start3;
std::chrono::time_point<std::chrono::steady_clock> start4;

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
    Bins = new vector<particle_t *>[BinCnt*BinCnt];
    Locks = new omp_lock_t[BinCnt*BinCnt];
    // Bins = new unordered_set<int>[BinCnt*BinCnt];

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    //Put particles into Bins
    #pragma omp master
    {
        start1 = std::chrono::steady_clock::now();
    }
    #pragma omp for
    for (int i = 0; i<num_parts; ++i){
        int rowidx = int(parts[i].x/BIN_SIZE);
        int colidx = int(parts[i].y/BIN_SIZE);
        parts[i].ax=parts[i].ay=0;
	omp_set_lock(&Locks[rowidx*BinCnt+colidx]);
        Bins[rowidx*BinCnt+colidx].push_back(&parts[i]);
        omp_unset_lock(&Locks[rowidx*BinCnt+colidx]);
    }

    #pragma omp master
    {
        auto end1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end1-start1;
        t1 += diff.count();
    }

    #pragma omp master
    {
        start2 = std::chrono::steady_clock::now();
    }    
    #pragma omp for collapse(2)
    for (int a = 0; a<BinCnt; ++a){
        for (int b = 0; b<BinCnt; ++b){
            // for (unordered_set<int>::iterator pt = Bins[a*BinCnt+b].begin(); pt!=Bins[a*BinCnt+b].end(); ++pt){
            for (auto &pt : Bins[a*BinCnt+b]){
                for (auto const &dir : dirs){
                    int row = a+dir[0];
                    int col = b+dir[1];
                    if (row<0 || col<0 || row>= BinCnt || col>=BinCnt)
                        continue; 
                    // for (unordered_set<int>::iterator neipts = Bins[row*BinCnt+col].begin(); neipts!=Bins[row*BinCnt+col].end();++neipts) {
                    for (auto &neipts : Bins[row*BinCnt+col]){    
                        apply_force(*pt, *neipts);
                    }
                }
            }
        }
    }

    #pragma omp master
    {
        auto end2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end2-start2;
        t2 += diff.count();
    }

    // Move Particles
    #pragma omp master
    {
        start3 = std::chrono::steady_clock::now();
    }
    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
    #pragma omp master
    {
        auto end3 = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end3-start3;
        t3 += diff.count();
    }

    // Clear Particles in Bins
    #pragma omp master
    {
        start4 = std::chrono::steady_clock::now();
    }
    #pragma omp for
    for (int a = 0; a<BinCnt; ++a){
        for (int b = 0; b<BinCnt; ++b){
            Bins[a*BinCnt+b].clear();
        }
    }

    #pragma omp master
    {
        auto end4 = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end4-start4;
        t4 += diff.count();
    }
}

void callback(){
    cout<<"Insertion Time: "<<t1<<endl;
    cout<<"Apply Force: "<<t2<<endl;
    cout<<"Move Particle: "<<t3<<endl;
    cout<<"Clear Bins: "<<t4<<endl;
}

