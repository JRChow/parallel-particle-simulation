#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

///////////////////////////////////// Static Global Variables /////////////////////////////////////

// Length of a bin's side
#define BIN_SIZE 0.01

int Num_Proc_Per_Side;  // Number of processors per side
double Proc_Size;  // Size of a processor's side
int Num_Bins_Per_Proc_Side;  // Number of bins on a processor's side

int My_Row_Idx;  // Row index of this processor
int My_Col_Idx;  // Column index of this processor
double My_Min_X;  // Minimum X value of this processor
double My_Min_Y;  // Minimum Y value of this processor

vector<particle_t*> *Bins;  // The bins belonging to this processor

//////////////////////////////////////// Helper Functions ////////////////////////////////////////

// Calculate the rank that the input particle belongs to
inline int calculate_particle_rank(const particle_t& pt) {
    int row_idx = floor(pt.x / Proc_Size);
    int col_idx = floor(pt.y / Proc_Size);
    // Processors are assigned in row-major
    return row_idx * Num_Proc_Per_Side + col_idx;
}

// Insert the input particle to the correct bin
inline void put_particle_to_bin(particle_t& pt) {
    int row_idx = floor( (pt.x - My_Min_X) / BIN_SIZE );
    int col_idx = floor( (pt.y - My_Min_Y) / BIN_SIZE );
    int idx = row_idx * Num_Bins_Per_Proc_Side + col_idx;
    Bins[idx].push_back(&pt);
}

///////////////////////////////////////// Key Functions /////////////////////////////////////////

// Initialize data objects that we need
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    // Calculate necessary global constants
    Num_Proc_Per_Side = floor( sqrt(num_proc) );
    Proc_Size = size / Num_Proc_Per_Side;
    Num_Bins_Per_Proc_Side = ceil(Proc_Size / BIN_SIZE);

    // Calculate variables specific to this processor
    My_Row_Idx = rank / Num_Proc_Per_Side;
    My_Col_Idx = rank % Num_Proc_Per_Side;
    My_Min_X = My_Row_Idx * Proc_Size;
    My_Min_Y = My_Col_Idx * Proc_Size;

    // Initialize bins specific to this processor
    Bins = new vector<particle_t*>[Num_Bins_Per_Proc_Side * Num_Bins_Per_Proc_Side];
    // Assign particles belonging to this processor to their corresponding bins
    for (int i = 0; i < num_parts; i++) {
        particle_t& pt = parts[i];
        // If particle belongs to this processor
        if (calculate_particle_rank(pt) == rank) {
            put_particle_to_bin(pt);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    // Apply_Force()

    // MPI_Isend(&Bins[0], );

    // Move()
    // TODO: implement
    unordered_map<int, vector<particle_t*>> map;
    for (int i = 0; i < num_parts; ++i) {
        p = parts[i];
    
        int oldRow = floor( (p.x - My_Min_X) / BIN_SIZE );
        int oldCol = floor( (p.y - My_Min_Y) / BIN_SIZE );
        int oldRank = calculate_particle_rank(p);
        int oldIdx = oldRow * Num_Bins_Per_Proc_Side + oldCol;

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

        int newRank = calculate_particle_rank(p);

        if (newRank != oldRank){
            Bins[oldIdx].erase(&p);

            map[newRank].push_back(p);

        } else if (newIdx != oldIdx){
            int newRow = floor( (p.x - My_Min_X) / BIN_SIZE );
            int newCol = floor( (p.y - My_Min_Y) / BIN_SIZE );
            int newIdx = newRow * Num_Bins_Per_Proc_Side + newCol;
            Bins[oldIdx].erase(&p);
            Bins[newIdx].insert(&p);
        }
    }

    vector<particle_t*> go(10);
    for (int i=0; i<num_proc; ++i){
        search = map.find(i);
        if (search != map.end()){
            go = map.at(i);
        MPI_Isend(go, go.size(), PARTICLE, i, 0, MPI_COMM_WORLD);
    }


    vector<particle_t*> come(10);
    for (int i = 0; i<num_proc; ++i){
        MPI_Status status;
        MPI_Irecv(come, come.size(), PARTICLE, i, 0, MPI_COMM_WORLD, &status);
        for (int j=0; j<come.size(); ++j){
            p = come[j];
            int newRow = floor( (p.x - My_Min_X) / BIN_SIZE );
            int newCol = floor( (p.y - My_Min_Y) / BIN_SIZE );
            int newIdx = newRow * Num_Bins_Per_Proc_Side + newCol; 
            Bins[newIdx].insert(&p);
        }
    }

}



/* Write this function such that at the end of it, the master (rank == 0)
 * processor has an in-order view of all particles. That is, the array
 * parts is complete and sorted by particle id. */
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    // TODO: implement
//    cout << "gather_for_save() at " << rank << "/" << num_proc << endl;
}
