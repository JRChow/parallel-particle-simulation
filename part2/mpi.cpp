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

/* Assuming this processor is "5" on a number pad, get the rank of the neighbor corresponding
 * to the input neighbor_num on the number pad. The number pad looks like:
 * 1 2 3
 * 4 + 6
 * 7 8 9
 * For example, neighbor_num 1 means this processor's top-left neighbor.
 * */
inline int get_neighbor_proc_rank(int my_rank, int neighbor_num) {
    int neighbor_rank = -1;
    switch (neighbor_num) {
        case 1:  // Top-left neighbor
            neighbor_rank = my_rank - Num_Proc_Per_Side - 1;
            break;
        case 2:  // Top neighbor
            neighbor_rank = my_rank - Num_Proc_Per_Side;
            break;
        case 3:  // Top-right neighbor
            break;
        case 4:  // Left neighbor
            neighbor_rank = my_rank - 1;
            break;
        case 6:  // Right neighbor
            neighbor_rank = my_rank + 1;
            break;
        case 7:  // Bottom-left neighbor
            neighbor_rank = my_rank + Num_Proc_Per_Side - 1;
            break;
        case 8:  // Bottom neighbor
            neighbor_rank = my_rank + Num_Proc_Per_Side;
            break;
        case 9:  // Bottom-right neighbor
            neighbor_rank = my_rank + Num_Proc_Per_Side + 1;
            break;
        default:
            neighbor_rank = -1;
    }
    if (neighbor_rank < 0 ||
        neighbor_rank >= Num_Proc_Per_Side * Num_Proc_Per_Side)
        return -1;
    return neighbor_rank;
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
    MPI_Isend(&Bins[0][0], Bins[0].size(), PARTICLE, );

    // Move()
    // TODO: implement
}

/* Write this function such that at the end of it, the master (rank == 0)
 * processor has an in-order view of all particles. That is, the array
 * parts is complete and sorted by particle id. */
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    // TODO: implement
//    cout << "gather_for_save() at " << rank << "/" << num_proc << endl;
}
