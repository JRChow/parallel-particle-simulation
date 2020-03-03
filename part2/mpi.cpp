#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>

using namespace std;

///////////////////////////////////// Static Global Variables /////////////////////////////////////

// Length of a bin's side
#define BIN_SIZE 0.01
// Maximum number of particles per bin
#define MAX_NUM_PT_PER_BIN 10

int Num_Proc_Per_Side;  // Number of processors per side
double Proc_Size;  // Size of a processor's side
int Num_Bins_Per_Proc_Side;  // Number of bins on a processor's side

int My_Row_Idx;  // Row index of this processor
int My_Col_Idx;  // Column index of this processor
double My_Min_X;  // Minimum X value of this processor
double My_Min_Y;  // Minimum Y value of this processor

// The bins containing all the relevant particles
unordered_set<particle_t*> *Bins;

enum Neighbor {TOP, BOTTOM, LEFT, RIGHT,  // Four directions
        TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT};  // Corners

// Buffers for receiving particles
vector<particle_t*> recv_buffers[8];

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
    int row_idx = int( (pt.x - My_Min_X) / BIN_SIZE ) + 1;
    int col_idx = int( (pt.y - My_Min_Y) / BIN_SIZE ) + 1;
    int idx = row_idx * Num_Bins_Per_Proc_Side + col_idx;
    Bins[idx].insert(&pt);
}

// Get the rank of a neighbor processor
inline int get_neighbor_proc_rank(int my_rank, Neighbor neighbor_enum) {
    int neighbor_rank;
    switch (neighbor_enum) {
        case TOP_LEFT:
            neighbor_rank = my_rank - Num_Proc_Per_Side - 1;
            break;
        case TOP:
            neighbor_rank = my_rank - Num_Proc_Per_Side;
            break;
        case TOP_RIGHT:  // Top-right neighbor
            neighbor_rank = my_rank - Num_Proc_Per_Side + 1;
            break;
        case LEFT:  // Left neighbor
            neighbor_rank = my_rank - 1;
            break;
        case RIGHT:  // Right neighbor
            neighbor_rank = my_rank + 1;
            break;
        case BOTTOM_LEFT:  // Bottom-left neighbor
            neighbor_rank = my_rank + Num_Proc_Per_Side - 1;
            break;
        case BOTTOM:  // Bottom neighbor
            neighbor_rank = my_rank + Num_Proc_Per_Side;
            break;
        case BOTTOM_RIGHT:  // Bottom-right neighbor
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

    // Initialize bins specific to this processor (+2 because of padding)
    int total_num_bins = (Num_Bins_Per_Proc_Side + 2) * (Num_Bins_Per_Proc_Side + 2);
    Bins = new unordered_set<particle_t*>[total_num_bins];
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

//    MPI_Status stats[8];
//    // Communicate with neighbor processors
//    // Top
//    int top_rank = get_neighbor_proc_rank(rank, 2);
//    if (top_rank != -1) {
//
//    }
//    // Bottom
//    int bottom_rank = get_neighbor_proc_rank(rank, 8);
//    // Left
//    int left_rank = get_neighbor_proc_rank(rank, 4);
//    // Right
//    int right_rank = get_neighbor_proc_rank(rank, 6);
//    // Corner bins
//    // Top-left
//    int top_left_rank = get_neighbor_proc_rank(rank, TOP_LEFT);
//    if (top_left_rank != -1) {
//        unordered_set<particle_t *>& top_left_bin = Bins[0];
//        vector<particle_t*> top_left_pts(top_left_bin.begin(), top_left_bin.end());
//        recv_buffers[TOP_LEFT].resize(MAX_NUM_PT_PER_BIN);
//        MPI_Status status;
//        MPI_Sendrecv(&top_left_pts[0], top_left_pts.size(), PARTICLE,
//                     top_left_rank, MPI_ANY_TAG,
//                     &recv_buffers[TOP_LEFT][0], MAX_NUM_PT_PER_BIN, PARTICLE,
//                     top_left_rank, MPI_ANY_TAG,
//                     MPI_COMM_WORLD, &status);
//    }
//    // Top-right
//    int top_right_rank = get_neighbor_proc_rank(rank, 3);
//    // Bottom-left
//    int bottom_left_rank = get_neighbor_proc_rank(rank, 7);
//    // Bottom-right
//    int bottom_right_rank = get_neighbor_proc_rank(rank, 9);

//    // Apply_Force()
//    MPI_Request request, request2;
//    MPI_Status status;
//    int top_left_rank = get_neighbor_proc_rank(rank, 1);
//    if (top_left_rank != -1) {
//
////        cout << "sending " << Bins[0].size() << " pts to " << top_left_rank << endl;
//    }
////
//    std::vector<uint32_t> recv_buffer;
//    recv_buffer.resize(20);
//    int bottom_right_rank = get_neighbor_proc_rank(rank, 9);
//    if (bottom_right_rank != -1) {
//        MPI_Irecv(&recv_buffer[0], 20, PARTICLE, bottom_right_rank,
//                0, MPI_COMM_WORLD, &request2);
////        cout << "receiving from " << bottom_right_rank << endl;
//    }
//
//    if (top_left_rank != -1) {
//        MPI_Wait(&request, &status);
//    }
//
//    if (bottom_right_rank != -1) {
//        MPI_Wait(&request2, &status);
//    }

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
