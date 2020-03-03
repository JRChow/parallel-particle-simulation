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

enum Direction {TOP, BOTTOM, LEFT, RIGHT,  // Horizontal and vertical
        TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT};  // Corners

// Buffers for receiving particles from 8 neighbors
vector<particle_t*> Recv_Buffers[8];

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
inline int get_neighbor_proc_rank(Direction nei_dir) {
    int nei_row_idx = My_Row_Idx;
    int nei_col_idx = My_Col_Idx;

    switch (nei_dir) {
        case TOP_LEFT:
            nei_row_idx -= 1;
            nei_col_idx -= 1;
            break;
        case TOP:
            nei_row_idx -= 1;
            break;
        case TOP_RIGHT:
            nei_row_idx -= 1;
            nei_col_idx += 1;
            break;
        case LEFT:
            nei_col_idx -= 1;
            break;
        case RIGHT:
            nei_col_idx += 1;
            break;
        case BOTTOM_LEFT:
            nei_row_idx += 1;
            nei_col_idx -= 1;
            break;
        case BOTTOM:
            nei_row_idx += 1;
            break;
        case BOTTOM_RIGHT:
            nei_row_idx += 1;
            nei_col_idx += 1;
            break;
        default:
            return -1;
    }

    if (nei_row_idx < 0 || nei_row_idx >= Num_Proc_Per_Side ||
        nei_col_idx < 0 || nei_col_idx >= Num_Proc_Per_Side)
        return -1;
    return nei_row_idx * Num_Proc_Per_Side + nei_col_idx;;
}

// Collect particles from certain bins and put them in a vector
void collect_particles_from_bins(Direction direction, vector<particle_t*>& pt_vec) {
    int N_padded = Num_Bins_Per_Proc_Side + 2;
    if (direction == TOP || direction == BOTTOM) {
        int row_idx = direction == TOP ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            unordered_set<particle_t*>& bin = Bins[row_idx * N_padded + i];
            pt_vec.insert(pt_vec.end(), bin.begin(), bin.end());
        }
    } else if (direction == LEFT || direction == RIGHT) {
        int col_idx = direction == LEFT ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            unordered_set<particle_t*>& bin = Bins[i * N_padded + col_idx];
            pt_vec.insert(pt_vec.end(), bin.begin(), bin.end());
        }
    }
}

// Communicate with horizontal and vertical neighbors | TODO: add assertion check
void communicate_with_non_diagonal_neighbors(Direction nei_dir) {
    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If neighbor exists
        // Collect to-be-sent particles from their bins
        vector<particle_t*> pt_vec(Num_Bins_Per_Proc_Side);
        collect_particles_from_bins(nei_dir, pt_vec);
        // Get the receiving buffer
        vector<particle_t*>& buffer = Recv_Buffers[nei_dir];
        // Send and receive
        MPI_Status status;
        MPI_Sendrecv(&pt_vec[0], pt_vec.size(), PARTICLE,
                     nei_rank, 1234,
                     &buffer[0], MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side,
                     PARTICLE, nei_rank, 1234,
                     MPI_COMM_WORLD, &status);
    }
}

// Communicate with diagonal processors | TODO: add assertion check
void communicate_with_diagonal_neighbors(Direction nei_dir) {
    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If the neighbor exists
        // Copy all the points in the bin to a vector
        unordered_set<particle_t *>& my_bin = Bins[0];
        vector<particle_t*> my_pts(my_bin.begin(), my_bin.end());
        // Get the receiving buffer
        vector<particle_t*>& buffer = Recv_Buffers[nei_dir];
        MPI_Status status;
        // Send and receive
        MPI_Sendrecv(&my_pts[0], my_pts.size(), PARTICLE,
                     nei_rank, 5678,
                     &buffer[0], MAX_NUM_PT_PER_BIN, PARTICLE,
                     nei_rank, 5678,
                     MPI_COMM_WORLD, &status);
    }
}

// Is the rank used in actual computation?
bool is_useful_rank(int rank) {
    return rank < Num_Proc_Per_Side * Num_Proc_Per_Side;
}

///////////////////////////////////////// Key Functions /////////////////////////////////////////

// Initialize data objects that we need
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    // Calculate necessary global constants
    Num_Proc_Per_Side = floor( sqrt(num_proc) );
    if (!is_useful_rank(rank)) return;
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

    // Allocate memory to receiving buffers
    Recv_Buffers[TOP].reserve(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[BOTTOM].reserve(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[LEFT].reserve(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[RIGHT].reserve(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[TOP_LEFT].reserve(MAX_NUM_PT_PER_BIN);
    Recv_Buffers[TOP_RIGHT].reserve(MAX_NUM_PT_PER_BIN);
    Recv_Buffers[BOTTOM_LEFT].reserve(MAX_NUM_PT_PER_BIN);
    Recv_Buffers[BOTTOM_RIGHT].reserve(MAX_NUM_PT_PER_BIN);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_proc) {
    if (is_useful_rank(rank)) {
        // Communicate with horizontal and vertical neighbor processors
        communicate_with_non_diagonal_neighbors(TOP);
        communicate_with_non_diagonal_neighbors(BOTTOM);
        communicate_with_non_diagonal_neighbors(LEFT);
        communicate_with_non_diagonal_neighbors(RIGHT);
        // Communicate with diagonal processors
        communicate_with_diagonal_neighbors(TOP_LEFT);
        communicate_with_diagonal_neighbors(TOP_RIGHT);
        communicate_with_diagonal_neighbors(BOTTOM_LEFT);
        communicate_with_diagonal_neighbors(BOTTOM_RIGHT);
    }

    MPI_Barrier(MPI_COMM_WORLD);

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
