#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <unordered_map>

using namespace std;

///////////////////////////////////// Static Global Variables /////////////////////////////////////

// Length of a bin's side
#define BIN_SIZE 0.01
// Maximum number of particles per bin
#define MAX_NUM_PT_PER_BIN 5

int Num_Proc_Per_Side;  // Number of processors per side
int Num_Useful_Proc;  // Total number of processors involved in computation
double Proc_Size;  // Size of a processor's side
int Num_Bins_Per_Proc_Side;  // Number of bins on a processor's side

int My_Row_Idx;  // Row index of this processor
int My_Col_Idx;  // Column index of this processor
double My_Min_X;  // Minimum X value of this processor
double My_Min_Y;  // Minimum Y value of this processor

// The bins containing all the relevant particles
unordered_set<particle_t *> *Bins;

enum Direction {
    TOP, BOTTOM, LEFT, RIGHT,  // Horizontal and vertical
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT  // Corners
};

// Buffers for receiving particles from 8 neighbors
vector<particle_t> Recv_Buffers[8];

// A map between process rank and particles to be send
unordered_map<int, vector<particle_t>> Map;

//////////////////////////////////////// Helper Functions ////////////////////////////////////////

// Calculate the rank that the input particle belongs to
inline int calculate_particle_rank(const particle_t &pt) {
    int row_idx = floor(pt.x / Proc_Size);
    int col_idx = floor(pt.y / Proc_Size);
    // Processors are assigned in row-major
    return row_idx * Num_Proc_Per_Side + col_idx;
}

// Insert the input particle to the correct bin
inline void put_particle_to_bin(particle_t &pt) {
    int row_idx = static_cast <int> (floor((pt.x - My_Min_X) / BIN_SIZE)) + 1;
    int col_idx = static_cast <int> (floor((pt.y - My_Min_Y) / BIN_SIZE)) + 1;
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
    return nei_row_idx * Num_Proc_Per_Side + nei_col_idx;
}

// Add all particles in a bin to a vector
void add_all_pts_from_bin_to_vec_as_struct(const unordered_set<particle_t *> &bin,
                                           vector<particle_t> &vec) {
    for (auto &pt : bin) {
        vec.push_back(*pt);
    }
}

// Collect particles from certain bins and put them in a vector
void collect_pts_from_halo_bins(Direction which_border, vector<particle_t> &pt_vec) {
    int N_padded = Num_Bins_Per_Proc_Side + 2;
    if (which_border == TOP || which_border == BOTTOM) {
        int row_idx = which_border == TOP ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            auto &bin = Bins[row_idx * N_padded + i];
            add_all_pts_from_bin_to_vec_as_struct(bin, pt_vec);
        }
    } else if (which_border == LEFT || which_border == RIGHT) {
        int col_idx = which_border == LEFT ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            auto &bin = Bins[i * N_padded + col_idx];
            add_all_pts_from_bin_to_vec_as_struct(bin, pt_vec);
        }
    }
}

// Collect all particle structs from my inner bins
void collect_pts_from_all_my_bins(vector<particle_t> &dest_vec) {
    int N = Num_Bins_Per_Proc_Side + 2;
    for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
        for (int j = 1; j <= Num_Bins_Per_Proc_Side; ++j) {
            auto &bin = Bins[i * N + j];
            add_all_pts_from_bin_to_vec_as_struct(bin, dest_vec);
        }
    }
}

// Put particles in the receiving buffer into correct bins
void put_buffered_particles_into_bins(Direction which_border, int num_pts) {
    vector<particle_t> &buffer = Recv_Buffers[which_border];
    for (int i = 0; i < num_pts; i++) {
        particle_t &pt = buffer[i];
        put_particle_to_bin(pt);
    }
}

// Communicate with horizontal and vertical neighbors | TODO: add assertion check if not too slow
void communicate_with_non_diagonal_neighbors(Direction nei_dir) {
    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If neighbor exists
        // Collect to-be-sent particles from their bins
        vector<particle_t> pt_vec;
        collect_pts_from_halo_bins(nei_dir, pt_vec);
        // Get the receiving buffer
        vector<particle_t> &buffer = Recv_Buffers[nei_dir];
        // Send and receive
        MPI_Status status;
        MPI_Sendrecv(&pt_vec[0], pt_vec.size(), PARTICLE,
                     nei_rank, 1234,
                     &buffer[0], MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side,
                     PARTICLE, nei_rank, 1234,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Fill received particles into right bins
        put_buffered_particles_into_bins(nei_dir, recv_cnt);
    }
}

// Communicate with diagonal processors | TODO: add assertion check if not too slow
void communicate_with_diagonal_neighbors(Direction nei_dir) {
    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If the neighbor exists
        // Copy all the points in the bin to a vector
        unordered_set<particle_t *> &my_bin = Bins[0];
        vector<particle_t> my_pts;
        add_all_pts_from_bin_to_vec_as_struct(my_bin, my_pts);
        // Get the receiving buffer
        vector<particle_t> &buffer = Recv_Buffers[nei_dir];
        MPI_Status status;
        // Send and receive
        MPI_Sendrecv(&my_pts[0], my_pts.size(), PARTICLE,
                     nei_rank, 5678,
                     &buffer[0], MAX_NUM_PT_PER_BIN, PARTICLE,
                     nei_rank, 5678,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Fill received particles into right bins
        put_buffered_particles_into_bins(nei_dir, recv_cnt);
    }
}

// Is the rank used in actual computation?
inline bool is_useful_rank(int rank) {
    return rank < Num_Useful_Proc;
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

// For a particle, make it interact with all particles in a neighboring bin.
inline void interact_with_neighbor(particle_t *pt, int neiRow, int neiCol) {
    int N = Num_Bins_Per_Proc_Side + 2;
    // Interact with all particles in a valid neighbor
    for (auto &neiPts : Bins[neiRow * N + neiCol]) {
        apply_force(*pt, *neiPts);
    }
}

// Helper function to calculate 9-by-9 bins
inline void calculate_bin_forces(int row, int col) {
    int N = Num_Bins_Per_Proc_Side + 2;
    auto &bin = Bins[row * N + col];
    // For each particle in the input bin
    for (auto &pt : Bins[row * N + col]) {
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

void move(particle_t &p, double size) {
    int oldRow = static_cast <int>(floor((p.x - My_Min_X) / BIN_SIZE)) + 1;
    int oldCol = static_cast <int>(floor((p.y - My_Min_Y) / BIN_SIZE)) + 1;
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

    // Put the particle into new places
    int newRank = calculate_particle_rank(p);

    if (newRank != oldRank) {
        Bins[oldIdx].erase(&p);
        Map[newRank].push_back(p);
    } else {
        int newRow = static_cast <int>(floor((p.x - My_Min_X) / BIN_SIZE)) + 1;
        int newCol = static_cast <int>(floor((p.y - My_Min_Y) / BIN_SIZE)) + 1;
        int newIdx = newRow * Num_Bins_Per_Proc_Side + newCol;

        if (newIdx != oldIdx) {
            Bins[oldIdx].erase(&p);
            Bins[newIdx].insert(&p);
        }
    }
}

void move_particle_cross_processor(int num_proc) {
    for (int i = 0; i < num_proc; ++i) {
        int Num_Bins_Per_Proc = Num_Bins_Per_Proc_Side * Num_Bins_Per_Proc_Side;
        vector<particle_t> go(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc);
        auto search = Map.find(i);
        if (search != Map.end()) {
            go = Map.at(i);
            MPI_Send(&go[0], go.size(), PARTICLE, i, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < num_proc; ++i) {
        int Num_Bins_Per_Proc = Num_Bins_Per_Proc_Side * Num_Bins_Per_Proc_Side;
        vector<particle_t> come(MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc);
        MPI_Status status;
        MPI_Recv(&come[0], come.size(), PARTICLE, i, 0, MPI_COMM_WORLD, &status);
        for (auto &pt : come) {
            put_particle_to_bin(pt);
        }
    }
}

// Write particle structs in the source array to correct slots in the destination array
void write_pts_to_dest_arr(particle_t *src_pts_arr, int N_src, particle_t *dest_pts_arr) {
    for (int i = 0; i < N_src; ++i) {
        particle_t &pt = src_pts_arr[i];
        uint64_t idx = pt.id - 1;
        dest_pts_arr[idx] = pt;
    }
}

///////////////////////////////////////// Key Functions /////////////////////////////////////////

// Initialize data objects that we need
void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    // Calculate necessary global constants
    Num_Proc_Per_Side = floor(sqrt(num_proc));
    Num_Useful_Proc = Num_Proc_Per_Side * Num_Proc_Per_Side;
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
    Bins = new unordered_set<particle_t *>[total_num_bins];
    // Assign particles belonging to this processor to their corresponding bins
    for (int i = 0; i < num_parts; i++) {
        particle_t &pt = parts[i];
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

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
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

        // Apply forces in each bin
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            for (int j = 1; j <= Num_Bins_Per_Proc_Side; ++j) {
                calculate_bin_forces(i, j);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    // Move()
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
    move_particle_cross_processor(num_proc);

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?
}

/* Write this function such that at the end of it, the master (rank == 0)
 * processor has an in-order view of all particles. That is, the array
 * parts is complete and sorted by particle id. */
// TODO: maybe try Isend/Irecv
void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    if (!is_useful_rank(rank)) return;

    // For collecting particles from my bins
    vector<particle_t> my_pts;

    // Collect all the particles in the inner bins
    collect_pts_from_all_my_bins(my_pts);

    if (rank != 0) {  // If not master, send my particles to master
        MPI_Send(&my_pts[0], my_pts.size(), PARTICLE, 0,
                 1234, MPI_COMM_WORLD);
    } else {  // If master
        // Write master particles to correct slots
        write_pts_to_dest_arr(my_pts.data(), my_pts.size(), parts);

        // Create buffer for receiving particles
        int max_num_pts_per_proc = MAX_NUM_PT_PER_BIN * Num_Bins_Per_Proc_Side * Num_Bins_Per_Proc_Side;
        vector<particle_t> buffer(max_num_pts_per_proc);

        // Receive particles from all other useful processors
        for (int r = 1; r < Num_Useful_Proc; ++r) {
            MPI_Status status;
            MPI_Recv(&buffer[0], max_num_pts_per_proc, PARTICLE, r,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int recv_cnt;
            MPI_Get_count(&status, PARTICLE, &recv_cnt);
            // Write received particles to correct slots
            write_pts_to_dest_arr(buffer.data(), recv_cnt, parts);
        }
    }
}
