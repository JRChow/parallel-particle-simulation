#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <cassert>

using namespace std;

///////////////////////////////////// Static Global Variables /////////////////////////////////////

// Length of a bin's side
#define BIN_SIZE 0.01
// Maximum number of particles per bin
#define MAX_NUM_PTS_PER_BIN 20
// Indexing Bins matrix
#define BIN_IDX(row, col) row * NumBinCols + col

// Global variables shared across processors
double ProcHeight;  // Height of a processor block
int NumBinRowsPerProc;  // Number of rows of bins per processor
int NumBinCols;  // Number of columns of bins (same across processors)

// Convenience variables for reuse
int MaxNumPtsPerRow;  // Maximum number of particles per row of bins
int MaxNumPtsPerProc;  // Maximum number of particles per processor

// Processor-specific variables
double MyMinX;  // Minimum X value of this processor

// The bins containing all the relevant particles (plus padding)
unordered_set<int> *Bins;

enum Direction {
    TOP, BOTTOM
};

// Buffers for receiving particles top & bottom neighbors
vector<particle_t> TopBuffer;
vector<particle_t> BottomBuffer;

// Buffers particles that are about to move to/from a different processor
vector<particle_t> *OutgoingPtsBuffer;
vector<particle_t> *IncomingPtsBuffer;

//////////////////////////////////////// Helper Functions ////////////////////////////////////////

// Calculate the rank that the input particle belongs to
inline int get_particle_rank(const particle_t &pt) {
    return floor(pt.x / ProcHeight);
}

// Get the particle's index in the Bins array
inline int which_bin(const particle_t &pt) {
    int row_idx = (int) floor((pt.x - MyMinX) / BIN_SIZE) + 1;
    int col_idx = floor(pt.y / BIN_SIZE);
    return BIN_IDX(row_idx, col_idx);
}

// Insert the input particle to the correct bin
inline void put_one_pt_to_bin(const particle_t &pt) {
    int idx = which_bin(pt);
    Bins[idx].insert(pt.id);
}

// Get the rank of a neighbor processor (if valid)
inline int get_neighbor_proc_rank(Direction nei_dir, int my_rank, int num_proc) {
    int nei_rank = my_rank;
    if (nei_dir == TOP) {
        nei_rank -= 1;
    } else {
        assert(nei_dir == BOTTOM);  // TODO: remove once debugging done
        nei_rank += 1;
    }

    if (nei_rank < 0 || nei_rank >= num_proc) {
        return -1;
    }
    return nei_rank;
}

// Add all particles in a bin to a vector
void copy_add_pts_from_bin_to_vec(const unordered_set<int> &bin,
                                  particle_t *parts,
                                  vector<particle_t> &vec) {
    for (const auto &pt_id : bin) {
        vec.push_back(parts[pt_id]);  // Push copy of particle struct to vector
    }
}

// Collect particles from certain bins and put their copies in a vector
void copy_pts_in_halo_bins(Direction which_row,
                           particle_t *parts,
                           vector<particle_t> &pts_buffer) {
    int row_idx = which_row == TOP ? 1 : NumBinRowsPerProc;
    for (int c = 0; c <= NumBinCols; ++c) {
        const auto &bin = Bins[BIN_IDX(row_idx, c)];
        copy_add_pts_from_bin_to_vec(bin, parts, pts_buffer);
    }
}

// Collect all particle structs from my inner bins
void copy_pts_from_all_my_bins(particle_t* parts, vector<particle_t> &dest_vec) {
    for (int r = 1; r <= NumBinRowsPerProc; ++r) {
        for (int c = 0; c < NumBinCols; ++c) {
            copy_add_pts_from_bin_to_vec(Bins[BIN_IDX(r, c)], parts, dest_vec);
        }
    }
}

// Put received particles into where they belong in the particle array and the bins
void distribute_received_particles(vector<particle_t> recv_buffer, int recv_cnt, particle_t* parts) {
    for (int i = 0; i < recv_cnt; i++) {
        particle_t &pt = recv_buffer[i];
        uint64_t arr_idx = pt.id - 1;
        parts[arr_idx] = pt;
        put_one_pt_to_bin(pt);
    }
}

// Communicate with horizontal and vertical neighbors
void communicate_with_neighbor_proc(Direction nei_dir, particle_t *parts, int rank, int num_proc) {
    int nei_rank = get_neighbor_proc_rank(nei_dir, rank, num_proc);
    if (nei_rank != -1) {  // If neighbor exists
        // Collect to-be-sent particles from boundary bins
        vector<particle_t> pts_to_send;
        copy_pts_in_halo_bins(nei_dir, parts, pts_to_send);
        // Get the receiving recv_buffer
        auto &recv_buffer = nei_dir == TOP ? TopBuffer : BottomBuffer;
        // Send and receive
        MPI_Status status;
        MPI_Sendrecv(&pts_to_send[0], pts_to_send.size(), PARTICLE,
                     nei_rank, 1234,
                     &recv_buffer[0], MaxNumPtsPerRow, PARTICLE,
                     nei_rank, 1234,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Update particle array and bins with received particles
        distribute_received_particles(recv_buffer, recv_cnt, parts);
    }
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
inline void interact_with_neighbor_bin(particle_t* parts, int pt_id, int nei_row, int nei_col) {
    // Interact with all particles in a valid neighbor bin
    for (auto &nei_id : Bins[BIN_IDX(nei_row, nei_col)]) {
        apply_force(parts[pt_id], parts[nei_id]);
    }
}

// Helper function to calculate 9-by-9 bins
inline void calculate_bin_forces(particle_t *parts, int row, int col) {
    // For each particle in the input bin
    for (auto &pt_id : Bins[BIN_IDX(row, col)]) {
        // Interact with all valid neighboring bins
        interact_with_neighbor_bin(parts, pt_id, row, col);  // Self
        interact_with_neighbor_bin(parts, pt_id, row - 1, col);  // Top
        interact_with_neighbor_bin(parts, pt_id, row + 1, col);  // Bottom
        interact_with_neighbor_bin(parts, pt_id, row, col - 1);  // Left
        interact_with_neighbor_bin(parts, pt_id, row, col + 1);  // Right
        interact_with_neighbor_bin(parts, pt_id, row - 1, col - 1);  // Top left
        interact_with_neighbor_bin(parts, pt_id, row - 1, col + 1);  // Top right
        interact_with_neighbor_bin(parts, pt_id, row + 1, col - 1);  // Bottom left
        interact_with_neighbor_bin(parts, pt_id, row + 1, col + 1);  // Bottom right
    }
}

// Exhaust acceleration to move a particle
void move_one_particle(particle_t &p, double size) {
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

    p.ax = p.ay = 0;  // Important: exhaust all acceleration!
}

// Copy particle structs in the source vector to correct slots in the destination array
void copy_pts_to_arr_by_id(const vector<particle_t> &src_pts_vec, int N_src,
                           particle_t *dest_pts_arr) {
    for (int i = 0; i < N_src; ++i) {
        particle_t pt_cpy = src_pts_vec[i];
        uint64_t idx = pt_cpy.id - 1;
        dest_pts_arr[idx] = pt_cpy;
    }
}

// Exchange moved particles with neighbor processor
void move_particle_cross_processor(Direction nei_dir, int rank, int num_proc, particle_t *parts) {
    int nei_rank = get_neighbor_proc_rank(nei_dir, rank, num_proc);
    if (nei_rank != -1) {  // If neighbor is valid
        // Prepare all cross-processor particles to send
        vector<particle_t> &to_send = OutgoingPtsBuffer[nei_dir];
        // Prepare buffer to receive cross-processor particles
        vector<particle_t> &to_recv = IncomingPtsBuffer[nei_dir];
        // Two-way communication to avoid deadlock
        MPI_Status status;
        MPI_Sendrecv(&to_send[0], to_send.size(), PARTICLE,
                     nei_rank, 1234,
                     &to_recv[0], MaxNumPtsPerProc, PARTICLE,
                     nei_rank, 1234,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Clear all sent particles
        to_send.clear();
        // Put all received particles to the right bins
        distribute_received_particles(to_recv, recv_cnt, parts);
    }
}

///////////////////////////////////////// Key Functions /////////////////////////////////////////

// Initialize data objects that we need
void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    // Calculate global variables
    ProcHeight = size / num_proc;  // Distribute input grid to processors
    NumBinRowsPerProc = ceil(ProcHeight / BIN_SIZE);
    NumBinCols = ceil(size / BIN_SIZE);
    // FIXME: do we need to account for padding (like below)?
    MaxNumPtsPerRow = MAX_NUM_PTS_PER_BIN * NumBinCols;
    MaxNumPtsPerProc = MaxNumPtsPerRow * NumBinRowsPerProc;
    MyMinX = rank * ProcHeight;

    // Initialize bins specific to this processor (+2 because of padding)
    int num_bins_with_padding = (NumBinRowsPerProc + 2) * NumBinCols;
    Bins = new unordered_set<int>[num_bins_with_padding];
    // Collecting particles belonging to this processor to their corresponding bins
    for (int i = 0; i < num_parts; i++) {
        const particle_t &pt = parts[i];
        if (get_particle_rank(pt) == rank) {
            put_one_pt_to_bin(pt);
        }
    }

    // Allocate memory to receiving buffers
    TopBuffer.reserve(MaxNumPtsPerRow);
    BottomBuffer.reserve(MaxNumPtsPerRow);

    // Allocate memory to buffers for cross-processor particles
    OutgoingPtsBuffer = new vector<particle_t>[2];
    IncomingPtsBuffer = new vector<particle_t>[2];
    OutgoingPtsBuffer[TOP].reserve(MaxNumPtsPerProc);
    OutgoingPtsBuffer[BOTTOM].reserve(MaxNumPtsPerProc);
    IncomingPtsBuffer[TOP].reserve(MaxNumPtsPerProc);
    IncomingPtsBuffer[BOTTOM].reserve(MaxNumPtsPerProc);
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    // Exchange particles with top and bottom neighbor processors
    communicate_with_neighbor_proc(TOP, parts, rank, num_proc);
    communicate_with_neighbor_proc(BOTTOM, parts, rank, num_proc);

    // Calculate forces in each bin
    for (int r = 1; r <= NumBinRowsPerProc; ++r) {
        for (int c = 0; c < NumBinCols; ++c) {
            calculate_bin_forces(parts, r, c);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    // Apply forces to all particles in inner bins
    for (int r = 1; r <= NumBinRowsPerProc; ++r) {
        for (int c = 0; c < NumBinCols; ++c) {
            int bin_idx = BIN_IDX(r, c);
            unordered_set<int> &bin = Bins[bin_idx];
            auto itr = bin.begin();
            while (itr != bin.end()) {
                int pt_idx = *itr;
                particle_t pt_struct = parts[pt_idx];
                // Move the particle
                move_one_particle(pt_struct, size);
                int new_rank = get_particle_rank(pt_struct);
                int new_bin_idx = which_bin(pt_struct);
                // If the particle should stay in the same bin
                if (new_rank == rank && new_bin_idx == bin_idx) {
                    ++itr;  // Continue
                } else {  // If the particle should change bins
                    // Remove from current bin
                    itr = Bins[bin_idx].erase(itr);
                    // If particle is bound to another processor
                    if (new_rank != rank) {
                        // Push copy of particle to outgoing buffer
                        Direction which_nei = new_rank < rank ? TOP : BOTTOM;
                        OutgoingPtsBuffer[which_nei].push_back(pt_struct);
                    } else {  // If particle should stay in same processor
                        // Insert into another local bin
                        put_one_pt_to_bin(pt_struct);
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    // Exchange moved particles with top & bottom neighbor processors
    move_particle_cross_processor(TOP, rank, num_proc, parts);
    move_particle_cross_processor(BOTTOM, rank, num_proc, parts);

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?
}

/* Write this function such that at the end of it, the master (rank == 0)
 * processor has an in-order view of all particles. That is, the array
 * parts is complete and sorted by particle id. */
// NOTE: Performance isn't critical here as this function is only called when outputting (-o)
void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    int num_parts_received = 0;  // Received number of particles (debug)

    // Collect all the particles in my inner bins
    vector<particle_t> my_pts;
    copy_pts_from_all_my_bins(parts, my_pts);

    if (rank != 0) {  // If not master, send my particles to master
        MPI_Send(&my_pts[0], my_pts.size(), PARTICLE, 0,
                 1234, MPI_COMM_WORLD);
    } else {  // If is master
        // Write master particles to correct slots
        copy_pts_to_arr_by_id(my_pts, my_pts.size(), parts);
        num_parts_received += my_pts.size();

        // Create buffer for receiving particles
        vector<particle_t> buffer(MaxNumPtsPerProc);
        // Receive particles from all other useful processors
        for (int p = 1; p < num_proc; ++p) {
            MPI_Status status;
            MPI_Recv(&buffer[0], MaxNumPtsPerProc, PARTICLE, p,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int recv_cnt;
            MPI_Get_count(&status, PARTICLE, &recv_cnt);
            // Write received particles to correct slots
            copy_pts_to_arr_by_id(buffer, recv_cnt, parts);
            num_parts_received += recv_cnt;
        }
        // Make sure all particles have been gathered
        assert(num_parts_received == num_parts);
    }
}
