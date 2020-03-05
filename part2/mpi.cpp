#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <assert.h>

using namespace std;

///////////////////////////////////// Static Global Variables /////////////////////////////////////

// Length of a bin's side
//#define BIN_SIZE 0.01
double BIN_SIZE;
// Maximum number of particles per bin
#define MAX_NUM_PTS_PER_BIN 20
// Indexing Bins matrix
#define BIN_IDX(row, col) row * (Num_Bins_Per_Proc_Side + 2) + col

int Num_Proc_Per_Side;  // Number of processors per side
int Num_Useful_Proc;  // Total number of processors involved in computation
double Proc_Size;  // Size of a processor's side
//int Num_Bins_Per_Proc_Side;  // Number of bins on a processor's side
#define Num_Bins_Per_Proc_Side 10  // TODO: change to dynamic and scale with proc size
int Max_Num_Pts_Per_Proc;  // Maximum number of particles per processor

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
vector <particle_t> Recv_Buffers[8];

// Buffers particles that are about to move to a different processor
vector <particle_t> *Outgoing_Pts_Buffer;
// Buffers particles that are about to come in from another processor
vector <particle_t> *Incoming_Pts_Buffer;

//////////////////////////////////////// Helper Functions ////////////////////////////////////////

// Calculate the rank that the input particle belongs to
inline int calculate_particle_rank(const particle_t &pt) {
    int row_idx = floor(pt.x / Proc_Size);
    int col_idx = floor(pt.y / Proc_Size);
    // Processors are assigned in row-major
    return row_idx * Num_Proc_Per_Side + col_idx;
}

// Get the particle's index in the Bins array
inline int which_bin(const particle_t &pt) {
    int row_idx = floor((pt.x - My_Min_X) / BIN_SIZE) + 1;
    int col_idx = floor((pt.y - My_Min_Y) / BIN_SIZE) + 1;
    return BIN_IDX(row_idx, col_idx);
}

// Insert the input particle to the correct bin
inline void put_one_pt_addr_to_bin(particle_t &pt) {
    int idx = which_bin(pt);
    Bins[idx].insert(&pt);
}

// Put a vector of particles into corresponding bins
void put_pts_addr_to_bins(vector <particle_t> &pts, int num_pts) {
    for (int i = 0; i < num_pts; i++) {
        put_one_pt_addr_to_bin(pts[i]);
    }
}

// Is the rank used in actual computation?
inline bool is_useful_rank(int rank) {
    return rank < Num_Useful_Proc;
}

// Get the rank of a neighbor processor (if valid)
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
    int nei_rank = nei_row_idx * Num_Proc_Per_Side + nei_col_idx;
    return is_useful_rank(nei_rank) ? nei_rank : -1;
}

// Add all particles in a bin to a vector
void copy_and_add_pts_struct_from_bin_to_vec(const unordered_set<particle_t *> &bin,
                                             vector <particle_t> &vec) {
    for (auto &pt : bin) {
        vec.push_back(*pt);  // Push copy of particle struct to vector
    }
}

// Collect particles from certain bins and put them in a vector
void copy_pts_from_halo_bins(Direction which_border, vector <particle_t> &pt_vec) {
    if (which_border == TOP || which_border == BOTTOM) {
        int row_idx = which_border == TOP ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            auto &bin = Bins[BIN_IDX(row_idx, i)];
            copy_and_add_pts_struct_from_bin_to_vec(bin, pt_vec);
        }
    } else if (which_border == LEFT || which_border == RIGHT) {
        int col_idx = which_border == LEFT ? 1 : Num_Bins_Per_Proc_Side;
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            auto &bin = Bins[BIN_IDX(i, col_idx)];
            copy_and_add_pts_struct_from_bin_to_vec(bin, pt_vec);
        }
    }
}

// Collect all particle structs from my inner bins
void copy_pts_from_all_my_bins(vector <particle_t> &dest_vec) {
    for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
        for (int j = 1; j <= Num_Bins_Per_Proc_Side; ++j) {
            copy_and_add_pts_struct_from_bin_to_vec(Bins[BIN_IDX(i, j)], dest_vec);
        }
    }
}

// Communicate with horizontal and vertical neighbors | TODO: add assertion check if not too slow
void communicate_with_non_diagonal_neighbors(Direction nei_dir, particle_t* parts) {

    assert(nei_dir == TOP || nei_dir == BOTTOM || nei_dir == LEFT || nei_dir == RIGHT);

    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If neighbor exists
        // Collect to-be-sent particles from their bins
        vector <particle_t> pt_vec;
        copy_pts_from_halo_bins(nei_dir, pt_vec);
        // Get the receiving buffer
        vector <particle_t> &buffer = Recv_Buffers[nei_dir];
        // Send and receive
        MPI_Status status;
        MPI_Sendrecv(&pt_vec[0], pt_vec.size(), PARTICLE,
                     nei_rank, 1234,
                     &buffer[0], MAX_NUM_PTS_PER_BIN * Num_Bins_Per_Proc_Side,
                     PARTICLE, nei_rank, 1234,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Fill received particles into right bins
        for (int i = 0; i < recv_cnt; ++i) {
            particle_t pt_cpy = buffer[i];
            uint64_t idx = pt_cpy.id - 1;
            parts[idx] = pt_cpy;
            put_one_pt_addr_to_bin(pt_cpy);
        }
//        put_pts_addr_to_bins(Recv_Buffers[nei_dir], recv_cnt);
    }
}

// Return the bin on the specified corner
inline int get_corner_bin_idx(Direction dir) {
    int row_idx, col_idx;
    switch (dir) {
        case TOP_LEFT:
            row_idx = 1;
            col_idx = 1;
            break;
        case TOP_RIGHT:
            row_idx = 1;
            col_idx = Num_Bins_Per_Proc_Side;
            break;
        case BOTTOM_LEFT:
            row_idx = Num_Bins_Per_Proc_Side;
            col_idx = 1;
            break;
        case BOTTOM_RIGHT:
            row_idx = Num_Bins_Per_Proc_Side;
            col_idx = Num_Bins_Per_Proc_Side;
            break;
        default:
            return -1;
    }
    return BIN_IDX(row_idx, col_idx);
}

// Communicate with diagonal processors | TODO: add assertion check if not too slow
void communicate_with_diagonal_neighbors(Direction nei_dir, particle_t* parts) {
    int nei_rank = get_neighbor_proc_rank(nei_dir);
    if (nei_rank != -1) {  // If the neighbor exists
        // Copy all the points in the corner bin to a vector
        int bin_idx = get_corner_bin_idx(nei_dir);
        unordered_set < particle_t * > &my_bin = Bins[bin_idx];
        vector <particle_t> my_pts;
        copy_and_add_pts_struct_from_bin_to_vec(my_bin, my_pts);
        // Get the receiving buffer
        vector <particle_t> &buffer = Recv_Buffers[nei_dir];
        MPI_Status status;
        // Send and receive
//        cout << "send size = " << my_pts.size() << endl;
//        cout << "recv capacity = " << buffer.capacity() << endl;
        if (my_pts.size() >= buffer.capacity())
            cout << "FFFFFFFFFFFFFUUUUUUUUUUUUUUCCCCCCCCCCCCKKKKKKKKKKKK" << endl;
        MPI_Sendrecv(&my_pts[0], my_pts.size(), PARTICLE,
                     nei_rank, 5678,
                     &buffer[0], MAX_NUM_PTS_PER_BIN, PARTICLE,
                     nei_rank, 5678,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Fill received particles into right bins
//        put_pts_addr_to_bins(Recv_Buffers[nei_dir], recv_cnt);
        for (int i = 0; i < recv_cnt; ++i) {
            particle_t pt_cpy = buffer[i];
            uint64_t idx = pt_cpy.id - 1;
            parts[idx] = pt_cpy;
            put_one_pt_addr_to_bin(pt_cpy);
        }
    }
}

// Apply the force from neighbor to particle
inline void apply_force(particle_t &particle, particle_t &neighbor) {
    assert(calculate_particle_rank(particle) == My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx);

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
inline void interact_with_neighbor_bin(particle_t *pt, int neiRow, int neiCol) {
    assert(calculate_particle_rank(*pt) == My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx);

    // Interact with all particles in a valid neighbor
    for (auto &neiPts : Bins[BIN_IDX(neiRow, neiCol)]) {
        apply_force(*pt, *neiPts);
    }
}

// Helper function to calculate 9-by-9 bins
inline void calculate_bin_forces(int row, int col) {
    // For each particle in the input bin
    for (auto &pt : Bins[BIN_IDX(row, col)]) {
        pt->ax = pt->ay = 0;
        // Interact with all valid neighboring bins
        interact_with_neighbor_bin(pt, row, col);          // Self
        interact_with_neighbor_bin(pt, row - 1, col);      // Top
        interact_with_neighbor_bin(pt, row + 1, col);      // Bottom
        interact_with_neighbor_bin(pt, row, col - 1);      // Left
        interact_with_neighbor_bin(pt, row, col + 1);      // Right
        interact_with_neighbor_bin(pt, row - 1, col - 1);  // Top left
        interact_with_neighbor_bin(pt, row - 1, col + 1);  // Top right
        interact_with_neighbor_bin(pt, row + 1, col - 1);  // Bottom left
        interact_with_neighbor_bin(pt, row + 1, col + 1);  // Bottom right
    }
}

void move(particle_t &p, double size) {
    assert(is_useful_rank(My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx));
//    assert(calculate_particle_rank(p) == My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx);

//    cout << "rank " << My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx << " in move()" << endl;

    int oldIdx = which_bin(p);
    int oldRank = calculate_particle_rank(p);

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

//    cout << "rank " << My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx << " deciding..." << endl;

    // Put the particle into new places
    int newRank = calculate_particle_rank(p);
    if (newRank != oldRank) {  // If the particle jumps to a new processor
        Outgoing_Pts_Buffer[newRank].push_back(p);  // Push copy of particle to buffer
        Bins[oldIdx].erase(&p);
    } else {  // If the particle stays in this processor
        int newIdx = which_bin(p);
        if (newIdx != oldIdx) {  // If the particle jumps to a different bin
            Bins[newIdx].insert(&p);
            Bins[oldIdx].erase(&p);
        }
    }

//    cout << "rank " << My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx << " decided" << endl;
}

// Copy particle structs in the source vector to correct slots in the destination array
void copy_pts_to_dest_arr(const vector <particle_t> &src_pts_vec, int N_src,
                          particle_t *dest_pts_arr) {
    for (int i = 0; i < N_src; ++i) {
        particle_t pt_cpy = src_pts_vec[i];

        assert(calculate_particle_rank(pt_cpy) == My_Row_Idx * Num_Proc_Per_Side + My_Col_Idx);

        uint64_t idx = pt_cpy.id - 1;
        dest_pts_arr[idx] = pt_cpy;
    }
}

void move_particle_cross_processor(int my_rank, particle_t *parts) {
    if (!is_useful_rank(my_rank)) return;
    // Send and receive all cross-processor particles
    for (int proc = 0; proc < Num_Useful_Proc; ++proc) {
        if (proc == my_rank) continue;
        // Prepare all cross-processor particles to send
        vector <particle_t> &to_send = Outgoing_Pts_Buffer[proc];
        // Prepare buffer to receive cross-processor particles
        vector <particle_t> &to_recv = Incoming_Pts_Buffer[proc];
        // Two-way communication to avoid deadlock
        MPI_Status status;
        MPI_Sendrecv(&to_send[0], to_send.size(), PARTICLE,
                     proc, 1234,
                     &to_recv[0], Max_Num_Pts_Per_Proc, PARTICLE,
                     proc, 1234,
                     MPI_COMM_WORLD, &status);
        int recv_cnt;
        MPI_Get_count(&status, PARTICLE, &recv_cnt);
        // Clear all sent particles
        to_send.clear();
        // DEBUG: make sure all received particles are mine
        for (auto& pt : to_recv) {
            assert(calculate_particle_rank(pt) == my_rank);
        }
        // Put all received particles to the right bins
        for (int i = 0; i < recv_cnt; ++i) {
            particle_t pt_cpy = to_recv[i];
            uint64_t idx = pt_cpy.id - 1;
            parts[idx] = pt_cpy;
            put_one_pt_addr_to_bin(pt_cpy);
        }
    }
}

///////////////////////////////////////// Key Functions /////////////////////////////////////////

// Initialize data objects that we need
void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    if (rank == 0)
        cout << "start init()" << endl;

    // Calculate global processor constants
    Num_Proc_Per_Side = floor(sqrt(num_proc));
    Num_Useful_Proc = Num_Proc_Per_Side * Num_Proc_Per_Side;
    Proc_Size = size / Num_Proc_Per_Side;
    // Calculate global bin & particle constants
//    Num_Bins_Per_Proc_Side = ceil(Proc_Size / BIN_SIZE);
    BIN_SIZE = Proc_Size / Num_Bins_Per_Proc_Side;
    int total_bins_per_proc = Num_Bins_Per_Proc_Side * Num_Bins_Per_Proc_Side;
    Max_Num_Pts_Per_Proc = MAX_NUM_PTS_PER_BIN * total_bins_per_proc;

    // Calculate variables specific to this processor
    My_Row_Idx = rank / Num_Proc_Per_Side;
    My_Col_Idx = rank % Num_Proc_Per_Side;
    My_Min_X = My_Row_Idx * Proc_Size;
    My_Min_Y = My_Col_Idx * Proc_Size;

    if (!is_useful_rank(rank)) return;

    if (rank == 0)
        cout << "initializing bins..." << endl;

    // Initialize bins specific to this processor (+2 because of padding)
    int total_num_bins = (Num_Bins_Per_Proc_Side + 2) * (Num_Bins_Per_Proc_Side + 2);
    Bins = new unordered_set<particle_t *>[total_num_bins];
    // Assign particles belonging to this processor to their corresponding bins
    for (int i = 0; i < num_parts; i++) {
        particle_t &pt = parts[i];
        // If particle belongs to this processor
        if (calculate_particle_rank(pt) == rank) {
            put_one_pt_addr_to_bin(pt);
        }
    }

    if (rank == 0)
        cout << "alloc mem to recv buff..." << endl;

    // Allocate memory to receiving buffers
    Recv_Buffers[TOP].reserve(MAX_NUM_PTS_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[BOTTOM].reserve(MAX_NUM_PTS_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[LEFT].reserve(MAX_NUM_PTS_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[RIGHT].reserve(MAX_NUM_PTS_PER_BIN * Num_Bins_Per_Proc_Side);
    Recv_Buffers[TOP_LEFT].reserve(MAX_NUM_PTS_PER_BIN);
    Recv_Buffers[TOP_RIGHT].reserve(MAX_NUM_PTS_PER_BIN);
    Recv_Buffers[BOTTOM_LEFT].reserve(MAX_NUM_PTS_PER_BIN);
    Recv_Buffers[BOTTOM_RIGHT].reserve(MAX_NUM_PTS_PER_BIN);

    if (rank == 0)
        cout << "alloc mem to cross proc buffer..." << endl;

    // Initialize buffers for cross-processor particles
    Outgoing_Pts_Buffer = new vector<particle_t>[Num_Useful_Proc];
    Incoming_Pts_Buffer = new vector<particle_t>[Num_Useful_Proc];
    for (int i = 0; i < Num_Useful_Proc; i++) {
        Incoming_Pts_Buffer[i].reserve(Max_Num_Pts_Per_Proc);
    }

    if (rank == 0)
        cout << "end init()" << endl;
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_proc) {
    if (is_useful_rank(rank)) {

        for (int r = 1; r <= Num_Bins_Per_Proc_Side; r++) {
            for (int c = 1; c <= Num_Bins_Per_Proc_Side; c++) {
                for (auto &pt : Bins[BIN_IDX(r, c)]) {  // FIXME: invalid read
                    if (calculate_particle_rank(*pt) != rank)
                        cout << "corrupted inner bins" << endl;
                }
            }
        }

        if (rank == 0)
            cout << "communicate with non-diagonal neighbors..." << endl;

        // Communicate with horizontal and vertical neighbor processors
        communicate_with_non_diagonal_neighbors(TOP, parts);
        communicate_with_non_diagonal_neighbors(BOTTOM, parts);
        communicate_with_non_diagonal_neighbors(LEFT, parts);
        communicate_with_non_diagonal_neighbors(RIGHT, parts);

        if (rank == 0)
            cout << "communicate with diagonal neighbors..." << endl;

        // Communicate with diagonal processors
        communicate_with_diagonal_neighbors(TOP_LEFT, parts);
        communicate_with_diagonal_neighbors(TOP_RIGHT, parts);
        communicate_with_diagonal_neighbors(BOTTOM_LEFT, parts);
        communicate_with_diagonal_neighbors(BOTTOM_RIGHT, parts);

        if (rank == 0)
            cout << "apply_force start..." << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    if (is_useful_rank(rank)) {
        // Apply forces in each bin
        for (int i = 1; i <= Num_Bins_Per_Proc_Side; ++i) {
            for (int j = 1; j <= Num_Bins_Per_Proc_Side; ++j) {
                calculate_bin_forces(i, j);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    if (rank == 0)
        cout << "Pass Barrier!" << endl;

    if (is_useful_rank(rank)) {

        if (rank == 0)
            cout << "move() start" << endl;

        // Iterate over inner bins that belong to this processor
        for (int r = 1; r <= Num_Bins_Per_Proc_Side; r++) {
            for (int c = 1; c <= Num_Bins_Per_Proc_Side; c++) {

//                if (Bins[BIN_IDX(r, c)].size() > 9)
//                    cout << "# pts in bin = " << Bins[BIN_IDX(r, c)].size() << endl;

                // Move particles in each bin
                for (auto &pt : Bins[BIN_IDX(r, c)]) {  // FIXME: invalid read
                    move(*pt, size);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    if (rank == 0)
        cout << "move_particle_cross_processor() start" << endl;

//    move_particle_cross_processor(rank, parts);

    MPI_Barrier(MPI_COMM_WORLD);  // TODO: try to remove?

    if (rank == 0)
        cout << "done one step!" << endl;
}

/* Write this function such that at the end of it, the master (rank == 0)
 * processor has an in-order view of all particles. That is, the array
 * parts is complete and sorted by particle id. */
// TODO: maybe try Isend/Irecv
void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_proc) {

    if (rank == 0)
        cout << "gather for save() started!" << endl;

    if (!is_useful_rank(rank)) return;

    // Collect all the particles in my inner bins
    vector <particle_t> my_pts;
    copy_pts_from_all_my_bins(my_pts);

    if (rank != 0) {  // If not master, send my particles to master
        MPI_Send(&my_pts[0], my_pts.size(), PARTICLE, 0,
                 1234, MPI_COMM_WORLD);
    } else {  // If master
        // Write master particles to correct slots
        copy_pts_to_dest_arr(my_pts, my_pts.size(), parts);

        // Create buffer for receiving particles
        vector <particle_t> buffer(Max_Num_Pts_Per_Proc);

        // Receive particles from all other useful processors
        for (int r = 1; r < Num_Useful_Proc; ++r) {
            MPI_Status status;
            MPI_Recv(&buffer[0], Max_Num_Pts_Per_Proc, PARTICLE, r,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int recv_cnt;
            MPI_Get_count(&status, PARTICLE, &recv_cnt);
            // Write received particles to correct slots
            copy_pts_to_dest_arr(buffer, recv_cnt, parts);
        }
    }

    if (rank == 0)
        cout << "gather for save() done!" << endl;
}
