#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <numeric>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include <map>
#include<tuple>
#include <algorithm>
#include <random>

#define CUDA_CHECK_KERNEL()                                                \
    do {                                                                   \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                          \
            std::cerr << "Kernel launch error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
        err = cudaDeviceSynchronize();                                     \
        if (err != cudaSuccess) {                                          \
            std::cerr << "Kernel execution error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA (call)                                                     \
{                                                                  \
    const cudaError_t error = call;                                \
    if (error != cudaSuccess)                                      \
    {                                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);             \
        printf("code: %d, reason: %s\n", error,                   \
               cudaGetErrorString(error));                         \
        exit(1);                                                   \
    }                                                              \
}

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 15.f
#define rule2Distance 11.f
#define rule3Distance 15.f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3* dev_pos;
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
    thrust::default_random_engine rng(hash((int)(index * time)));
    thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

    return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
        glm::vec3 rand = generateRandomVec3(time, index);
        arr[index].x = scale * rand.x;
        arr[index].y = scale * rand.y;
        arr[index].z = scale * rand.z;
    }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
    numObjects = N;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    // LOOK-1.2 - This is basic CUDA memory management and error checking.
    // Don't forget to cudaFree in  Boids::endSimulation.
    cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

    cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

    cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

    // LOOK-1.2 - This is a typical CUDA kernel invocation.
    kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects,
        dev_pos, scene_scale);
    checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

    // LOOK-2.1 computing grid params
    gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
    int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
    gridSideCount = 2 * halfSideCount;

    gridCellCount = gridSideCount * gridSideCount * gridSideCount;
    gridInverseCellWidth = 1.0f / gridCellWidth;
    float halfGridWidth = gridCellWidth * halfSideCount;
    gridMinimum.x -= halfGridWidth;
    gridMinimum.y -= halfGridWidth;
    gridMinimum.z -= halfGridWidth;

    // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
    cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale = -1.0f / s_scale;

    if (index < N) {
        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1.0f;
    }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index < N) {
        vbo[4 * index + 0] = vel[index].x + 0.3f;
        vbo[4 * index + 1] = vel[index].y + 0.3f;
        vbo[4 * index + 2] = vel[index].z + 0.3f;
        vbo[4 * index + 3] = 1.0f;
    }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
    kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

    checkCUDAErrorWithLine("copyBoidsToVBO failed!");

    cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
//__device__ glm::vec3 computeVelocityChange(int t, int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
//  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
//  // Rule 2: boids try to stay a distance d away from each other
//  // Rule 3: boids try to match the speed of surrounding boids
//
//
//	glm::vec3 velocity_change(0.0f);
//    glm::vec3 perceived_center(0.0f);
//	glm::vec3 mean_velocity(0.0f);
//    int near_count = 0;
//    glm::vec3 mean(0.0f);
//
//    int neighbors_count = 0;
//
//    for (int i = 0; i < N; i++) {
//		mean += pos[i];
//        if (i == iSelf) continue;  // 排除自己
//        float distance = glm::length(pos[i] - pos[iSelf]);
//        if (distance < rule1Distance) {
//
//            if(distance < rule2Distance) {
//                near_count++;
//                velocity_change -= (pos[i] - pos[iSelf]) * rule2Scale;
//			}
//            perceived_center += pos[i];
//			mean_velocity += vel[i];
//            neighbors_count++;
//        }
//    }
//
//    if (neighbors_count > 0) {
//        perceived_center /= neighbors_count;
//		mean_velocity /= neighbors_count;
//        velocity_change += (perceived_center - pos[iSelf]) * rule1Scale;
//		velocity_change += (mean_velocity - vel[iSelf]) * rule3Scale;
//    }
//
//    mean /= N;
//	float d = glm::length(mean - pos[iSelf]);
//    if (d > 200.0f) {
//	    velocity_change += (mean - pos[iSelf]) * 0.00001f;
//    }
//
//    auto rand = generateRandomVec3(t + 1, iSelf + 1) * 0.00001f;
//    
//    if (neighbors_count && near_count < 3) {
//        rand *= 1.5;
//    }
//    return velocity_change + rand;
//}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
    glm::vec3* vel1, glm::vec3* vel2) {
    // Compute a new velocity based on pos and vel1
    // Clamp the speed
    // Record the new velocity into vel2. Question: why NOT vel1?
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
    // Update position by velocity
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    glm::vec3 thisPos = pos[index];
    thisPos += vel[index] * dt;

    // Wrap the boids around so we don't lose them
    thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
    thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
    thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

    thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
    thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
    thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

    pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
    return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth,
    glm::vec3* pos, int* indices, int* gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
        intBuffer[index] = value;
    }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
    int* gridCellStartIndices, int* gridCellEndIndices) {
    // TODO-2.1
    // Identify the start point of each cell in the gridIndices array.
    // This is basically a parallel unrolling of a loop that goes
    // "this index doesn't match the one before it, must be a new cell!"
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
    // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
    // except with one less level of indirection.
    // This should expect gridCellStartIndices and gridCellEndIndices to refer
    // directly to pos and vel1.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    //   DIFFERENCE: For best results, consider what order the cells should be
    //   checked in to maximize the memory benefits of reordering the boids data.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/

__host__ __device__ int vec3_to_index(glm::vec3 pos, glm::vec3 min_pos, float width, int B) {
    auto tmp = (pos - min_pos) / width;
    int x = (int)std::floor(tmp.x);
    int y = (int)std::floor(tmp.y);
    int z = (int)std::floor(tmp.z);
    return x + y * B + z * B * B;
}

//__global__ void Simple(int t, int N, glm::vec3* org, glm::vec3* aim, glm::vec3* pos, int B) {
//
//	int id_x = blockIdx.x* blockDim.x + threadIdx.x;
//    auto vel_change = computeVelocityChange(t, N, id_x, pos, org);
//	aim[id_x] = org[id_x] + vel_change;
//}

__global__ void Simple(int t, int n,
    glm::vec3* org, glm::vec3* aim, glm::vec3* pos, int* launch_index, int* box_index,
    int* begin_index, int* size_index, int B) {

    int my_launch_index = launch_index[blockIdx.x * blockDim.x + threadIdx.x];

    auto my_pos = pos[my_launch_index];
    auto my_box_index = box_index[my_launch_index];// = vec3_to_index(my_pos, min_pos, width, B);


    glm::vec3 velocity_change(0.0f);
    glm::vec3 perceived_center(0.0f);
    glm::vec3 mean_velocity(0.0f);
    int near_count = 0;
    glm::vec3 mean(0.0f);

    int neighbors_count = 0;

    auto iSelf = my_launch_index;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int near_box_index = my_box_index + dx + dy * B + dz * B * B;
                if (0 <= near_box_index && near_box_index < B * B * B) {
                    for (int i = begin_index[near_box_index]; i < begin_index[near_box_index] + size_index[near_box_index]; i++) {
                        mean += pos[i];
                        if (i == iSelf) continue;  // 排除自己
                        float distance = glm::length(pos[i] - pos[iSelf]);
                        if (distance < rule1Distance) {

                            if (distance < rule2Distance) {
                                near_count++;
                                velocity_change -= (pos[i] - pos[iSelf]) * rule2Scale;
                            }
                            perceived_center += pos[i];
                            mean_velocity += org[i];
                            neighbors_count++;
                        }
                    }
                }
            }
        }
    }

    if (neighbors_count > 0) {
        perceived_center /= neighbors_count;
        mean_velocity /= neighbors_count;
        velocity_change += (perceived_center - pos[iSelf]) * rule1Scale;
        velocity_change += (mean_velocity - org[iSelf]) * rule3Scale;
    }




    auto rand = generateRandomVec3(t + 1, iSelf + 1) * 0.00001f;

    if (neighbors_count && near_count < 3) {
        rand *= 1.5;
    }
    velocity_change += rand;

    float f = hash((unsigned int)velocity_change.x) % 1000 / 1000.0;
    f = pow(f, 3) * 1000 / glm::length(org[iSelf]);
    aim[iSelf] = org[iSelf] + velocity_change * f;
}

__global__ void MoveBoids(float dt, glm::vec3* pos, glm::vec3* vel) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    pos[id_x] += vel[id_x] * dt;
}

__global__ void randAllVel(int t, glm::vec3* vel1, glm::vec3* vel2) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    vel1[id_x] = generateRandomVec3(t + 1, id_x);
    vel2[id_x] = generateRandomVec3(t + 1, id_x);
}

template<class T>
T* to_device(const std::vector<T>& it) {
    auto p = it.data();
    T* ret = nullptr;
    cudaMalloc((void**)&ret, it.size() * sizeof(T));
    cudaMemcpy(ret, p, it.size() * sizeof(T), cudaMemcpyHostToDevice);
    return ret;
}


__global__ void CalculateNewSpeed(
    glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* box_index,
    int* index,
    int* launch_index,
    int* begin_of_box, int* size_of_box, int B) {

    int id = index[launch_index[blockIdx.x * blockDim.x + threadIdx.x]];


    glm::vec3 ret(0.0f);
    glm::vec3 mean_neighbor_speed(0.0f);
    glm::vec3 sum_near_speed(0.0f);
    glm::vec3 center_neighbor(0.0f);


    int near_count = 0;
    int neighbors_count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int near_box_index = box_index[id] + dx + dy * B + dz * B * B;
                if (0 <= near_box_index && near_box_index < B * B * B) {
                    for (int i = begin_of_box[near_box_index]; i < begin_of_box[near_box_index] + size_of_box[near_box_index]; i++) {
                        int it = index[i];
                        auto distance = glm::length(pos[it] - pos[id]);

                        if (it == id || distance > rule1Distance) {
                            continue;
                        }

                        if (distance < rule2Distance) {
                            near_count++;
                            ret -= (pos[it] - pos[id]) * rule2Scale;
                        }

                        neighbors_count++;
                        center_neighbor += pos[it];
                        mean_neighbor_speed += org_speed[it];

                    }
                }
            }
        }
    }

    if (neighbors_count > 0) {
        center_neighbor /= neighbors_count;
        mean_neighbor_speed /= neighbors_count;
        ret += (center_neighbor - pos[id]) * rule1Scale;
        ret += (mean_neighbor_speed - org_speed[id]) * rule3Scale;
    }

    auto rand = generateRandomVec3(1, id + 1) * 0.0001f;
    if (neighbors_count && near_count < 3) {
        rand *= 1.5;
    }
    ret += rand;

    new_speed[id] = org_speed[id] + ret;
}

__global__ void CalculateNewSpeedNaive(
    int n, glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* index) {

    int id = index[blockIdx.x * blockDim.x + threadIdx.x];


    glm::vec3 ret(0.0f);
    glm::vec3 mean_neighbor_speed(0.0f);
    glm::vec3 sum_near_speed(0.0f);
    glm::vec3 center_neighbor(0.0f);


    int near_count = 0;
    int neighbors_count = 0;
    for (int it = 0; it < n; it++) {
        auto distance = glm::length(pos[it] - pos[id]);

        if (it == id || distance > rule1Distance) {
            continue;
        }

        if (distance < rule2Distance) {
            near_count++;
            ret -= (pos[it] - pos[id]) * rule2Scale;
        }

        neighbors_count++;
        center_neighbor += pos[it];
        mean_neighbor_speed += org_speed[it];




    }

    if (neighbors_count > 0) {
        center_neighbor /= neighbors_count;
        mean_neighbor_speed /= neighbors_count;
        ret += (center_neighbor - pos[id]) * rule1Scale;
        ret += (mean_neighbor_speed - org_speed[id]) * rule3Scale;
    }

    auto rand = generateRandomVec3(1, id + 1) * 0.0001f;
    if (neighbors_count && near_count < 3) {
        rand *= 1.5;
    }
    ret += rand;

    new_speed[id] = org_speed[id] + ret;
}


void Boids::stepSimulationNaive(float dt) {
    static int __ = 0;

    static int* dev_box_index = nullptr;
    static int* dev_index = nullptr;
    static int* dev_launch_index = nullptr;
    static int* dev_begin_of_box = nullptr;
    static int* dev_size_of_box = nullptr;

    static char* dev_buffer = nullptr;
    static char* host_buffer = nullptr;
    static glm::vec3* dev_org_speed = nullptr;
    static glm::vec3* dev_new_speed = nullptr;

    int n = numObjects;
    const int B = 128;
    const int BBB = B * B * B;
    const int max_count_of_thread = BBB + n;

    const int size_of_buffer = (
        n * sizeof(int) +
        n * sizeof(int) +
        max_count_of_thread * sizeof(int) +
        BBB * sizeof(int) +
        BBB * sizeof(int));
    const int size_of_pos = n * sizeof(glm::vec3);

    static int* host_box_index = nullptr;
    static int* host_index = nullptr;
    static int* host_launch_index = nullptr;
    static int* host_begin_of_box = nullptr;
    static int* host_size_of_box = nullptr;
    static std::vector<glm::vec3> host_pos;

    __++;
    if (__ == 1) {
        cudaMalloc((void**)&dev_buffer, size_of_buffer);
        host_buffer = new char[size_of_buffer];

        dev_box_index = (int*)(dev_buffer);
        dev_index = (int*)(dev_box_index + n);
        dev_launch_index = (int*)(dev_index + n);
        dev_begin_of_box = (int*)(dev_launch_index + max_count_of_thread);
        dev_size_of_box = (int*)(dev_begin_of_box + BBB);

        host_box_index = (int*)(host_buffer);
        host_index = (int*)(host_box_index + n);
        host_launch_index = (int*)(host_index + n);
        host_begin_of_box = (int*)(host_launch_index + max_count_of_thread);
        host_size_of_box = (int*)(host_begin_of_box + BBB);

        host_pos.resize(n);
        dev_org_speed = dev_vel1;
        dev_new_speed = dev_vel2;
        CUDA_CHECK_KERNEL();
    }

    std::swap(dev_org_speed, dev_new_speed);
    std::memset(host_buffer, 0, size_of_buffer);
    cudaMemcpy(host_pos.data(), dev_pos, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    {
        std::vector<int > index(n);
        for (int i = 0; i < n; i++) {
            index[i] = i;
        }

        float r = rule1Distance;
        { // x
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].x < host_pos[b].x;
                });
            float last_x = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].x <= last_x + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += i - 1;
                }

                last_x = host_pos[index[p]].x;
                count = 0;
            }
        }

        { // y
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].y < host_pos[b].y;
                });
            float last_y = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].y <= last_y + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += (i - 1) * B;
                }
                last_y = host_pos[index[p]].y;
                count = 0;
            }
        }

        { // z
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].z < host_pos[b].z;
                });
            float last_z = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].z <= last_z + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += (i - 1) * B * B;
                }
                last_z = host_pos[index[p]].z;
                count = 0;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        host_index[i] = i;
    }

    std::sort(host_index, host_index + n, [&](const int& a, const int& b) {
        return host_box_index[a] < host_box_index[b];
        });

    for (int i = 0; i < n; i++) {
        if (host_box_index[i] < BBB) {
            host_size_of_box[host_box_index[i]]++;
        }
    }
    for (int i = 1; i < BBB; i++) {
        host_begin_of_box[i] = host_begin_of_box[i - 1] + host_size_of_box[i - 1];
    }

    int used_count_of_thread = 0;
    const int Block = 512;

    used_count_of_thread = (n + Block - 1) / Block * Block;
    for (int i = 0; i < used_count_of_thread; i++) {
        host_launch_index[i] = std::min(i, n - 1);
    }

    cudaMemcpy(dev_buffer, host_buffer, size_of_buffer, cudaMemcpyHostToDevice);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();

    std::cout << "used_count_of_thread: " << used_count_of_thread << std::endl;
    CalculateNewSpeed << <used_count_of_thread / Block, Block >> > (
        dev_pos, dev_org_speed, dev_new_speed, dev_box_index,
        dev_index,
        dev_launch_index,
        dev_begin_of_box, dev_size_of_box, B);
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL();

    MoveBoids << <n / Block, Block >> > (dt, dev_pos, dev_new_speed);
    CUDA_CHECK_KERNEL();

}

//void Boids::stepSimulationNaive(float dt) {
//
//    static int __ = 0;
//    int n = numObjects;
//    const int Block = 512;
//
//    static glm::vec3* dev_org_speed = nullptr;
//    static glm::vec3* dev_new_speed = nullptr;
//    static int* dev_index = nullptr;
//
//    int used_count_of_thread = (n + 512 - 1) / 512 * 512;
//    __++;
//    if (__ == 1) {
//        std::vector<int> index(used_count_of_thread);
//        cudaMalloc((void**)&dev_index, used_count_of_thread * sizeof(int));
//        for (int i = 0; i < used_count_of_thread; i++) {
//            index[i] = std::min(i, n - 1);
//        }
//        cudaMemcpy(dev_index, index.data(), used_count_of_thread * sizeof(int), cudaMemcpyHostToDevice);
//
//        dev_org_speed = dev_vel1;
//        dev_new_speed = dev_vel2;
//        CUDA_CHECK_KERNEL();
//    }
//    std::swap(dev_org_speed, dev_new_speed);
//
//
//    cudaDeviceSynchronize();
//
//    CalculateNewSpeedNaive << <used_count_of_thread / Block, Block >> > (
//        n, dev_pos, dev_org_speed, dev_new_speed, dev_index);
//    cudaDeviceSynchronize();
//    CUDA_CHECK_KERNEL();
//
//    MoveBoids << <n / Block, Block >> > (dt, dev_pos, dev_new_speed);
//    CUDA_CHECK_KERNEL();
//
//    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
//    // TODO-1.2 ping-pong the velocity buffers
//}


void Boids::stepSimulationScatteredGrid(float dt) {
    static int __ = 0;

    static int* dev_box_index = nullptr;
    static int* dev_index = nullptr;
    static int* dev_launch_index = nullptr;
    static int* dev_begin_of_box = nullptr;
    static int* dev_size_of_box = nullptr;

    static glm::vec3* dev_org_speed = nullptr;
    static glm::vec3* dev_new_speed = nullptr;

    int n = numObjects;
    const int B = 32;
    const int BBB = B * B * B;
    const int max_count_of_thread = BBB + n;

    __++;
    if (__ == 1) {
        cudaMalloc((void**)&dev_box_index, n * sizeof(int));
        cudaMalloc((void**)&dev_index, n * sizeof(int));
        cudaMalloc((void**)&dev_launch_index, max_count_of_thread * sizeof(int));
        cudaMalloc((void**)&dev_begin_of_box, BBB * sizeof(int));
        cudaMalloc((void**)&dev_size_of_box, BBB * sizeof(int));
        dev_org_speed = dev_vel1;
        dev_new_speed = dev_vel2;
        CUDA_CHECK_KERNEL();
    }

    std::swap(dev_org_speed, dev_new_speed);
    std::vector<glm::vec3> host_pos(n);
    std::vector<int > host_box_index(n);
    std::vector<int > host_index(n);
    std::vector<int > host_launch_index(max_count_of_thread);
    std::vector<int > host_begin_of_box(BBB, 0);
    std::vector<int > host_size_of_box(BBB, 0);

    cudaMemcpy(host_pos.data(), dev_pos, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    {
        glm::vec3 max_pos(-INFINITY, -INFINITY, -INFINITY);
        glm::vec3 min_pos(+INFINITY, +INFINITY, +INFINITY);
        std::vector<int > index(n);
        for (int i = 0; i < n; i++) {
            index[i] = i;
            min_pos = glm::min(min_pos, host_pos[i]);
            max_pos = glm::max(max_pos, host_pos[i]);
        }

        float width = std::max({
            (max_pos.x - min_pos.x) / B + 0.1f,
            (max_pos.y - min_pos.y) / B + 0.1f,
            (max_pos.z - min_pos.z) / B + 0.1f,
            rule1Distance });

        for (int i = 0; i < n; i++) {
            host_box_index[i] = vec3_to_index(host_pos[i], min_pos, width, B);
        }
    }

    for (int i = 0; i < n; i++) {
        host_index[i] = i;
    }

    std::sort(host_index.begin(), host_index.end(), [&](const int& a, const int& b) {
        return host_box_index[a] < host_box_index[b];
        });

    for (int i = 0; i < n; i++) {
        if (host_box_index[i] < BBB) {
            host_size_of_box[host_box_index[i]]++;
        }
    }
    for (int i = 1; i < BBB; i++) {
        host_begin_of_box[i] = host_begin_of_box[i - 1] + host_size_of_box[i - 1];
    }

    int used_count_of_thread = 0;
    const int Block = 512;
    /*
    {
        std::vector<int >st, rabbish;
        const int Warp = 32;

        for (int i = 0; i < n; i++) {
            if (host_box_index[i] >= BBB) {
                continue;
            }
            int id = host_index[i];
            if (st.size() && host_box_index[host_index[i]] != host_box_index[host_index[st.back()]]) {
                for (auto j : st) {
                    rabbish.push_back(host_index[j]);
                }
                st.clear();
            }

            st.push_back(i);

            if (st.size() == Warp) {
                for (auto j : st) {
                    host_launch_index[used_count_of_thread++] = j;
                }
                st.clear();
            }
        }
        for (auto j : st) {
            rabbish.push_back(host_index[j]);
        }
        st.clear();

        int rest_count_warp = (max_count_of_thread - used_count_of_thread) / Warp;
        int rest_count_task = rabbish.size();
        int per_task_of_warp = (rest_count_task + rest_count_warp - 1) / rest_count_warp;

        while (rabbish.empty() == false) {
            for (int i = 0; i < per_task_of_warp && rabbish.empty() == false; i++) {
                host_launch_index[used_count_of_thread++] = rabbish.back();
                rabbish.pop_back();
            }
            while (used_count_of_thread % Warp != 0) {
                host_launch_index[used_count_of_thread] = host_launch_index[used_count_of_thread - 1];
                used_count_of_thread++;
            }
        }

        used_count_of_thread = (used_count_of_thread + Block - 1) / Block * Block;
    }
    */

    used_count_of_thread = (n + Block - 1) / Block * Block;
    for (int i = 0; i < used_count_of_thread; i++) {
        host_launch_index[i] = std::min(i, n - 1);
    }

    {
        // test
        std::cout << "size of box top 10:    ";
        auto tmp = host_size_of_box;
        std::sort(tmp.begin(), tmp.end());
        for (int i = BBB - 1; i >= BBB - 10; i--) {
            std::cout << tmp[i] << " ";
        }

    }


    cudaMemcpy(dev_pos, host_pos.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_box_index, host_box_index.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_index, host_index.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_launch_index, host_launch_index.data(), max_count_of_thread * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_begin_of_box, host_begin_of_box.data(), BBB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_size_of_box, host_size_of_box.data(), BBB * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK_KERNEL();

    cudaDeviceSynchronize();

    std::cout << "used_count_of_thread: " << used_count_of_thread << std::endl;
    CalculateNewSpeed << <used_count_of_thread / Block, Block >> > (
        dev_pos, dev_org_speed, dev_new_speed, dev_box_index,
        dev_index,
        dev_launch_index,
        dev_begin_of_box, dev_size_of_box, B);
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL();

    MoveBoids << <n / Block, Block >> > (dt, dev_pos, dev_new_speed);
    CUDA_CHECK_KERNEL();
    // TODO-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    // - Perform velocity updates using neighbor search
    // - Update positions
    // - Ping-pong buffers as needed
}

void Boids::stepSimulationCoherentGrid(float dt) {
    static int __ = 0;

    static int* dev_box_index = nullptr;
    static int* dev_index = nullptr;
    static int* dev_launch_index = nullptr;
    static int* dev_begin_of_box = nullptr;
    static int* dev_size_of_box = nullptr;

    static char* dev_buffer = nullptr;
    static char* host_buffer = nullptr;
    static glm::vec3* dev_org_speed = nullptr;
    static glm::vec3* dev_new_speed = nullptr;

    int n = numObjects;
    const int B = 32;
    const int BBB = B * B * B;
    const int max_count_of_thread = BBB + n;

    const int size_of_buffer = (
        n * sizeof(int) +
        n * sizeof(int) +
        max_count_of_thread * sizeof(int) +
        BBB * sizeof(int) +
        BBB * sizeof(int));
    const int size_of_pos = n * sizeof(glm::vec3);

    //std::vector<glm::vec3> _host_pos(n);
    //std::vector<int > _host_box_index(n);
    //std::vector<int > _host_index(n);
    //std::vector<int > _host_launch_index(max_count_of_thread);
    //std::vector<int > _host_begin_of_box(BBB, 0);
    //std::vector<int > _host_size_of_box(BBB, 0);

    static int* host_box_index = nullptr;
    static int* host_index = nullptr;
    static int* host_launch_index = nullptr;
    static int* host_begin_of_box = nullptr;
    static int* host_size_of_box = nullptr;
    static std::vector<glm::vec3> host_pos;

    __++;
    if (__ == 1) {
        cudaMalloc((void**)&dev_buffer, size_of_buffer);
        host_buffer = new char[size_of_buffer];

        dev_box_index = (int*)(dev_buffer);
        dev_index = (int*)(dev_box_index + n);
        dev_launch_index = (int*)(dev_index + n);
        dev_begin_of_box = (int*)(dev_launch_index + max_count_of_thread);
        dev_size_of_box = (int*)(dev_begin_of_box + BBB);

        host_box_index = (int*)(host_buffer);
        host_index = (int*)(host_box_index + n);
        host_launch_index = (int*)(host_index + n);
        host_begin_of_box = (int*)(host_launch_index + max_count_of_thread);
        host_size_of_box = (int*)(host_begin_of_box + BBB);


        host_pos.resize(n);
        dev_org_speed = dev_vel1;
        dev_new_speed = dev_vel2;
        CUDA_CHECK_KERNEL();
    }


    std::swap(dev_org_speed, dev_new_speed);
    std::memset(host_buffer, 0, size_of_buffer);
    cudaMemcpy(host_pos.data(), dev_pos, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    {
        std::vector<int > index(n);
        for (int i = 0; i < n; i++) {
            index[i] = i;
        }

        float r = rule1Distance;
        { // x
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].x < host_pos[b].x;
                });
            float last_x = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].x <= last_x + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += i - 1;
                }

                last_x = host_pos[index[p]].x;
                count = 0;
            }
        }

        { // y
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].y < host_pos[b].y;
                });
            float last_y = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].y <= last_y + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += (i - 1) * B;
                }
                last_y = host_pos[index[p]].y;
                count = 0;
            }
        }

        { // z
            std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
                return host_pos[a].z < host_pos[b].z;
                });
            float last_z = -INFINITY, count = 0, p = -1;
            for (int i = 1; i <= B; i++) {
                while (p + 1 < n && (host_pos[index[p + 1]].z <= last_z + r || count < n / B + 1)) {
                    p++;
                    count++;
                    host_box_index[index[p]] += (i - 1) * B * B;
                }
                last_z = host_pos[index[p]].z;
                count = 0;
            }
        }
    }


    for (int i = 0; i < n; i++) {
        host_index[i] = i;
    }

    std::sort(host_index, host_index + n, [&](const int& a, const int& b) {
        return host_box_index[a] < host_box_index[b];
        });

    for (int i = 0; i < n; i++) {
        if (host_box_index[i] < BBB) {
            host_size_of_box[host_box_index[i]]++;
        }
    }
    for (int i = 1; i < BBB; i++) {
        host_begin_of_box[i] = host_begin_of_box[i - 1] + host_size_of_box[i - 1];
    }

    int used_count_of_thread = 0;
    const int Block = 512;
    /*
    {
        std::vector<int >st, rabbish;
        const int Warp = 32;

        for (int i = 0; i < n; i++) {
            if (host_box_index[i] >= BBB) {
                continue;
            }
            int id = host_index[i];
            if (st.size() && host_box_index[host_index[i]] != host_box_index[host_index[st.back()]]) {
                for (auto j : st) {
                    rabbish.push_back(host_index[j]);
                }
                st.clear();
            }

            st.push_back(i);

            if (st.size() == Warp) {
                for (auto j : st) {
                    host_launch_index[used_count_of_thread++] = j;
                }
                st.clear();
            }
        }
        for (auto j : st) {
            rabbish.push_back(host_index[j]);
        }
        st.clear();

        int rest_count_warp = (max_count_of_thread - used_count_of_thread) / Warp;
        int rest_count_task = rabbish.size();
        int per_task_of_warp = (rest_count_task + rest_count_warp - 1) / rest_count_warp;

        while (rabbish.empty() == false) {
            for (int i = 0; i < per_task_of_warp && rabbish.empty() == false; i++) {
                host_launch_index[used_count_of_thread++] = rabbish.back();
                rabbish.pop_back();
            }
            while (used_count_of_thread % Warp != 0) {
                host_launch_index[used_count_of_thread] = host_launch_index[used_count_of_thread - 1];
                used_count_of_thread++;
            }
        }

        used_count_of_thread = (used_count_of_thread + Block - 1) / Block * Block;
    }
    */

    used_count_of_thread = (n + Block - 1) / Block * Block;
    for (int i = 0; i < used_count_of_thread; i++) {
        host_launch_index[i] = std::min(i, n - 1);
    }

    //{
    //    // test
    //    std::cout << "size of box top 10:    ";
    //    auto tmp = host_size_of_box;
    //    std::sort(tmp.begin(), tmp.end());
    //    for (int i = BBB - 1; i >= BBB - 10; i--) {
    //        std::cout << tmp[i] << " ";
    //    }

    //}

    cudaMemcpy(dev_buffer, host_buffer, size_of_buffer, cudaMemcpyHostToDevice);
    ////std::shuffle(host_launch_index.begin(), host_launch_index.begin() + used_count_of_thread, std::default_random_engine(0));
 //   cudaMemcpy(dev_pos, host_pos.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice);
 //   cudaMemcpy(dev_box_index, host_box_index.data(), n * sizeof(int), cudaMemcpyHostToDevice);
 //   cudaMemcpy(dev_index, host_index.data(), n * sizeof(int), cudaMemcpyHostToDevice);
 //   cudaMemcpy(dev_launch_index, host_launch_index.data(), max_count_of_thread * sizeof(int), cudaMemcpyHostToDevice);
 //   cudaMemcpy(dev_begin_of_box, host_begin_of_box.data(), BBB * sizeof(int), cudaMemcpyHostToDevice);
 //   cudaMemcpy(dev_size_of_box, host_size_of_box.data(), BBB * sizeof(int), cudaMemcpyHostToDevice);

    CUDA_CHECK_KERNEL();

    cudaDeviceSynchronize();

    std::cout << "used_count_of_thread: " << used_count_of_thread << std::endl;
    CalculateNewSpeed << <used_count_of_thread / Block, Block >> > (
        dev_pos, dev_org_speed, dev_new_speed, dev_box_index,
        dev_index,
        dev_launch_index,
        dev_begin_of_box, dev_size_of_box, B);
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL();

    MoveBoids << <n / Block, Block >> > (dt, dev_pos, dev_new_speed);
    CUDA_CHECK_KERNEL();
    // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
    // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
    // In Parallel:
    // - Label each particle with its array index as well as its grid index.
    //   Use 2x width grids
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
    //   the particle data in the simulation array.
    //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    // - Perform velocity updates using neighbor search
    // - Update positions
    // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
    cudaFree(dev_vel1);
    cudaFree(dev_vel2);
    cudaFree(dev_pos);

    // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
    // LOOK-1.2 Feel free to write additional tests here.

    // test unstable sort
    int* dev_intKeys;
    int* dev_intValues;
    int N = 10;

    std::unique_ptr<int[]>intKeys{ new int[N] };
    std::unique_ptr<int[]>intValues{ new int[N] };

    intKeys[0] = 0; intValues[0] = 0;
    intKeys[1] = 1; intValues[1] = 1;
    intKeys[2] = 0; intValues[2] = 2;
    intKeys[3] = 3; intValues[3] = 3;
    intKeys[4] = 0; intValues[4] = 4;
    intKeys[5] = 2; intValues[5] = 5;
    intKeys[6] = 2; intValues[6] = 6;
    intKeys[7] = 0; intValues[7] = 7;
    intKeys[8] = 5; intValues[8] = 8;
    intKeys[9] = 6; intValues[9] = 9;

    cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

    cudaMalloc((void**)&dev_intValues, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    std::cout << "before unstable sort: " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  key: " << intKeys[i];
        std::cout << " value: " << intValues[i] << std::endl;
    }

    // How to copy data to the GPU
    cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

    // Wrap device vectors in thrust iterators for use with thrust.
    thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
    thrust::device_ptr<int> dev_thrust_values(dev_intValues);
    // LOOK-2.1 Example for using thrust::sort_by_key
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

    // How to copy data back to the CPU side from the GPU
    cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("memcpy back failed!");

    std::cout << "after unstable sort: " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  key: " << intKeys[i];
        std::cout << " value: " << intValues[i] << std::endl;
    }

    // cleanup
    cudaFree(dev_intKeys);
    cudaFree(dev_intValues);
    checkCUDAErrorWithLine("cudaFree failed!");
    return;
}
