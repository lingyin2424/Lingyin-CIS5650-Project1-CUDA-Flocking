#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <vector>

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


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


#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm,
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

namespace Boids {
    void initSimulation(int N);
    //void stepSimulationNaive(float dt);
    //void stepSimulationScatteredGrid(float dt);
    //void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();

    void StepSimulation_1(float dt);
    void StepSimulation_2(float dt);
    void StepSimulation_3(float dt);
    void StepSimulation_4(float dt);
	void StepSimulation_5(float dt);


    inline int numObjects;
    inline dim3 threadsPerBlock(blockSize);

    inline glm::vec3* dev_pos;
    inline glm::vec3* dev_vel1;
    inline glm::vec3* dev_vel2;
}


