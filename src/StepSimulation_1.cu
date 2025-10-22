
#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <numeric>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include "kernel.cuh"
#include <map>
#include<tuple>
#include <algorithm>
#include <random>
/*
LINGYIN：
全局变量说明：
glm::vec3* dev_pos: 位置数组（GPU）
glm::vec3* dev_vel1: 速度数组1（GPU）
glm::vec3* dev_vel2: 速度数组2（GPU）
int n: boids 数量

你需要做的是：
根据当前的速度计算新的速度，然后根据新的速度更新位置。
输入的 dt 就是一个时间步长，用于更新位置时位移的权重 
pos = pos + new_speed * dt。
*/


void Boids::StepSimulation_1(float dt) {
    static int __ = 0;
    int n = numObjects;
    const int Block = 512;

    static glm::vec3* dev_org_speed = nullptr;
    static glm::vec3* dev_new_speed = nullptr;
    static int* dev_index = nullptr;

    int used_count_of_thread = (n + 512 - 1) / 512 * 512;
    __++;
    if (__ == 1) {
        std::vector<int> index(used_count_of_thread);
        cudaMalloc((void**)&dev_index, used_count_of_thread * sizeof(int));
        for (int i = 0; i < used_count_of_thread; i++) {
            index[i] = std::min(i, n - 1);
        }
        cudaMemcpy(dev_index, index.data(), used_count_of_thread * sizeof(int), cudaMemcpyHostToDevice);

        dev_org_speed = dev_vel1;
        dev_new_speed = dev_vel2;
        CUDA_CHECK_KERNEL();
    }

    std::swap(dev_org_speed, dev_new_speed);

    cudaDeviceSynchronize();

    CalculateNewSpeedNaive << <used_count_of_thread / Block, Block >> > (
        n, dev_pos, dev_org_speed, dev_new_speed, dev_index);
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL();

    //MoveBoids << <n / Block, Block >> > (dt, dev_pos, dev_new_speed);
    //CUDA_CHECK_KERNEL();
}