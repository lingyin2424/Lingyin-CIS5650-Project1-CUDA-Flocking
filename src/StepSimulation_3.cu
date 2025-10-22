
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

void Boids::StepSimulation_3(float dt) {

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
    //const int size_of_pos = n * sizeof(glm::vec3);

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
            float last_x = -FLT_MAX, count = 0, p = -1;
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
            float last_y = -FLT_MAX, count = 0, p = -1;
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
            float last_z = -FLT_MAX, count = 0, p = -1;
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
    CalculateNewSpeed << <(n + Block - 1) / Block, Block >> > (
        n, dt, 
        dev_pos, dev_org_speed, dev_new_speed, dev_box_index,
        dev_index,
        dev_begin_of_box, dev_size_of_box, B);

}
