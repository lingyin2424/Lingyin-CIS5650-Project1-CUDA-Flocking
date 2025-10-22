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
__global__ void UpdateBox(int m,
    int* dev_box_size, int* dev_bex_begin, // BBB
    int* dev_box_size_src, int* dev_bex_begin_src, int* map_to) { // m
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < m) {
        dev_box_size [map_to[idx]] = dev_box_size_src [idx];
	    dev_bex_begin[map_to[idx]] = dev_bex_begin_src[idx];
    }
}

__global__ void ClearBox(int m,
    int* dev_box_size, int* dev_bex_begin, // BBB
    int* map_to) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m) {
        dev_box_size [map_to[idx]] = 0;
        dev_bex_begin[map_to[idx]] = 0;
    }
}

void Boids::StepSimulation_4(float dt) {

    static int __ = 0;
    static char* dev_buffer = nullptr;
    static char* host_buffer = nullptr;

    static int* dev_box_index = nullptr; // n
	static int* dev_index = nullptr; // n
	//static int* dev_launch_index = nullptr; // max_ount_of_thread
	static int* dev_begin_of_box_src = nullptr; // n
	static int* dev_size_of_box_src = nullptr; // n
	static int* dev_map_to = nullptr; // n
    static int* dev_begin_of_box = nullptr; // BBB
    static int* dev_size_of_box = nullptr; //BBB


	//static long long* dev_timestep = nullptr;

    static glm::vec3* dev_org_speed = nullptr;
    static glm::vec3* dev_new_speed = nullptr;

    int n = numObjects;
    const int B = 512; // (1 << 8)**3
    const int BBB = B * B * B;
    const int max_count_of_thread = n * 2;

    static int* host_box_index = nullptr; // n
    static int* host_index = nullptr; // n
    //static int* host_launch_index = nullptr; // max_ount_of_thread
    static int* host_begin_of_box_src = nullptr; // n
    static int* host_size_of_box_src = nullptr; // n
	static int* host_map_to = nullptr; // n
    static std::vector<glm::vec3> host_pos;


    const int size_of_buffer = (
        n * sizeof(int) +
        n * sizeof(int) +
        max_count_of_thread * sizeof(int) +
		n * sizeof(int) +
		n * sizeof(int) + 
		n * sizeof(int) +
        BBB * sizeof(int) +
        BBB * sizeof(int));
    const int size_of_buffer_without_BBB = (
        n * sizeof(int) +
        n * sizeof(int) +
        max_count_of_thread * sizeof(int) +
        n * sizeof(int) +
        n * sizeof(int) +
		n * sizeof(int));
    const int size_of_pos = n * sizeof(glm::vec3);




    __++;
    if (__ == 1) {

		//cudaMalloc((void**)&dev_timestep, sizeof(long long) * );

        cudaMalloc((void**)&dev_buffer, size_of_buffer);
		cudaMemset(dev_buffer, 0, size_of_buffer);
        host_buffer = new char[size_of_buffer_without_BBB]();

        dev_box_index = (int*)dev_buffer;
		dev_index = (int*)(dev_box_index + n);
		//dev_launch_index = (int*)(dev_index + n);
		dev_begin_of_box_src = (int*)(dev_index + max_count_of_thread);
		dev_size_of_box_src = (int*)(dev_begin_of_box_src + n);
		dev_map_to = (int*)(dev_size_of_box_src + n);
		dev_begin_of_box = (int*)(dev_map_to + n);
		dev_size_of_box = (int*)(dev_begin_of_box + BBB);

		host_box_index = (int*)(host_buffer);
		host_index = (int*)(host_box_index + n);
		//host_launch_index = (int*)(host_index + n);
		host_begin_of_box_src = (int*)(host_index + max_count_of_thread);
		host_size_of_box_src = (int*)(host_begin_of_box_src + n);
		host_map_to = (int*)(host_size_of_box_src + n);

        host_pos.resize(n);
        dev_org_speed = dev_vel1;
        dev_new_speed = dev_vel2;
        CUDA_CHECK_KERNEL();
    }

    std::swap(dev_org_speed, dev_new_speed);
    std::memset(host_buffer, 0, size_of_buffer_without_BBB);
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
		assert(0 <= host_box_index[i] && host_box_index[i] < BBB);
    }

    std::sort(host_index, host_index + n, [&](const int& a, const int& b) {
        return host_box_index[a] < host_box_index[b];
        });

    const int Block = 512;
    int count_of_not_empty_box = 0;
    {
		std::vector<int > box_index, box_size, box_begin;
        for (int count = 0, i = 0; i < n; i++) {
            count++;
            if (i + 1 == n || host_box_index[host_index[i]] != host_box_index[host_index[i + 1]]) {
				int bi = host_box_index[host_index[i]];
                box_index.push_back(bi);
                box_size.push_back(count);
                count = 0;
            }
        }

		box_begin.push_back(0);
        for(int i = 1; i < box_size.size(); i++) {
            box_begin.push_back(box_begin[i - 1] + box_size[i - 1]);
		}

        for (int i = 0; i < box_index.size(); i++) {
            host_size_of_box_src[i] = box_size[i];
            host_begin_of_box_src[i] = box_begin[i];
			host_map_to[i] = box_index[i];
        }


		count_of_not_empty_box = box_index.size();
    }

    cudaDeviceSynchronize();
    cudaMemcpy(dev_buffer, host_buffer, size_of_buffer_without_BBB, cudaMemcpyHostToDevice);
    
    UpdateBox << <(count_of_not_empty_box + Block - 1) / Block, Block >> > (
        count_of_not_empty_box,
        dev_size_of_box, dev_begin_of_box,
		dev_size_of_box_src, dev_begin_of_box_src, dev_map_to);

    CUDA_CHECK_KERNEL();

    CalculateNewSpeed << < (n + Block - 1) / Block, Block >> > (
        n, dt,
        dev_pos, dev_org_speed, dev_new_speed, dev_box_index,
        dev_index,
        dev_begin_of_box, dev_size_of_box, B);
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL();

    ClearBox << <(count_of_not_empty_box + Block - 1) / Block, Block >> > (
        count_of_not_empty_box,
        dev_size_of_box, dev_begin_of_box,
        dev_map_to);
}
