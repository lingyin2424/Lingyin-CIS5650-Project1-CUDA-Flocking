
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


void Boids::StepSimulation_5(float dt) {

}