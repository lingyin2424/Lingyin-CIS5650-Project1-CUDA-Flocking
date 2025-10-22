
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
LINGYIN��
ȫ�ֱ���˵����
glm::vec3* dev_pos: λ�����飨GPU��
glm::vec3* dev_vel1: �ٶ�����1��GPU��
glm::vec3* dev_vel2: �ٶ�����2��GPU��
int n: boids ����

����Ҫ�����ǣ�
���ݵ�ǰ���ٶȼ����µ��ٶȣ�Ȼ������µ��ٶȸ���λ�á�
����� dt ����һ��ʱ�䲽�������ڸ���λ��ʱλ�Ƶ�Ȩ��
pos = pos + new_speed * dt��
*/


void Boids::StepSimulation_5(float dt) {

}