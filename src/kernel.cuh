// LINGYIN: 枚举周围的 27 个 格子，计算新的速度并移动 pos
__global__ void CalculateNewSpeed(
    int n, float dt,
    glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* box_index,
    int* index,
    int* begin_of_box, int* size_of_box, int B);
// LINGYIN: 朴素地计算新的速度并移动 pos
__global__ void CalculateNewSpeedNaive(
    int n, glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* index);

// LINGYIN: 根据速度移动 boids，但我没用
__global__ void MoveBoids(float dt, glm::vec3* pos, glm::vec3* vel);

