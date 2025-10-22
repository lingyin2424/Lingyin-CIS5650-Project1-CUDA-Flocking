// LINGYIN: ö����Χ�� 27 �� ���ӣ������µ��ٶȲ��ƶ� pos
__global__ void CalculateNewSpeed(
    int n, float dt,
    glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* box_index,
    int* index,
    int* begin_of_box, int* size_of_box, int B);
// LINGYIN: ���صؼ����µ��ٶȲ��ƶ� pos
__global__ void CalculateNewSpeedNaive(
    int n, glm::vec3* pos, glm::vec3* org_speed, glm::vec3* new_speed, int* index);

// LINGYIN: �����ٶ��ƶ� boids������û��
__global__ void MoveBoids(float dt, glm::vec3* pos, glm::vec3* vel);

