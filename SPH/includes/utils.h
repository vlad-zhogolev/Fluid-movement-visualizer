#pragma once
#include <helper_math.h>

#define M_PI 3.14159265359
#define LIM_EPS 1e-3

#define KERNEL_EPSILON 1e-4

#define MAX_DELTA_POSITION 0.1
const int MAX_PARTICLES_NUM = 150000;

__host__ __device__
inline int ceilDiv(int a, int b)
{
    return (int)((a + b - 1) / b);
}

__host__ __device__
inline float norm2(float3 u)
{
    return u.x * u.x + u.y * u.y + u.z * u.z;
}

void fexit(const int code = -1, const char* msg = nullptr);

void checkGLErr();
void checkFramebufferComplete();