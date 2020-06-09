#pragma once

#include <utils.h>
#include <math_constants.h>

struct Poly6Kernel
{
public:
    __host__ __device__
    explicit Poly6Kernel(float kernelRadius)
        : m_kernelRadius2(kernelRadius * kernelRadius)
        , m_coefficient(315.f * powf(1.f / kernelRadius, 9) / (64.f * CUDART_PI_F)){}

    __host__ __device__ __forceinline__
        float operator()(float squaredDistance) const
    {
        if (squaredDistance >= m_kernelRadius2)
        {
            return 0;
        }
        float difference = m_kernelRadius2 - squaredDistance;
        return 0.125f * m_coefficient * difference * difference * difference;
    }

private:
    float m_coefficient;
    float m_kernelRadius2;
};

struct SpikyGradientKernel
{
public:
    __host__ __device__
    explicit SpikyGradientKernel(float kernelRadius)
        : m_kernelRadius(kernelRadius)
        , m_coefficient(-45.f / (CUDART_PI_F * powf(kernelRadius, 6))){}

    __host__ __device__ __forceinline__
    float3 operator()(float3 vector) const
    {
        float vectorLength = length(vector);
        if (vectorLength < KERNEL_EPSILON || vectorLength >= m_kernelRadius)
        {
            return float3{};
        }
        float difference = m_kernelRadius - vectorLength;
        return 0.125f * m_coefficient * difference * difference / vectorLength * vector;
    }

private:
    float m_kernelRadius;
    float m_coefficient;
};