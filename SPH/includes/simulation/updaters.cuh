#pragma once

#include <utils.h>
#include <helper_math.h>
#include <thrust/tuple.h>

struct PositionUpdater
{
public:
    __host__ __device__
    explicit PositionUpdater(float3 upperBoundary, float3 lowerBoundary)
        : m_upperBoundary(upperBoundary)
        , m_lowerBoundary(lowerBoundary) {}

    __host__ __device__ __forceinline__
    float3 operator()(const thrust::tuple<float3, float3>& tuple) const
    {
        float3 position = thrust::get<0>(tuple);
        float3 deltaPosition = thrust::get<1>(tuple);
        float3 newPosition = clamp(position + deltaPosition, m_lowerBoundary + LIM_EPS, m_upperBoundary - LIM_EPS);

        return newPosition;
    }

private:
    float3 m_upperBoundary;
    float3 m_lowerBoundary;
};

struct VelocityUpdater
{
public:
    __host__ __device__
    explicit VelocityUpdater(float deltaTime) : m_deltaTimeInverse(1.f / deltaTime) {}

    __host__ __device__ __forceinline__
    float3 operator()(const thrust::tuple<float3, float3>& tuple) const
    {
        float3 pos = thrust::get<0>(tuple);
        float3 npos = thrust::get<1>(tuple);
        float3 newVelocity = (npos - pos) * m_deltaTimeInverse;

        return newVelocity;
    }

private:
    float m_deltaTimeInverse;
};
