#pragma once

#include <simulation/cube_provider.h>
#include <simulation/simulation_parameters.h>
#include <helper.h>
#include <stdexcept>
#include <iostream>


CubeProvider::CubeProvider(const float3& position, int sizeInParticles)
    : m_cubeCenter(position)
    , m_sizeInParticles(sizeInParticles)
    , m_edgeLength(CalculateEdgeLength(sizeInParticles))
{
    if (!IsInsideBoundaries(
            SimulationParameters::GetUpperBoundary(),
            SimulationParameters::GetLowerBoundary()))
    {
        throw std::logic_error("Provided particles won't fit to specified boundaries");
    }
}

void CubeProvider::SetTargets(GLuint positions, GLuint velocities)
{
    m_positionsBuffer = positions;
    m_velocitiesBuffer = velocities;
}

void CubeProvider::Provide()
{    if (m_positionsBuffer == 0)
    {
        std::cout << "buffer is not set";
        return;
    }

    float radius = SimulationParameters::GetParticleRadius();
    float step = 2.0f * radius;
    float halfEdgeLength = GetHalfEdge();
    int particleIndex = 0;

    int particlesNumber = m_sizeInParticles * m_sizeInParticles * m_sizeInParticles;
    ReallocateIfNeeded(particlesNumber);
    float3 cubeLowerBoundary = m_cubeCenter - halfEdgeLength;
    float3 cubeUpperBoundary = m_cubeCenter + halfEdgeLength;

    m_positions.clear();
    m_velocities.clear();

    for (float i = 0; i < m_sizeInParticles; ++i)
    {
        float x = cubeLowerBoundary.x + step * i;
        for (float j = 0; j < m_sizeInParticles; ++j)
        {
            float y = cubeLowerBoundary.y + step * j;
            for (float k = 0; k < m_sizeInParticles; ++k)
            {
                float z = cubeLowerBoundary.z + step * k;
                float r1 = 1.f * rand() / RAND_MAX, r2 = 1.f * rand() / RAND_MAX, r3 = 1.f * rand() / RAND_MAX;
                ++particleIndex;
                m_positions.push_back(make_float3(x, y, z) + 0.1 * make_float3(r1, r2, r3));
                m_velocities.emplace_back(make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_positionsBuffer);
    //glInvalidateBufferData(m_positionsBuffer);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename PositionsVector::value_type), m_positions.data());

    glBindBuffer(GL_ARRAY_BUFFER, m_velocitiesBuffer);
    //glInvalidateBufferData(m_velocitiesBuffer);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename VelocitiesVector::value_type), m_velocities.data());
}

int CubeProvider::GetParticlesNumber()
{
    return m_sizeInParticles * m_sizeInParticles * m_sizeInParticles;
}

bool CubeProvider::TrySetPosition(const float3& position)
{
    if (!IsInsideBoundaries(position, m_edgeLength,
            SimulationParameters::GetUpperBoundary(), SimulationParameters::GetLowerBoundary()))
    {
        return false;
    }

    SetPosition(position);
}

bool CubeProvider::SetPosition(const float3& position)
{
    m_cubeCenter = position;
    return true;
}

bool CubeProvider::TrySetSize(int particlesNumber)
{
    float edgeLength = CalculateEdgeLength(particlesNumber);
    if (!IsInsideBoundaries(m_cubeCenter, edgeLength,
            SimulationParameters::GetUpperBoundary(), SimulationParameters::GetLowerBoundary()))
    {
        return false;
    }

    m_sizeInParticles = particlesNumber;
    m_edgeLength = edgeLength;
    return true;
}

bool CubeProvider::SetDensity(float density)
{
    if (!IsInsideBoundaries(m_cubeCenter, CalculateEdgeLength(m_sizeInParticles),
        SimulationParameters::GetUpperBoundary(), SimulationParameters::GetLowerBoundary()))
    {
        return false;
    }

    m_density = density;
    return true;
}

bool CubeProvider::IsInsideBoundaries(const float3& upperBoundary, const float3& lowerBoundary)
{  
    return IsInsideBoundaries(m_cubeCenter, m_edgeLength, upperBoundary, lowerBoundary);
}

void CubeProvider::ReallocateIfNeeded(int particlesNumber)
{
    m_positions.reserve(particlesNumber);
    m_velocities.reserve(particlesNumber);
}

bool CubeProvider::IsInsideBoundaries(float3 center, float edgeLength, const float3& upperBoundary, const float3& lowerBoundary)
{
    float halfEdge = 0.5f * edgeLength;
    bool result =
        center.x + halfEdge < upperBoundary.x &&
        center.y + halfEdge < upperBoundary.y &&
        center.z + halfEdge < upperBoundary.z &&

        center.x - halfEdge > lowerBoundary.x &&
        center.y - halfEdge > lowerBoundary.y &&
        center.z - halfEdge > lowerBoundary.z;

    return result;
}

float CubeProvider::CalculateEdgeLength(float sizeInParticles)
{
    float particleDiameter = 2 * SimulationParameters::GetParticleRadius();
    float edgeLength = sizeInParticles * particleDiameter;
    return edgeLength;
}
