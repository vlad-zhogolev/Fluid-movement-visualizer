#pragma once

#include <simulation/cube_provider.h>
#include <simulation/simulation_parameters.h>
#include <stdexcept>

CubeProvider::CubeProvider(const float3& position, float size, const float3& upperBoundary, const float3& lowerBoundary)
    : m_cubeCenter(position)
    , m_edgeLength(size)
{
    if (!IsInsideBoundaries(upperBoundary, lowerBoundary))
    {
        throw std::logic_error("Provided particles won't fit to specified boundaries");
    }
}

void CubeProvider::SetTargets(GLuint positions, GLuint velocities)
{
    m_positionsBuffer = positions;
    m_velocitiesBuffer = velocities;
}

void CubeProvider::Provide(int& outParticlesNumber)
{
    float radius = SimulationParameters::GetParticleRadius();
    float step = m_edgeLength / radius;
    float halfEdgeLength = GetHalfEdge();
    int particleIndex = 0;

    int particlesNumber = m_sizeInParticles * m_sizeInParticles * m_sizeInParticles;
    ReallocateIfNeeded(particlesNumber);
    for (float x = m_cubeCenter.x - halfEdgeLength; x <= m_cubeCenter.x + halfEdgeLength; x += step)
    {
        for (float y = m_cubeCenter.y - halfEdgeLength; y <= m_cubeCenter.y + halfEdgeLength; y += step)
        {
            for (float z = m_cubeCenter.z - halfEdgeLength; z <= m_cubeCenter.z + halfEdgeLength; z += step)
            {
                ++particleIndex;
                m_positions[particleIndex] = make_float3(x, y, z);
                m_velocities[particleIndex] = make_float3(x, y, z);
            }
        }
    }

    outParticlesNumber = particlesNumber;
    glBindBuffer(GL_ARRAY_BUFFER, m_positionsBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename PositionsVector::value_type), m_positions.data());
    glBindBuffer(GL_ARRAY_BUFFER, m_velocitiesBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename VelocitiesVector::value_type), m_velocities.data());
}

bool CubeProvider::SetPosition(const float3& position)
{
    if (!IsInsideBoundaries(position, m_edgeLength,
            SimulationParameters::GetUpperBoundary(), SimulationParameters::GetLowerBoundary()))
    {
        return false;
    }

    m_cubeCenter = position;
    return true;
}

bool CubeProvider::SetSize(int particlesNumber)
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
    return IsInsideBoundaries(m_cubeCenter, GetHalfEdge(), upperBoundary, lowerBoundary);
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
