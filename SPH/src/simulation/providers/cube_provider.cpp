#pragma once

#include <simulation/providers/cube_provider.h>
#include <simulation/simulation_parameters.h>
#include <utils.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>


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
{   
    if (m_positionsBuffer == 0)
    {
        std::cout << "buffer is not set";
        return;
    }

    m_positions.clear();
    m_velocities.clear();

    float halfEdgeLength = GetHalfEdge();
    float3 cubeLowerBoundary = m_cubeCenter - halfEdgeLength;
    float diameter = 2.0f * SimulationParameters::GetParticleRadius();

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_real_distribution<> distribution(0.0f, 0.4f * diameter);

    float3 position = cubeLowerBoundary;
    for (float i = 0; i < m_sizeInParticles; ++i)
    {
        position.x += diameter;
        for (float j = 0; j < m_sizeInParticles; ++j)
        {
            position.y += diameter;
            for (float k = 0; k < m_sizeInParticles; ++k)
            {
                position.z += diameter;
                float3 offset = make_float3(distribution(generator), distribution(generator), distribution(generator));
                m_positions.push_back(position + offset);
            }
            position.z = cubeLowerBoundary.z;
        }
        position.y = cubeLowerBoundary.y;
    }
    m_velocities.assign(m_positions.size(), float3{ 0.0f, 0.0f, 0.0f });

    UpdateTargetBuffersData();
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

bool CubeProvider::SetSize(int particlesNumber)
{
    m_sizeInParticles = particlesNumber;
    m_edgeLength = CalculateEdgeLength(particlesNumber);
    return true;
}

bool CubeProvider::TrySetDensity(float density)
{
    float edgeLength = CalculateEdgeLength(m_sizeInParticles, density);
    if (!IsInsideBoundaries(m_cubeCenter, edgeLength,
        SimulationParameters::GetUpperBoundary(), SimulationParameters::GetLowerBoundary()))
    {
        return false;
    }

    m_edgeLength = edgeLength;
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

float CubeProvider::CalculateEdgeLength(float sizeInParticles) const
{
    float particleDiameter = 2 * SimulationParameters::GetParticleRadius();
    float edgeLength = sizeInParticles * particleDiameter;
    return edgeLength;
}

float CubeProvider::CalculateEdgeLength(float sizeInParticles, float density) const
{
    auto& instance = SimulationParameters::GetInstance();
    float particleDiameter = 2 * instance.GetParticleRadius(density);
    float edgeLength = sizeInParticles * particleDiameter;
    return edgeLength;
}

void CubeProvider::UpdateTargetBuffersData()
{
    int particlesNumber = GetParticlesNumber();
    glBindBuffer(GL_ARRAY_BUFFER, m_positionsBuffer);
    //glInvalidateBufferData(m_positionsBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename PositionsVector::value_type), m_positions.data());

    glBindBuffer(GL_ARRAY_BUFFER, m_velocitiesBuffer);
    //glInvalidateBufferData(m_velocitiesBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename VelocitiesVector::value_type), m_velocities.data());
}
