#include <simulation/providers/sphere_provider.h>
#include <simulation/simulation_parameters.h>
#include <utils.h>

#include <memory>
#include <iostream>
#include <random>

void SphereProvider::Provide()
{
    if (m_positionsBuffer == 0)
    {
        std::cout << "buffer is not set";
        return;
    }

    m_positions.clear();
    m_velocities.clear();

    float sphereRadius = GetHalfEdge();
    float3 cubeLowerBoundary = m_cubeCenter - sphereRadius;
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
                float distanceFromCenter = std::sqrtf(norm2(position - m_cubeCenter));
                if (distanceFromCenter >= sphereRadius)
                {
                    continue;
                }
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

int SphereProvider::GetParticlesNumber()
{
    return m_positions.size();
}