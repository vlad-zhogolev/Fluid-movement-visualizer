#include <simulation/providers/sphere_provider.h>
#include <simulation/simulation_parameters.h>
#include <helper.h>

#include <memory>
#include <iostream>

void SphereProvider::Provide()
{
    if (m_positionsBuffer == 0)
    {
        std::cout << "buffer is not set";
        return;
    }

    float radius = SimulationParameters::GetParticleRadius();
    //float step = 1.8f * radius;
    float sphereRadius = GetHalfEdge();
    float step = 2 * sphereRadius / m_sizeInParticles;
    int particleIndex = 0;

    //ReallocateIfNeeded(particlesNumber);
    float3 cubeLowerBoundary = m_cubeCenter - sphereRadius;
    float3 cubeUpperBoundary = m_cubeCenter + sphereRadius;

    m_positions.clear();
    m_velocities.clear();

    for (float i = 0; i < m_sizeInParticles; ++i)
    {
        float x = cubeLowerBoundary.x + step * i + step * 0.5;
        for (float j = 0; j < m_sizeInParticles; ++j)
        {
            float y = cubeLowerBoundary.y + step * j + step * 0.5;
            for (float k = 0; k < m_sizeInParticles; ++k)
            {
                float z = cubeLowerBoundary.z + step * k + step * 0.5;
                float r1 = 1.f * rand() / RAND_MAX, r2 = 1.f * rand() / RAND_MAX, r3 = 1.f * rand() / RAND_MAX;
                ++particleIndex;
                float3 position{ x, y, z };
                float distance = std::sqrtf(norm2(position - m_cubeCenter));
                if (distance > sphereRadius)
                {
                    continue;
                }
                m_positions.push_back(position + 0.1 * make_float3(r1, r2, r3));
                m_velocities.emplace_back(make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    int particlesNumber = GetParticlesNumber();
    glBindBuffer(GL_ARRAY_BUFFER, m_positionsBuffer);
    //glInvalidateBufferData(m_positionsBuffer);
    //glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename PositionsVector::value_type), m_positions.data());

    glBindBuffer(GL_ARRAY_BUFFER, m_velocitiesBuffer);
    //glInvalidateBufferData(m_velocitiesBuffer);
    //glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particlesNumber * sizeof(typename VelocitiesVector::value_type), m_velocities.data());
}

int SphereProvider::GetParticlesNumber()
{
    return m_positions.size();
}