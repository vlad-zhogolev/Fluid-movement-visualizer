#pragma once

#include <simulation/i_particles_provider.h>
#include <vector>

class CubeProvider : IParticlesProvider
{
public:
    CubeProvider(const glm::vec3& position, float size, const float3& upperBoundary, const float3& lowerBoundary);

    // IParticlesProvider
    void Provide(int& particlesNumber) override;
    bool SetPosition(const glm::vec3& position) override;
    bool SetSize(int particlesNumber) override;
    bool SetDensity(float density) override;
    bool IsInsideBoundaries(const float3& upperBoundary, const float3& lowerBoundary) override;

private:
    inline float GetHalfEdge() const { return m_edgeLength / 2.0f; }
    inline int GetParticlesNumber() { return m_positions.size(); }
    void ReallocateIfNeeded(int particlesNumber);

    bool IsInsideBoundaries(float center, float edgeLength, const float3& upperBoundary, const float3& lowerBoundary);

private:
    glm::vec3 m_cubeCenter;
    int m_sizeInParticles;
    float m_edgeLength;
    float m_density;

    std::vector<float3> m_positions;
    std::vector<float3> m_velocities;
};