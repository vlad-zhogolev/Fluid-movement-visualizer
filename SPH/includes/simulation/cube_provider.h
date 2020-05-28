#pragma once

#include <simulation/i_particles_provider.h>
#include <vector>

class CubeProvider : IParticlesProvider
{

    using PositionsVector = std::vector<float3>;
    using VelocitiesVector = std::vector<float3>;

public:
    CubeProvider(const float3& position, float size, const float3& upperBoundary, const float3& lowerBoundary);

    // IParticlesProvider
    void SetTargets(GLuint positions, GLuint velocities) override;
    void Provide(int& particlesNumber) override;
    bool SetPosition(const float3& position) override;
    bool SetSize(int particlesNumber) override;
    bool SetDensity(float density) override;
    bool IsInsideBoundaries(const float3& upperBoundary, const float3& lowerBoundary) override;

private:
    inline float GetHalfEdge() const { return m_edgeLength / 2.0f; }
    inline int GetParticlesNumber() { return m_positions.size(); }
    void ReallocateIfNeeded(int particlesNumber);

    bool IsInsideBoundaries(float3 center, float edgeLength, const float3& upperBoundary, const float3& lowerBoundary);
    float CalculateEdgeLength(float sizeInParticles);

private:
    float3 m_cubeCenter;
    int m_sizeInParticles;
    float m_edgeLength;
    float m_density;

    GLuint m_positionsBuffer;
    GLuint m_velocitiesBuffer;

    PositionsVector m_positions;
    VelocitiesVector m_velocities;
};