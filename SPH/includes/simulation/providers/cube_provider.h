#pragma once

#include <simulation/providers/i_particles_provider.h>
#include <vector>

class CubeProvider : public IParticlesProvider
{

    using PositionsVector = std::vector<float3>;
    using VelocitiesVector = std::vector<float3>;

public:
    explicit CubeProvider(const float3& position, int sizeInParticles);

    // IParticlesProvider
    void SetTargets(GLuint positions, GLuint velocities) override;
    void Provide() override;
    int GetParticlesNumber() override;
    bool TrySetPosition(const float3& position) override;
    bool SetPosition(const float3& position) override;
    inline float3 GetPosition() const override { return m_cubeCenter; }
    bool TrySetSize(int particlesNumber) override;
    bool SetSize(int particlesNumber) override;
    inline int GetSize() const override { return m_sizeInParticles; }
    bool TrySetDensity(float density) override;
    bool IsInsideBoundaries(const float3& upperBoundary, const float3& lowerBoundary) override;

protected:
    inline float GetHalfEdge() const { return 0.5f * m_edgeLength; }
    //inline int GetParticlesNumber() { return m_positions.size(); }
    void ReallocateIfNeeded(int particlesNumber);

    bool IsInsideBoundaries(float3 center, float edgeLength, const float3& upperBoundary, const float3& lowerBoundary);
    float CalculateEdgeLength(float sizeInParticles) const;
    float CalculateEdgeLength(float sizeInParticles, float density) const;
    void UpdateTargetBuffersData();

protected:
    float3 m_cubeCenter;
    int m_sizeInParticles;
    float m_edgeLength;
    float m_density;

    GLuint m_positionsBuffer;
    GLuint m_velocitiesBuffer;

    PositionsVector m_positions;
    VelocitiesVector m_velocities;
};