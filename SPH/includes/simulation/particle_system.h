#pragma once

#include <GLFW/glfw3.h>
#include <simulation/position_based_fluid_simulator.h>
#include <rendering/renderer.h>

class ParticleSystem
{
public:
    ParticleSystem();
    ~ParticleSystem();

    void InitializeParticles();
    void PerformSimulationStep();
    void UpdateParticles();

    GLuint GetPositionsForRenderingHandle() const;
    GLuint GetIndicesHandle() const { return m_particleIndices; }
    int GetParticleNumber() const { return m_particlesNumber; }
    float3 GetUpperLimit() const { return m_upperBoundary; }
    float3 GetLowerLimit() const { return m_lowerBoundary; }

private:
    GLuint m_positions1;
    GLuint m_positions2;
    GLuint m_velocities1;
    GLuint m_velocities2;
    GLuint m_particleIndices;

    struct cudaGraphicsResource* m_positionsResource1;
    struct cudaGraphicsResource* m_positionsResource2;
    struct cudaGraphicsResource* m_velocitiesResource1;
    struct cudaGraphicsResource* m_velocitiesResource2;
    struct cudaGraphicsResource* m_indicesResource;

    int m_particlesNumber;
    bool m_isSecondParticlesUsedForRendering = false;

    PositionBasedFluidSimulator* m_simulator;

    float3 m_upperBoundary;
    float3 m_lowerBoundary;

    SimulationParameters* m_simulationParams = SimulationParameters::GetInstancePtr();
};

