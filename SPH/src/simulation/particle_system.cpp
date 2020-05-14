#include <simulation/particle_system.h>
#include <simulation/particles_cube.h>
#include <simulation/simulation_parameters.h>
#include <glm/common.hpp>

ParticleSystem::ParticleSystem()
{
    m_upperBoundary = make_float3(1.f, 1.f, 4.f);
    
    m_lowerBoundary = make_float3(-1.f, -1.f, 0.f);
    //m_lowerBoundary = make_float3(0.f, 0.f, 0.f);

    m_simulator = new PositionBasedFluidSimulator(m_upperBoundary, m_lowerBoundary);

    // Initialize particles
    float dd = 1.f / 20;
    float d1 = dd * 30, d2 = dd * 30, d3 = dd * 30;
    const int particlesInDimension = 30;
    const float upperBoundary = 0.75f;
    m_source = new ParticlesCube(
        make_float3(upperBoundary, upperBoundary, 3.8f), // upper boundary
        make_float3(upperBoundary - d1, upperBoundary - d2, 3.8f - d3), // lower boundary
        make_int3(particlesInDimension)); // particles number by dimensions
    m_particlesNumber = particlesInDimension * particlesInDimension * particlesInDimension;

    // Particle positions and velocities
    glGenBuffers(1, &m_positions1);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions1);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_positions2);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions2);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_velocities1);
    glBindBuffer(GL_ARRAY_BUFFER, m_velocities1);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_velocities2);
    glBindBuffer(GL_ARRAY_BUFFER, m_velocities2);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_particleIndices);
    glBindBuffer(GL_ARRAY_BUFFER, m_particleIndices);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(unsigned int), nullptr, GL_STATIC_DRAW);
    checkGLErr();
}

void ParticleSystem::InitializeParticles() 
{
    m_source->initialize(m_positions1, m_velocities1, m_particleIndices, MAX_PARTICLE_NUM);
}

void ParticleSystem::PerformSimulationStep() 
{
    auto& input = Input::getInstance();
    if (!(input.running || input.nextFrame))
    {
        return;
    }
    input.nextFrame = false;
    ++input.frameCount;

    m_simulator->UpdateParameters();

    if (m_isSecondParticlesUsedForRendering)
    {
        m_simulator->Step(m_positions2, m_positions1, m_velocities2, m_velocities1, m_particleIndices, m_particlesNumber);
    }
    else
    {
        m_simulator->Step(m_positions1, m_positions2, m_velocities1, m_velocities2, m_particleIndices, m_particlesNumber);
    }
    m_isSecondParticlesUsedForRendering = !m_isSecondParticlesUsedForRendering;
}

GLuint ParticleSystem::GetPositionsForRenderingHandle() const
{
    if (m_isSecondParticlesUsedForRendering)
    {
        return m_positions2;
    }
    else
    {
        return m_positions1;
    }
}

ParticleSystem::~ParticleSystem()
{
    if (m_simulator)
    {
        delete m_simulator;
    }
    if (m_source)
    {
        delete m_source;
    }
}
