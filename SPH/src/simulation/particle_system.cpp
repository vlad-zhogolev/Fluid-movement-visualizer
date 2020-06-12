#include <simulation/particle_system.h>
#include <simulation/simulation_parameters.h>
#include <glm/common.hpp>
#include <cuda_gl_interop.h>

ParticleSystem::ParticleSystem()
{
    //m_upperBoundary = make_float3(1.f, 1.f, 4.f);
    //m_lowerBoundary = make_float3(-1.f, -1.f, 0.f);

    //m_lowerBoundary = make_float3(0.f, 0.f, 0.f);

    m_upperBoundary = SimulationParameters::GetUpperBoundary();
    m_lowerBoundary = SimulationParameters::GetLowerBoundary();

    m_simulator = new PositionBasedFluidSimulator(m_upperBoundary, m_lowerBoundary);

    // Particle positions and velocities
    glGenBuffers(1, &m_positions1);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions1);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLES_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_positions2);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions2);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_velocities1);
    glBindBuffer(GL_ARRAY_BUFFER, m_velocities1);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLES_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_velocities2);
    glBindBuffer(GL_ARRAY_BUFFER, m_velocities2);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES_NUM * sizeof(float3), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    glGenBuffers(1, &m_particleIndices);
    glBindBuffer(GL_ARRAY_BUFFER, m_particleIndices);
    glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLES_NUM * sizeof(unsigned int), nullptr, GL_STATIC_DRAW);
    checkGLErr();

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_positionsResource1, m_positions1, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_positionsResource2, m_positions2, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_velocitiesResource1, m_velocities1, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_velocitiesResource2, m_velocities2, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_indicesResource, m_particleIndices, cudaGraphicsMapFlagsNone));
}

void ParticleSystem::InitializeParticles() 
{
    //m_source->initialize(m_positions1, m_velocities1, m_particleIndices, MAX_PARTICLES_NUM);
    auto& provider = SimulationParameters::GetParticlesProvider();
    provider.SetTargets(m_positions1, m_velocities1);
    provider.Provide();
    m_particlesNumber = provider.GetParticlesNumber();
}

void ParticleSystem::PerformSimulationStep() 
{
    auto& input = Input::GetInstance();
    //if (!(input.running || input.nextFrame))
    //{
    //    return;
    //}

    if (SimulationParameters::GetState() == SimulationState::NotStarted)
    {
        auto& provider = SimulationParameters::GetParticlesProvider();
        provider.SetTargets(m_positions1, m_velocities1);
    }
    auto command = m_simulationParams->GetCommand();
    m_particlesNumber = m_simulationParams->GetParticlesProvider().GetParticlesNumber();
    if (command == SimulationCommand::Restart)
    {
        InitializeParticles();
        m_isSecondParticlesUsedForRendering = false;
        m_simulationParams->SetCommand(SimulationCommand::Unknown);
        m_simulationParams->SetState(SimulationState::NotStarted);
        return;
    }
    if (command != SimulationCommand::StepOneFrame &&
        command != SimulationCommand::Run)
    {
        return;
    }
    else if (m_simulationParams->GetState() == SimulationState::NotStarted)
    {
        // TODO: check that source is inside the box
        m_simulationParams->SetState(SimulationState::Started);
    }
    if (command == SimulationCommand::StepOneFrame)
    {
        m_simulationParams->SetCommand(SimulationCommand::Unknown);
    }
    
    ++input.frameCount;
    m_simulator->UpdateParameters();

    if (m_isSecondParticlesUsedForRendering)
    {
        m_simulator->Step(m_positionsResource2, m_positionsResource1,
            m_velocitiesResource2, m_velocitiesResource1, m_indicesResource, m_particlesNumber);
    }
    else
    {
        m_simulator->Step(m_positionsResource1, m_positionsResource2,
            m_velocitiesResource1, m_velocitiesResource2,  m_indicesResource, m_particlesNumber);
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
    checkCudaErrors(cudaGraphicsUnregisterResource(m_positionsResource1));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_positionsResource2));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_velocitiesResource1));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_velocitiesResource2));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_indicesResource));

    if (m_simulator)
    {
        delete m_simulator;
    }
}
