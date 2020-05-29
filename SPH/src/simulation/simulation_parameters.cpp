#include <simulation/simulation_parameters.h>
#include <iostream>
#include <math_constants.h>
#include <simulation/providers/cube_provider.h>
#include <simulation/providers/sphere_provider.h>

const float SimulationParameters::PARTICLE_MASS = 0.125f;

SimulationParameters& SimulationParameters::GetInstance()
{
    static SimulationParameters instance;

    static bool isInitialized = false;
    if (isInitialized)
    {
        return instance;
    }
    isInitialized = true;

    

    instance.g = 9.8f;
    instance.kernelRadius = 0.1f;
    instance.deltaTime = 0.016f;
    instance.restDensity = 1000.f;
    instance.relaxationParameter = 1000.f;
    instance.deltaQ = 0.3 * instance.kernelRadius;
    instance.correctionCoefficient = 0.0001f;
    instance.correctionPower = 4;
    instance.c_XSPH = 0.1f;
    instance.viscosityIterations = 4;
    instance.vorticityEpsilon = 0.0002f;
    instance.substepsNumber = 4;
    instance.change = true;
    instance.fluidStartPosition = make_float3(0.0f, 0.0f, 2.5f);
    instance.sizeInParticles = 30;

    instance.m_domainSize = SimulationDomainSize::Small;
    AdjustDomainToSize();
    instance.m_command = SimulationCommand::Unknown;
    instance.m_state = SimulationState::NotStarted;

    instance.m_source = ParticleSource::Cube;
    instance.m_particlesProvider = std::make_shared<SphereProvider>(make_float3(0.0f, 0.0f, 2.5f), 30);

    return instance;
}

SimulationParameters* SimulationParameters::GetInstancePtr()
{
    return &GetInstance();
}

void SimulationParameters::SetCommand(SimulationCommand command)
{
    auto& instance = GetInstance();
    switch (command)
    {
        case SimulationCommand::StepOneFrame:
        {
            if (instance.m_command == SimulationCommand::Run)
            {
                return;
            }
        }
        break;
        case SimulationCommand::Run:
        {
            if (instance.m_command == SimulationCommand::StepOneFrame)
            {
                return;
            }
        }
        break;
        case SimulationCommand::Unknown:
        case SimulationCommand::Pause:
        case SimulationCommand::Restart:
        {
            /* empty */
        }
        break;
        default:
        {
            std::cout << "No such command present" << std::endl;
        }
        break;
    }
    instance.m_command = command;
}

SimulationCommand SimulationParameters::GetCommand()
{
    return GetInstance().m_command;
}

void SimulationParameters::SetDomainSize(SimulationDomainSize domain)
{
    auto& instance = GetInstance();
    instance.m_domainSize = domain;
    AdjustDomainToSize();

    float3 up = instance.GetUpperBoundary();
    float3 low = instance.GetLowerBoundary();

    if (!instance.GetParticlesProvider().IsInsideBoundaries(up, low))
    {
        instance.fluidStartPosition = make_float3(0.0f, 0.0f, 2.5f);
        instance.GetParticlesProvider().SetPosition(instance.fluidStartPosition);
        instance.GetParticlesProvider().Provide();
    }
}

SimulationDomainSize SimulationParameters::GetDomainSize()
{
    return GetInstance().m_domainSize;
}

float3 SimulationParameters::GetUpperBoundary()
{
    return GetInstance().m_domain.upperBoundary;
}

float3 SimulationParameters::GetLowerBoundary()
{
    return GetInstance().m_domain.lowerBoundary;
}

SimulationDomain SimulationParameters::GetDomain()
{
    return GetInstance().m_domain;
}

float SimulationParameters::GetParticleRadius()
{
    auto& instance = GetInstance();
    float particleVolume = PARTICLE_MASS / instance.restDensity;
    float radius = std::powf((0.75f / CUDART_PI) * particleVolume, 1.0f / 3.0f);
    return radius;
}

SimulationState SimulationParameters::GetState()
{
    return GetInstance().m_state;
}

void SimulationParameters::SetState(SimulationState state)
{
    GetInstance().m_state = state;
}

IParticlesProvider& SimulationParameters::GetParticlesProvider()
{
    auto& instance = GetInstance();
    IParticlesProvider& provider = *(instance.m_particlesProvider);
    return provider;
}

bool SimulationParameters::SetStartPosition(float3 position)
{
    if (!m_particlesProvider->TrySetPosition(position))
    {
        return false;
    }

    fluidStartPosition = position;
    m_particlesProvider->Provide();
    return true;
}


bool SimulationParameters::SetStartX(float x)
{
    float3 position = fluidStartPosition;
    position.x = x;
    return SetStartPosition(position);
}

bool SimulationParameters::SetStartY(float y)
{
    float3 position = fluidStartPosition;
    position.y = y;
    return SetStartPosition(position);
}

bool SimulationParameters::SetStartZ(float z)
{
    float3 position = fluidStartPosition;
    position.z = z;
    return SetStartPosition(position);
}

void SimulationParameters::SetFluidSize(int size)
{
    if (!m_particlesProvider->TrySetSize(size))
    {
        return;
    }

    sizeInParticles = size;
    m_particlesProvider->Provide();
}

void SimulationParameters::SetParticlesSource(ParticleSource source)
{
    m_source = source;
    float3 position = m_particlesProvider->GetPosition();
    int sizeInParticles = m_particlesProvider->GetSize();
    switch (source)
    {
        case ParticleSource::Cube:
        {
            m_particlesProvider = std::make_shared<CubeProvider>(position, sizeInParticles);
        }
        break;
        case ParticleSource::Sphere:
        {
            m_particlesProvider = std::make_shared<SphereProvider>(position, sizeInParticles);
        }
        break;
    }
    SetCommand(SimulationCommand::Restart);
}


void SimulationParameters::UpdateStartPosition()
{
    m_particlesProvider->TrySetPosition(fluidStartPosition);
}

void SimulationParameters::AdjustDomainToSize()
{
    auto& instance = GetInstance();
    float2 upperXY = make_float2(1.0f, 1.0f);
    float2 lowerXY = make_float2(-1.0f, -1.0f);
    float upperZ = 4.0f;
    float lowerZ = 0.0f;

    switch (instance.m_domainSize)
    {
        case SimulationDomainSize::Small:
        {
            /* empty */
        }
        break;
        case SimulationDomainSize::Medium:
        {
            upperXY *= 1.5f;
            lowerXY *= 1.5f;
        }
        break;
        case SimulationDomainSize::Large:
        {
            upperXY *= 2.0f;
            lowerXY *= 2.0f;
        }
        break;
        case SimulationDomainSize::Stretched:
        {
            upperXY.y *= 1.5f;
            lowerXY.y *= 1.5f;
        }
        break;
    }

    instance.m_domain =
    {
        make_float3(upperXY.x, upperXY.y, upperZ),
        make_float3(lowerXY.x, lowerXY.y, lowerZ)
    };
}
