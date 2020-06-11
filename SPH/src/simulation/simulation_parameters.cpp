#include <simulation/simulation_parameters.h>
#include <iostream>
#include <algorithm>
#include <math_constants.h>
#include <simulation/providers/cube_provider.h>
#include <simulation/providers/sphere_provider.h>

const float SimulationParameters::PARTICLE_MASS = 0.125f;
const float SimulationParameters::GRAVITY_MIN = -20.0f;
const float SimulationParameters::GRAVITY_MAX = 20.0f;
const int SimulationParameters::SUBSTEPS_NUMBER_MIN = 1;
const int SimulationParameters::SUBSTEPS_NUMBER_MAX = 10;
const float SimulationParameters::KERNEL_RADIUS_MIN = 0.05f;
const float SimulationParameters::KERNEL_RADIUS_MAX = 1.0f;
const float SimulationParameters::DENSITY_MIN = 50.0f;
const float SimulationParameters::DENSITY_MAX = 10000.0f;
const float SimulationParameters::DELTA_TIME_MIN = 0.0f;
const float SimulationParameters::DELTA_TIME_MAX = 0.002f;
const float SimulationParameters::RELAXATION_PARAM_MIN = 0.0f;
const float SimulationParameters::RELAXATION_PARAM_MAX = 100000.0f;
const float SimulationParameters::DELTA_Q_MIN = 0.1f;
const float SimulationParameters::DELTA_Q_MAX = 0.3f;
const float SimulationParameters::CORRECTION_COEF_MIN = 0.0f;
const float SimulationParameters::CORRECTION_COEF_MAX = 0.1f;
const float SimulationParameters::CORRECTION_POWER_MIN = 0.0f;
const float SimulationParameters::CORRECTION_POWER_MAX = 10.0f;
const float SimulationParameters::XSPH_COEF_MIN = 0.0f;
const float SimulationParameters::XSPH_COEF_MAX = 10.0f;
const int SimulationParameters::XSPH_ITERATIONS_MIN = 1;
const int SimulationParameters::XSPH_ITERATIONS_MAX = 10;
const float SimulationParameters::VORTICITY_MIN = 0.0f;
const float SimulationParameters::VORTICITY_MAX = 1.0f;


SimulationParameters& SimulationParameters::GetInstance()
{
    static SimulationParameters instance;

    static bool isInitialized = false;
    if (isInitialized)
    {
        return instance;
    }
    isInitialized = true;

    instance.gravity = make_float3(0.0f, 0.0f, -9.8f);
    instance.kernelRadius = 0.1f;
    instance.deltaTime = 0.016f;
    instance.startDensity = 1000.f;
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
    instance.m_particlesProvider = std::make_shared<CubeProvider>(make_float3(0.0f, 0.0f, 2.5f), 30);

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
        case SimulationCommand::Restart:
        {
            auto& provider = instance.GetParticlesProvider();
            float3 up = instance.GetUpperBoundary();
            float3 low = instance.GetLowerBoundary();
            if (!provider.TrySetDensity(instance.restDensity))
            {
                instance.restDensity = instance.startDensity;
            }
            else
            {
                //instance.startDensity = instance.restDensity;
                instance.SetState(SimulationState::NotStarted);
                instance.SetDensity(instance.restDensity);
            }
        }
        break;
        case SimulationCommand::Unknown:
        case SimulationCommand::Pause:
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

    auto& provider = instance.GetParticlesProvider();
    if (!provider.IsInsideBoundaries(up, low))
    {
        instance.fluidStartPosition = make_float3(0.0f, 0.0f, 2.5f);
        instance.GetParticlesProvider().SetPosition(instance.fluidStartPosition);

        if (!provider.IsInsideBoundaries(up, low))
        {
            float diameter = 2 * GetParticleRadius();
            int3 size = make_int3((up - low) / diameter);
            instance.sizeInParticles = std::min({ size.x, size.y, size.z });
            provider.SetSize(instance.sizeInParticles);
        }

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
    return instance.GetParticleRadius(instance.restDensity);
}

float SimulationParameters::GetParticleRadius(float density) const
{
    float particleVolume = PARTICLE_MASS / density;
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

void SimulationParameters::SetDensity(float density)
{
    auto& instance = GetInstance();
    switch (instance.m_state)
    {
        case SimulationState::NotStarted:
        {
            auto& provider = instance.GetParticlesProvider();

            //float tmp = instance.restDensity;
            //instance.restDensity = density;
            if (!provider.TrySetDensity(density))
            {
                return;
            }

            instance.startDensity = density;
            instance.restDensity = density;
            provider.Provide();
        }
        break;
        case SimulationState::Started:
        {
            instance.restDensity = density;
        }
        break;
    }
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
