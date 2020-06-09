#pragma once

#include <helper_math.h>
#include <simulation/providers/i_particles_provider.h>
#include <memory>

struct SimulationDomain
{
    float3 upperBoundary;
    float3 lowerBoundary;
};

enum class SimulationDomainSize
{
    Small,
    Medium,
    Large,
    Stretched
};

enum class SimulationCommand
{
    Unknown,
    StepOneFrame,
    Run,
    Pause,
    Restart
};

enum class SimulationState
{
    NotStarted,
    Started
};

enum class ParticleSource
{
    Cube,
    Sphere
};

class SimulationParameters
{
public:

    static const float PARTICLE_MASS;
    static const float GRAVITY_MIN;
    static const float GRAVITY_MAX;
    static const int SUBSTEPS_NUMBER_MIN;
    static const int SUBSTEPS_NUMBER_MAX;
    static const float KERNEL_RADIUS_MIN;
    static const float KERNEL_RADIUS_MAX;
    static const float DENSITY_MIN;
    static const float DENSITY_MAX;
    static const float DELTA_TIME_MIN;
    static const float DELTA_TIME_MAX;
    static const float RELAXATION_PARAM_MIN;
    static const float RELAXATION_PARAM_MAX;
    static const float DELTA_Q_MIN;
    static const float DELTA_Q_MAX;
    static const float CORRECTION_COEF_MIN;
    static const float CORRECTION_COEF_MAX;
    static const float CORRECTION_POWER_MIN;
    static const float CORRECTION_POWER_MAX;
    static const float XSPH_COEF_MIN;
    static const float XSPH_COEF_MAX;
    static const int XSPH_ITERATIONS_MIN;
    static const int XSPH_ITERATIONS_MAX;
    static const float VORTICITY_MIN;
    static const float VORTICITY_MAX;
    
    // Particle system
    int substepsNumber;
    float startDensity;
    float restDensity;
    float3 gravity;
    float kernelRadius;
    float deltaTime;
    float relaxationParameter;
    float deltaQ;
    float correctionCoefficient;
    float correctionPower;
    float c_XSPH;
    int viscosityIterations;
    float vorticityEpsilon;

    bool change;
    float3 fluidStartPosition;
    int sizeInParticles;
    
    static SimulationParameters& GetInstance();
    static SimulationParameters* GetInstancePtr();

    static void SetCommand(SimulationCommand command);
    static SimulationCommand GetCommand();

    static void SetDomainSize(SimulationDomainSize domain);
    static SimulationDomainSize GetDomainSize();

    static float3 GetUpperBoundary();
    static float3 GetLowerBoundary();
    static SimulationDomain GetDomain();

    static float GetParticleRadius();
    float GetParticleRadius(float density) const;

    static SimulationState GetState();
    static void SetState(SimulationState state);

    static IParticlesProvider& GetParticlesProvider();

    bool SetStartPosition(float3 position);
    bool SetStartX(float x);
    bool SetStartY(float y);
    bool SetStartZ(float z);

    inline int GetFluidSize() const { return sizeInParticles; }
    void SetFluidSize(int size);

    void SetParticlesSource(ParticleSource source);

    void SetDensity(float density);
    inline float GetDensity() const { return GetInstance().restDensity; }

private:
    static void AdjustDomainToSize();

    void UpdateStartPosition();

private:
    SimulationDomain m_domain;
    SimulationDomainSize m_domainSize;
    SimulationCommand m_command;
    SimulationState m_state;
    ParticleSource m_source;

    std::shared_ptr<IParticlesProvider> m_particlesProvider;

    float m_upperBoundary;
    float m_lowerBoundary;
};