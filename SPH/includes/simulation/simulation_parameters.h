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
    static const int FLUID_SIZE_MIN;
    static const int FLUID_SIZE_MAX;
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

    void SetFluidSize(int size);
    inline int GetFluidSize() const { return sizeInParticles; }

    void SetParticlesSource(ParticleSource source);

    void SetGravityX(float gravityX);
    inline float GetGravityX() const { return gravity.x; }

    void SetGravityY(float gravityY);
    inline float GetGravityY() const { return gravity.y; }

    void SetGravityZ(float gravityZ);
    inline float GetGravityZ() const { return gravity.z; }

    void SetSubstepsNumber(int substepsNumber);
    inline int GetSubstepsNumber() const { return substepsNumber; }

    void SetDeltaTime(float time);
    inline float GetDeltaTime() const { return deltaTime; }

    void SetDensity(float density);
    inline float GetDensity() const { return restDensity; }

    void SetKernelRadius(float radius);
    inline float GetKernelRadius() const { return kernelRadius; }

    void SetLambdaEpsilon(float value);
    inline float GetLambdaEpsilon() const { return relaxationParameter; }

    void SetDeltaQ(float value);
    inline float GetDeltaQ() const { return deltaQ; }

    void SetCorrectionCoefficient(float value);
    inline float GetCorrectionCoefficient() const { return correctionCoefficient; }

    void SetCorrectionPower(float value);
    inline float GetCorrectionPower() const { return correctionPower; }

    void SetXSPHCoefficient(float value);
    inline float GetXSPHCoefficient() const { return c_XSPH; }

    void SetViscosityIter(int value);
    inline int GetViscosityIter() const { return viscosityIterations; }

    void SetVorticity(float value);
    inline float GetVorticity() const { return vorticityEpsilon; }

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