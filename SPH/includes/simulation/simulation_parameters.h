#pragma once

#include <helper_math.h>

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

class SimulationParameters
{
public:

    // Particle system
    int substepsNumber;
    float restDensity;
    float g;
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
    float size;
    
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

    static SimulationState GetState();
    static void SetState(SimulationState state);

private:
    static void AdjustDomainToSize();

private:
    SimulationDomain m_domain;
    SimulationDomainSize m_domainSize;
    SimulationCommand m_command;
    SimulationState m_state;

    float m_upperBoundary;
    float m_lowerBoundary;
};