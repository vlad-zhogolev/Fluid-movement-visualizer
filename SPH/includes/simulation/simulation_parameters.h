#pragma once


enum class SimulationDomain
{
    Small,
    Medium,
    Large
};

enum class SimulationCommand
{
    Unknown,
    StepOneFrame,
    Run,
    Pause,
    Restart
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

    SimulationDomain simulationDomain;
    SimulationCommand m_command;
    
    static SimulationParameters& GetInstance();
    static SimulationParameters* GetInstancePtr();
    static void SetCommand(SimulationCommand command);
    static SimulationCommand GetCommand();
};