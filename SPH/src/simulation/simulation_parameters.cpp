#include <simulation/simulation_parameters.h>
#include <iostream>

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
    instance.simulationDomain = SimulationDomain::Medium;
    instance.m_command = SimulationCommand::Unknown;

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
