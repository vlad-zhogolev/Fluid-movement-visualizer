#pragma once

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

    // Renderer
    int fps;
    
    static SimulationParameters& getInstance();
};