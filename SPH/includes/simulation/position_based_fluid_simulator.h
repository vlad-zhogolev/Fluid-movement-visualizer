#pragma once

#include <helper.h>
#include <simulation/simulation_parameters.h>>
#include <math.h>
#include <helper_cuda.h>

class PositionBasedFluidSimulator
{
public:
    PositionBasedFluidSimulator(float3 upperBoundary, float3 lowerBoundary);

    ~PositionBasedFluidSimulator();

    void Step(
        unsigned int positions,
        unsigned int newPositions,
        unsigned int velocities,
        unsigned int newVelocities,
        unsigned int indices,
        int particlesNumber);

    void UpdateParameters();

private:
    void ApplyForcesAndPredictPositions();
    void BuildUniformGrid();
    void CorrectPosition();
    void UpdateVelocity();
    void CorrectVelocity();

private:
    float3* m_dPositions;
    float3* m_dTemporaryPositions;
    float3* m_dNewPositions;
    float3* m_dVelocities;
    float3* m_dNewVelocities;
    float* m_dLambdas;
    float* m_dDensities;
    float3* m_dCurl;

    float m_gravity;
    float m_h;
    float m_deltaTime;
    float m_pho0;
    float m_lambda_eps;
    float m_delta_q;
    float m_k_corr;
    float m_n_corr;
    float m_c_XSPH;
    float m_vorticityEpsilon;
    float m_coef_corr;
    int m_niter;
    int m_particlesNumber;

    unsigned int* m_dIid;
    unsigned int* m_dCellIds;
    unsigned int* m_dCellStarts;
    unsigned int* m_dCellEnds;
    float3 m_upperBoundary;
    float3 m_lowerBoundary;
    int3 m_gridDimension;

    int m_blockSize = 128;
};

