#pragma once

#include <helper.h>
#include <simulation/simulation_parameters.h>
#include <simulation/pbf_smoothing_kernels.cuh>
#include <helper_cuda.h>

class PositionBasedFluidSimulator
{
public:
    PositionBasedFluidSimulator(float3 upperBoundary, float3 lowerBoundary);

    ~PositionBasedFluidSimulator();

    void Step(
        cudaGraphicsResource* positionsResource,
        cudaGraphicsResource* newPositionsResource,
        cudaGraphicsResource* velocitiesResource,
        cudaGraphicsResource* newVelocitiesResource,
        cudaGraphicsResource* indicesResource,
        int particlesNumber);

    void UpdateParameters();

private:
    void PredictPositions();
    void BuildUniformGrid();
    void CorrectPosition();
    void UpdateVelocity();
    void CorrectVelocity();

    void UpdateSmoothingKernels();

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
    int m_viscosityIterations;
    float m_vorticityEpsilon;
    float m_coef_corr;
    int m_substepsNumber;
    int m_particlesNumber;

    unsigned int* m_dIid;
    unsigned int* m_dCellIds;
    unsigned int* m_dCellStarts;
    unsigned int* m_dCellEnds;
    float3 m_upperBoundary;
    float3 m_lowerBoundary;
    int3 m_gridDimension;

    Poly6Kernel m_poly6Kernel = Poly6Kernel(1.f);
    SpikyGradientKernel m_spikyGradientKernel = SpikyGradientKernel(1.f);

    int m_blockSize = 512;
};

