#include <glad/glad.h>
#include <simulation/position_based_fluid_simulator.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

PositionBasedFluidSimulator::PositionBasedFluidSimulator(float3 upperBoundary, float3 lowerBoundary)
    : m_upperBoundary(upperBoundary)
    , m_lowerBoundary(lowerBoundary)
{
    UpdateParameters();

    checkCudaErrors(cudaMalloc(&m_dCellIds, sizeof(unsigned int)   * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dCellStarts, sizeof(unsigned int)   * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dCellEnds, sizeof(unsigned int)   * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dLambdas, sizeof(float)          * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dDensities, sizeof(float)          * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dTemporaryPositions, sizeof(float3)         * MAX_PARTICLE_NUM));
    checkCudaErrors(cudaMalloc(&m_dCurl, sizeof(float3)         * MAX_PARTICLE_NUM));

    // TODO: check how much memory is needed here
    cudaMemset(m_dCellStarts, 0, sizeof(unsigned int) * MAX_PARTICLE_NUM);
    cudaMemset(m_dCellEnds, 0, sizeof(unsigned int) * MAX_PARTICLE_NUM);
}

void PositionBasedFluidSimulator::Step(
    unsigned int positions,
    unsigned int newPositions,
    unsigned int velocities,
    unsigned int newVelocities,
    unsigned int indices,
    int particlesNumber)
{
    m_particlesNumber = particlesNumber;

    struct cudaGraphicsResource* positionsResource;
    struct cudaGraphicsResource* newPositionsResource;
    struct cudaGraphicsResource* velocitiesResource;
    struct cudaGraphicsResource* newVelocitiesResource;
    struct cudaGraphicsResource* indicesResource;

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&positionsResource, positions, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&newPositionsResource, newPositions, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&velocitiesResource, velocities, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&newVelocitiesResource, newVelocities, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&indicesResource, indices, cudaGraphicsMapFlagsNone));


    checkCudaErrors(cudaGraphicsMapResources(1, &positionsResource, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &newPositionsResource, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &velocitiesResource, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &newVelocitiesResource, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &indicesResource, 0));

    size_t size;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_dPositions, &size, positionsResource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_dVelocities, &size, velocitiesResource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_dIid, &size, indicesResource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_dNewPositions, &size, newPositionsResource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_dNewVelocities, &size, newVelocitiesResource));


    cudaDeviceSynchronize();
    ApplyForcesAndPredictPositions();
    BuildUniformGrid();
    CorrectPosition();
    UpdateVelocity();
    CorrectVelocity();


    checkCudaErrors(cudaGraphicsUnmapResources(1, &positionsResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &newPositionsResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &velocitiesResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &newVelocitiesResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &indicesResource, 0));

    checkCudaErrors(cudaGraphicsUnregisterResource(positionsResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(newPositionsResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(velocitiesResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(newVelocitiesResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(indicesResource));
}

void PositionBasedFluidSimulator::UpdateParameters()
{
    const SimulationParameters& params = SimulationParameters::getInstance();
    m_deltaTime = params.deltaTime;
    m_gravity = params.g;
    m_h = params.kernelRadius;
    m_pho0 = params.restDensity;
    m_lambda_eps = params.relaxationParameter;
    m_delta_q = params.deltaQ;
    m_k_corr = params.correctionCoefficient;
    m_n_corr = params.correctionPower;
    m_c_XSPH = params.c_XSPH;
    m_vorticityEpsilon = params.vorticityEpsilon;
    m_niter = params.substepsNumber;
}

PositionBasedFluidSimulator::~PositionBasedFluidSimulator()
{
    checkCudaErrors(cudaFree(m_dCellIds));
    checkCudaErrors(cudaFree(m_dCellStarts));
    checkCudaErrors(cudaFree(m_dCellEnds));
    checkCudaErrors(cudaFree(m_dLambdas));
    checkCudaErrors(cudaFree(m_dDensities));
    checkCudaErrors(cudaFree(m_dTemporaryPositions));
    checkCudaErrors(cudaFree(m_dCurl));
}