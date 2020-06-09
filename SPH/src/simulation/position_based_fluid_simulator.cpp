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

    // Particles data
    checkCudaErrors(cudaMalloc(&m_dLambdas, sizeof(float)          * MAX_PARTICLES_NUM));
    checkCudaErrors(cudaMalloc(&m_dDensities, sizeof(float)          * MAX_PARTICLES_NUM));
    checkCudaErrors(cudaMalloc(&m_dTemporaryPositions, sizeof(float3)         * MAX_PARTICLES_NUM));
    checkCudaErrors(cudaMalloc(&m_dCurl, sizeof(float3)         * MAX_PARTICLES_NUM));
    checkCudaErrors(cudaMalloc(&m_dCellIds, sizeof(unsigned int)   * MAX_PARTICLES_NUM));

    // TODO: check how much memory is needed here
    // Grid data
    int size = 2000000;
    // checkCudaErrors(cudaMalloc(&m_dCellStarts, sizeof(unsigned int)   * MAX_PARTICLES_NUM));
    // checkCudaErrors(cudaMalloc(&m_dCellEnds, sizeof(unsigned int)   * MAX_PARTICLES_NUM));
    // cudaMemset(m_dCellStarts, 0, sizeof(unsigned int) * MAX_PARTICLES_NUM);
    // cudaMemset(m_dCellEnds, 0, sizeof(unsigned int) * MAX_PARTICLES_NUM);
    checkCudaErrors(cudaMalloc(&m_dCellStarts, sizeof(unsigned int)   * size));
    checkCudaErrors(cudaMalloc(&m_dCellEnds, sizeof(unsigned int)   * size));
    cudaMemset(m_dCellStarts, 0, sizeof(unsigned int) * size);
    cudaMemset(m_dCellEnds, 0, sizeof(unsigned int) * size);
}

void PositionBasedFluidSimulator::Step(
    cudaGraphicsResource* positionsResource,
    cudaGraphicsResource* newPositionsResource,
    cudaGraphicsResource* velocitiesResource,
    cudaGraphicsResource* newVelocitiesResource,
    cudaGraphicsResource* indicesResource,
    int particlesNumber)
{
    m_particlesNumber = particlesNumber;

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

    UpdateSmoothingKernels();

    cudaDeviceSynchronize();
    PredictPositions();
    BuildUniformGrid();
    CorrectPosition();
    UpdateVelocity();
    CorrectVelocity();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &positionsResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &newPositionsResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &velocitiesResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &newVelocitiesResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &indicesResource, 0));
}

void PositionBasedFluidSimulator::UpdateParameters()
{
    const SimulationParameters& params = SimulationParameters::GetInstance();
    m_deltaTime = params.deltaTime;
    m_gravity = params.gravity;
    m_h = params.kernelRadius;
    m_pho0 = params.restDensity;
    m_lambda_eps = params.relaxationParameter;
    m_delta_q = params.deltaQ;
    m_k_corr = params.correctionCoefficient;
    m_n_corr = params.correctionPower;
    m_c_XSPH = params.c_XSPH;
    m_viscosityIterations = params.viscosityIterations;
    m_vorticityEpsilon = params.vorticityEpsilon;
    m_substepsNumber = params.substepsNumber;
    m_upperBoundary = params.GetUpperBoundary();
    m_lowerBoundary = params.GetLowerBoundary();
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

void PositionBasedFluidSimulator::UpdateSmoothingKernels()
{
    m_poly6Kernel = Poly6Kernel(m_h);
    m_spikyGradientKernel = SpikyGradientKernel(m_h);

    float3 diff = m_upperBoundary - m_lowerBoundary;
    m_gridDimension = make_int3(
        static_cast<int>(ceilf(diff.x / m_h)),
        static_cast<int>(ceilf(diff.y / m_h)),
        static_cast<int>(ceilf(diff.z / m_h)));

    m_positionToCellIdConverter = PositionToCellIdConverter(m_lowerBoundary, m_gridDimension, m_h);
    m_positionToCellCoorinatesConverter = PositionToCellCoorinatesConverter(m_lowerBoundary, m_gridDimension, m_h);
    m_cellCoordinatesToCellIdConverter = CellCoordinatesToCellIdConverter(m_gridDimension);
}