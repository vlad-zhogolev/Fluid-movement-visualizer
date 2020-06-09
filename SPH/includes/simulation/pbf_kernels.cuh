#pragma once

#include <utils.h>
#include <helper_math.h>
#include <simulation/pbf_smoothing_kernels.cuh>
#include <simulation/converters.cuh>

namespace pbf {

namespace cuda {

namespace kernels {

__device__
__forceinline__
int GetGlobalThreadIndex_1D_1D()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__
void PredictPositions(
    const float3* positions,
    float3* velocities,
    float3* predictedPositions,
    int particleNumber,
    float3 gravityAcceleration,
    float deltaTime);

__global__
void FindCellStartEnd(
    unsigned int* cellIds,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int particlesNumber);

__global__
void CalculateLambda(
    float* lambdas,
    float* densities,
    const unsigned int* cellStarts,
    const unsigned int* cellEnds,
    int3 gridDimension,
    const float3* positions,
    int particlesNumber,
    float restDensityInverse,
    float lambdaEpsilon,
    float h,
    PositionToCellCoorinatesConverter positionToCellCoorinatesConverter,
    CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter,
    Poly6Kernel poly6Kernel,
    SpikyGradientKernel spikyGradientKernel);

__global__
void CalculateNewPositions(
    const float3* positions,
    float3* newPositions,
    const unsigned int* cellStarts,
    const unsigned int* cellEnds,
    int3  gridDimension,
    const float* lambdas,
    int particlesNumber,
    float restDensityInverse,
    float h,
    float correctionCoefficient,
    float n_corr,
    PositionToCellCoorinatesConverter positionToCellCoorinatesConverter,
    CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter,
    float3 upperBoundary,
    float3 lowerBoundary,
    Poly6Kernel poly6Kernel,
    SpikyGradientKernel spikyGradientKernel);

__global__
void CalculateVorticity(
    const unsigned int* cellStarts,
    const unsigned int* cellEnds,
    int3 gridDimension,
    const float3* positions,
    const float3* velocities,
    float3* curl,
    int particleNumber,
    float h,
    PositionToCellCoorinatesConverter positionToCellCoorinatesConverter,
    CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter,
    SpikyGradientKernel spikyGradient);

__device__
float3 CalculateEta(
    int index,
    const float3* position,
    float3* curl,
    int particleNumber,
    const unsigned int* cellStarts,
    const unsigned int* cellEnds,
    const int3& gridDimension,
    float h,
    const PositionToCellCoorinatesConverter& positionToCellCoorinatesConverter,
    const CellCoordinatesToCellIdConverter& cellCoordinatesToCellIdConverter,
    const SpikyGradientKernel& spikyGradient);

__global__
void ApplyVorticityConfinement(
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int3 cellDim,
    float3* position,
    float3* curl,
    float3* newVelocity,
    int particleNumber,
    float h,
    float vorticityEpsilon,
    float deltaTime,
    PositionToCellCoorinatesConverter positionToCellCoorinatesConverter,
    CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter,
    SpikyGradientKernel spikyGradient);

__global__
void ApplyXSPHViscosity(
    const float3* positions,
    const float3* velocities,
    const float* densities,
    float3* newVelocities,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int3 gridDimension,
    int particleNumber,
    float cXSPH,
    float h,
    PositionToCellCoorinatesConverter positionToCellCoorinatesConverter,
    CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter,
    Poly6Kernel poly6Kernel);

} // namespace kernels

} // namespace cuda

} // namespace pbf