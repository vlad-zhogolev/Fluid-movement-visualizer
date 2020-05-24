#pragma once

#include <helper.h>
#include <helper_math.h>

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
    int     particleNumber,
    float3  gravityAcceleration,
    float   deltaTime);

__global__
void CalculateCellStartEnd(
    unsigned int* cellIds,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int particlesNumber);

template <typename Func1, typename Func2, typename Poly6, typename SpikyGradient>
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
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    Poly6 poly6,
    SpikyGradient spiky)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particlesNumber)
    {
        return;
    }

    densities[index] = 0.f;
    float squaredGradientsSum{};
    float3 currentParticleGradient{};

    for (int xOffset = -1; xOffset <= 1; ++xOffset)
    {
        for (int yOffset = -1; yOffset <= 1; ++yOffset)
        {
            for (int zOffset = -1; zOffset <= 1; ++zOffset)
            {
                int3 cellCoordinates = positionToCellCoorinatesConverter(positions[index]);
                int x = cellCoordinates.x + xOffset;
                int y = cellCoordinates.y + yOffset;
                int z = cellCoordinates.z + zOffset;
                if (x < 0 || x >= gridDimension.x || y < 0 || y >= gridDimension.y || z < 0 || z >= gridDimension.z)
                {
                    continue;
                }
                int cellId = cellCoordinatesToCellIdConverter(x, y, z);
                for (int j = cellStarts[cellId]; j < cellEnds[cellId]; ++j)
                {
                    float3 positionDifference = positions[index] - positions[j];
                    float squaredPositionDifference = norm2(positionDifference);
                    densities[index] += poly6(squaredPositionDifference);
                    float3 gradient = spiky(positionDifference) * restDensityInverse;
                    currentParticleGradient += gradient;
                    if (index != j)
                    {
                        squaredGradientsSum += norm2(gradient);
                    }
                }
            }
        }
    }

    squaredGradientsSum += norm2(currentParticleGradient);
    float constraint = densities[index] * restDensityInverse - 1.0f;
    lambdas[index] = -constraint / (squaredGradientsSum + lambdaEpsilon);
}

template <typename Func1, typename Func2, typename Poly6, typename SpikyGradient>
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
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    float3 upperBoundary,
    float3 lowerBoundary,
    Poly6 poly6,
    SpikyGradient spiky)
{
    int i = GetGlobalThreadIndex_1D_1D();

    if (i >= particlesNumber)
    {
        return;
    }

    float3 deltaPosition{};

    for (int xOffset = -1; xOffset <= 1; ++xOffset)
    {
        for (int yOffset = -1; yOffset <= 1; ++yOffset)
        {
            for (int zOffset = -1; zOffset <= 1; ++zOffset)
            {
                int3 cellCoordinates = positionToCellCoorinatesConverter(positions[i]);
                int x = cellCoordinates.x + xOffset;
                int y = cellCoordinates.y + yOffset;
                int z = cellCoordinates.z + zOffset;
                if (x < 0 || x >= gridDimension.x || y < 0 || y >= gridDimension.y || z < 0 || z >= gridDimension.z)
                {
                    continue;
                }
                int cellId = cellCoordinatesToCellIdConverter(x, y, z);
                for (int j = cellStarts[cellId]; j < cellEnds[cellId]; ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }
                    float3 p = positions[i] - positions[j];
                    float corr = correctionCoefficient * powf(poly6(norm2(p)), n_corr);
                    deltaPosition += (lambdas[i] + lambdas[j] + corr) * spiky(p);
                }
            }
        }
    }

    deltaPosition = clamp(deltaPosition * restDensityInverse, -MAX_DP, MAX_DP);
    newPositions[i] = clamp(positions[i] + deltaPosition, lowerBoundary + LIM_EPS, upperBoundary - LIM_EPS);
    //deltaPositions[i] = clamp(deltaPosition, -MAX_DP, MAX_DP);
}

template<typename Func1, typename Func2, typename SpikyGradient>
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
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    SpikyGradient spikyGradient)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particleNumber)
    {
        return;
    }

    curl[index] = make_float3(0.0f);

    for (int xOffset = -1; xOffset <= 1; ++xOffset)
    {
        for (int yOffset = -1; yOffset <= 1; ++yOffset)
        {
            for (int zOffset = -1; zOffset <= 1; ++zOffset)
            {
                int3 cellCoordinates = positionToCellCoorinatesConverter(positions[index]);
                int x = cellCoordinates.x + xOffset;
                int y = cellCoordinates.y + yOffset;
                int z = cellCoordinates.z + zOffset;
                if (x < 0 || x >= gridDimension.x || y < 0 || y >= gridDimension.y || z < 0 || z >= gridDimension.z)
                {
                    continue;
                }
                int cellId = cellCoordinatesToCellIdConverter(x, y, z);
                for (int j = cellStarts[cellId]; j < cellEnds[cellId]; ++j)
                {

                    float3 positionDifference = positions[index] - positions[j];
                    if (length(positionDifference) >= h || j == index)
                    {
                        continue;
                        //return make_float3(0.0f, 0.0f, 0.0f);
                    }
                    float3 gradient = spikyGradient(positionDifference);
                    float3 velocityDifference = velocities[index] - velocities[j];
                    curl[index] += cross(velocityDifference, gradient);
                }
            }
        }
    }
    curl[index] = -curl[index];
}

template<typename Func1, typename Func2, typename SpikyGradient>
__device__
float3 CalculateVorticityGradient(
    int index,
    float3* position,
    float3* curl,
    int particleNumber,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int3 gridDimension,
    float h,
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    SpikyGradient spikyGradient)
{
    float3 vorticityGradient{};

    for (int xOffset = -1; xOffset <= 1; ++xOffset)
    {
        for (int yOffset = -1; yOffset <= 1; ++yOffset)
        {
            for (int zOffset = -1; zOffset <= 1; ++zOffset)
            {
                int3 cellCoordinates = positionToCellCoorinatesConverter(position[index]);
                int x = cellCoordinates.x + xOffset;
                int y = cellCoordinates.y + yOffset;
                int z = cellCoordinates.z + zOffset;
                if (x < 0 || x >= gridDimension.x || y < 0 || y >= gridDimension.y || z < 0 || z >= gridDimension.z)
                {
                    continue;
                }
                int cellId = cellCoordinatesToCellIdConverter(x, y, z);
                for (int j = cellStarts[cellId]; j < cellEnds[cellId]; ++j)
                {
                    float3 positionDifference = position[index] - position[j];
                    if (length(positionDifference) >= h || j == index)
                    {
                        continue;
                        //return make_float3(0.0f, 0.0f, 0.0f);
                    }
                    //float curlLengthDifference = length(curl[index] - curl[j]);
                    //vorticityGradient += make_float3(curlLengthDifference / positionDifference.x, curlLengthDifference / positionDifference.y, curlLengthDifference / positionDifference.z);
                    float3 gradient = spikyGradient(positionDifference);
                    float curlLength = length(curl[j]);
                    vorticityGradient += curlLength * gradient;
                }
            }
        }
    }

    return vorticityGradient;
}

template <typename Func1, typename Func2, typename SpikyGradient>
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
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    SpikyGradient spikyGradient)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particleNumber)
    {
        return;
    }

    float3 vorticityGradient = CalculateVorticityGradient(
        index,
        position,
        curl,
        particleNumber,
        cellStarts,
        cellEnds,
        cellDim,
        h,
        positionToCellCoorinatesConverter,
        cellCoordinatesToCellIdConverter,
        spikyGradient);

    float vorticityGradientLength = length(vorticityGradient);
    float3 normalizedVorticityGradient{};
    if (vorticityGradientLength > 1.0e-4f)  // TODO: define some parameter for this epsilon value
    {
        normalizedVorticityGradient = normalize(vorticityGradient);
    }

    float3 vorticityForce = vorticityEpsilon * cross(normalizedVorticityGradient, curl[index]);
    newVelocity[index] += vorticityForce * deltaTime;
}

template <typename Func1, typename Func2, typename Poly6>
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
    float c_XSPH,
    float h,
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    Poly6 poly6)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particleNumber)
    {
        return;
    }

    float3 accumulatedVelocity{};

    for (int xOffset = -1; xOffset <= 1; ++xOffset)
    {
        for (int yOffset = -1; yOffset <= 1; ++yOffset)
        {
            for (int zOffset = -1; zOffset <= 1; ++zOffset)
            {
                int3 cellCoordinates = positionToCellCoorinatesConverter(positions[index]);
                int x = cellCoordinates.x + xOffset;
                int y = cellCoordinates.y + yOffset;
                int z = cellCoordinates.z + zOffset;
                if (x < 0 || x >= gridDimension.x || y < 0 || y >= gridDimension.y || z < 0 || z >= gridDimension.z)
                {
                    continue;
                }
                int cellId = cellCoordinatesToCellIdConverter(x, y, z);
                for (int j = cellStarts[cellId]; j < cellEnds[cellId]; ++j)
                {
                    float3 positionDifference = positions[index] - positions[j];
                    float3 velocityDifference = velocities[j] - velocities[index];
                    float averageDensityInverse = 2.f / (densities[index] + densities[j]);
                    accumulatedVelocity += velocityDifference * averageDensityInverse * poly6(norm2(positionDifference));
                }
            }
        }
    }
    newVelocities[index] = velocities[index] + c_XSPH * accumulatedVelocity;
}


} // namespace kernels

} // namespace cuda

} // namespace pbf