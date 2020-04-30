#pragma once
#include <helper.h>
#include <helper_math.h>

__global__
void ApplyForcesAndPredictPositionsKernel(
    const float3* positions,
    float3* velocities,
    float3* predictedPositions,
    int     particleNumber,
    float3  gravityAcceleration,
    float   deltaTime)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleNumber)
    {
        return;
    }

    velocities[index] += gravityAcceleration * deltaTime;
    predictedPositions[index] = positions[index] + velocities[index] * deltaTime;
}

__global__
void CalculateCellStartEnd(
    unsigned int* cellIds,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int particlesNumber)
{
    extern __shared__ unsigned int sharedCellIds[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < particlesNumber)
    {
        sharedCellIds[threadIdx.x + 1] = cellIds[index];
        // skip writing previous id for first particle (because there is no particle before it)
        if (index > 0 && threadIdx.x == 0)
        {
            sharedCellIds[0] = cellIds[index - 1];
        }
    }

    __syncthreads();

    if (index < particlesNumber)
    {
        // If current particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell.

        unsigned int currentCellId = sharedCellIds[threadIdx.x + 1];
        unsigned int previousCellId = sharedCellIds[threadIdx.x];

        if (index == 0 || currentCellId != previousCellId)
        {
            cellStarts[currentCellId] = index;
            if (index != 0)
            {
                cellEnds[previousCellId] = index;
            }
        }

        if (index == particlesNumber - 1)
        {
            cellEnds[currentCellId] = particlesNumber;
        }
    }
}

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
    float pho0,
    float lambdaEpsilon,
    float h,
    Func1 positionToCellCoorinatesConverter,
    Func2 cellCoordinatesToCellIdConverter,
    Poly6 poly6,
    SpikyGradient spiky)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
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
                    float3 gradient = spiky(positionDifference) / pho0;
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
    lambdas[index] = -(densities[index] / pho0 - 1.0f) / (squaredGradientsSum + lambdaEpsilon);
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
    float pho0,
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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

    deltaPosition = clamp(deltaPosition / pho0, -MAX_DP, MAX_DP);
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
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
                    accumulatedVelocity += velocityDifference * poly6(norm2(positionDifference));
                }
            }
        }
    }
    newVelocities[index] = velocities[index] + c_XSPH * accumulatedVelocity;
}