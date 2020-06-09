#include <simulation/pbf_kernels.cuh>

namespace pbf {

namespace cuda {

namespace kernels {

__global__
void PredictPositions(
    const float3* positions,
    float3* velocities,
    float3* predictedPositions,
    int     particleNumber,
    float3  gravityAcceleration,
    float   deltaTime)
{
    int index = GetGlobalThreadIndex_1D_1D();
    if (index >= particleNumber)
    {
        return;
    }

    velocities[index] += gravityAcceleration * deltaTime;
    predictedPositions[index] = positions[index] + velocities[index] * deltaTime;
}

__global__
void FindCellStartEnd(
    unsigned int* cellIds,
    unsigned int* cellStarts,
    unsigned int* cellEnds,
    int particlesNumber)
{
    extern __shared__ unsigned int sharedCellIds[];

    int index = GetGlobalThreadIndex_1D_1D();

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
    SpikyGradientKernel spikyGradientKernel)
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
                    densities[index] += poly6Kernel(squaredPositionDifference);
                    float3 gradient = spikyGradientKernel(positionDifference) * restDensityInverse;
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
    SpikyGradientKernel spikyGradientKernel)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particlesNumber)
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
                    //if (index == j)
                    //{
                    //    continue;
                    //}
                    float3 positionDifference = positions[index] - positions[j];
                    float lambdaCorr = correctionCoefficient * powf(poly6Kernel(norm2(positionDifference)), n_corr);
                    deltaPosition += (lambdas[index] + lambdas[j] + lambdaCorr) * spikyGradientKernel(positionDifference);
                }
            }
        }
    }

    deltaPosition = clamp(deltaPosition * restDensityInverse, -MAX_DELTA_POSITION, MAX_DELTA_POSITION);
    newPositions[index] = clamp(positions[index] + deltaPosition, lowerBoundary + LIM_EPS, upperBoundary - LIM_EPS);
    //deltaPositions[i] = clamp(deltaPosition, -MAX_DELTA_POSITION, MAX_DELTA_POSITION);
}

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
    SpikyGradientKernel spikyGradient)
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
                    //if (length(positionDifference) >= h || j == index)
                    //{
                    //    continue;
                    //    //return make_float3(0.0f, 0.0f, 0.0f);
                    //}
                    float3 gradient = spikyGradient(positionDifference);
                    float3 velocityDifference = velocities[j] - velocities[index];
                    curl[index] += cross(velocityDifference, gradient);
                }
            }
        }
    }
    //curl[index] = -curl[index];
}

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
    const SpikyGradientKernel& spikyGradient)
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
    SpikyGradientKernel spikyGradient)
{
    int index = GetGlobalThreadIndex_1D_1D();

    if (index >= particleNumber)
    {
        return;
    }

    float3 eta = CalculateEta(
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

    float etaLength = length(eta);
    float3 normalizedEta{};
    if (etaLength > 1.0e-4f)  // TODO: define some parameter for this epsilon value
    {
        normalizedEta = normalize(eta);
    }

    float3 vorticityForce = vorticityEpsilon * cross(normalizedEta, curl[index]);
    newVelocity[index] += vorticityForce * deltaTime;
}

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
    Poly6Kernel poly6Kernel)
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
                    accumulatedVelocity += velocityDifference * averageDensityInverse * poly6Kernel(norm2(positionDifference));
                }
            }
        }
    }
    newVelocities[index] = velocities[index] + cXSPH * accumulatedVelocity;
}

} // namespace kernels

} // namespace cuda

} // namespace pbf