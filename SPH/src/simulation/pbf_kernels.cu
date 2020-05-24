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

} // namespace kernels

} // namespace cuda

} // namespace pbf