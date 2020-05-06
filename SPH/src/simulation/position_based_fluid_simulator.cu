#include <simulation/position_based_fluid_simulator.h>
#include <simulation/pbf_kernels.cuh>

#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <math_constants.h>

struct PositionToCellCoorinatesConverter
{
public:
    __host__ __device__
    PositionToCellCoorinatesConverter(const float3 &lowerBoundary, const int3 &gridDimension, float cellLength)
        : m_lowerBoundary(lowerBoundary)
        , m_gridDimension(gridDimension)
        , m_cellLengthInverse(1.0f / cellLength) {}

    __host__ __device__
    int3 operator()(float3 position)
    {
        float3 localPosition = position - m_lowerBoundary;
        int3 cellIndices = make_int3(localPosition * m_cellLengthInverse);
        cellIndices = clamp(cellIndices, make_int3(0, 0, 0), m_gridDimension - 1);

        return cellIndices;
    }

private:
    float3 m_lowerBoundary;
    int3 m_gridDimension;
    float m_cellLengthInverse;
};

struct CellCoordinatesToCellIdConverter
{
public:
    __host__ __device__
    CellCoordinatesToCellIdConverter(const int3 &gridDimension) : m_gridDimension(gridDimension) {}

    __host__ __device__
    inline int operator()(int cellX, int cellY, int cellZ)
    {
        int cellId = cellX * m_gridDimension.y * m_gridDimension.z + cellY * m_gridDimension.z + cellZ;
        return cellId;
    }

    __host__ __device__
    inline int operator()(int3 cellCoordinates)
    {
        int cellId = operator()(cellCoordinates.x, cellCoordinates.y, cellCoordinates.z);
        return cellId;
    }

private:
    int3 m_gridDimension;
};

struct PositionToCellIdConverter
{
public:
    __host__ __device__
    PositionToCellIdConverter(const float3& lowerBoundary, const int3& gridDimension, float cellLength)
        : m_positionToCellCoordinatesConverter(lowerBoundary, gridDimension, cellLength)
        , m_cellCoordinatesToCellIdConverter(gridDimension) {}

    __host__ __device__
    int operator()(float3 position)
    {
        int3 cellIndices = m_positionToCellCoordinatesConverter(position);
        int cellId = m_cellCoordinatesToCellIdConverter(cellIndices.x, cellIndices.y, cellIndices.z);
        
        return cellId;
    }

private:
    PositionToCellCoorinatesConverter m_positionToCellCoordinatesConverter;
    CellCoordinatesToCellIdConverter m_cellCoordinatesToCellIdConverter;
};

struct Poly6Kernel
{
public:
    __host__ __device__
    Poly6Kernel(float kernelRadius)
        : m_kernelRadius2(kernelRadius * kernelRadius)
        , m_coefficient(315.f * powf(1.f / kernelRadius, 9) / (64.f * CUDART_PI_F)) {}

    __host__ __device__
    float operator()(float squaredDistance)
    {
        if (squaredDistance >= m_kernelRadius2)
        {
            return 0;
        }
        float difference = m_kernelRadius2 - squaredDistance;
        return m_coefficient * difference * difference * difference;
    }

private:
    float m_coefficient;
    float m_kernelRadius2;
};

struct SpikyGradientKernel
{
public:
    __host__ __device__
    SpikyGradientKernel(float kernelRadius) 
        : m_kernelRadius(kernelRadius)
        , m_coefficient(-45.f / (CUDART_PI_F * powf(kernelRadius, 6))) {}

    __host__ __device__
    float3 operator()(float3 vector)
    {
        float vectorLength = length(vector);
        if (vectorLength < KERNEL_EPS || vectorLength >= m_kernelRadius)
        {
            return float3{};
        }
        float difference = m_kernelRadius - vectorLength;   
        return m_coefficient * difference * difference / vectorLength * vector;
    }

private:
    float m_kernelRadius;
    float m_coefficient;
};

struct PositionUpdater
{
public:
    __host__ __device__
    PositionUpdater(float3 upperBoundary, float3 lowerBoundary)
        : m_upperBoundary(upperBoundary)
        , m_lowerBoundary(lowerBoundary) {}

    __host__ __device__
    float3 operator()(const thrust::tuple<float3, float3>& tuple)
    {
        float3 position = thrust::get<0>(tuple);
        float3 deltaPosition = thrust::get<1>(tuple);
        float3 newPosition = clamp(position + deltaPosition, m_lowerBoundary + LIM_EPS, m_upperBoundary - LIM_EPS);

        return newPosition;
    }
private:
    float3 m_upperBoundary;
    float3 m_lowerBoundary;
};

struct VelocityUpdater
{
public:
    __host__ __device__
    VelocityUpdater(float deltaTime) : m_deltaTimeInverse(1.f / deltaTime) {}

    __host__ __device__
    float3 operator()(const thrust::tuple<float3, float3>& tuple)
    {
        float3 pos = thrust::get<0>(tuple);
        float3 npos = thrust::get<1>(tuple);
        float3 newVelocity = (npos - pos) * m_deltaTimeInverse;

        return newVelocity;
    }

private:
    float m_deltaTimeInverse;
};

void PositionBasedFluidSimulator::ApplyForcesAndPredictPositions()
{
    const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);

    ApplyForcesAndPredictPositionsKernel<<<gridSize, m_blockSize>>>(
        m_dPositions,
        m_dVelocities,
        m_dNewPositions, 
        m_particlesNumber,
        make_float3(0, 0, -m_gravity),
        m_deltaTime);

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::BuildUniformGrid()
{
    thrust::device_ptr<float3> positions(m_dPositions);
    thrust::device_ptr<float3> newPositions(m_dNewPositions);
    thrust::device_ptr<unsigned int> cellIds(m_dCellIds);
    
    float3 diff = m_upperBoundary - m_lowerBoundary;
    m_gridDimension = make_int3(
        static_cast<int>(ceilf(diff.x / m_h)),
        static_cast<int>(ceilf(diff.y / m_h)),
        static_cast<int>(ceilf(diff.z / m_h)));

    thrust::transform(
        newPositions,
        newPositions + m_particlesNumber,
        cellIds,
        PositionToCellIdConverter(m_lowerBoundary, m_gridDimension, m_h));

    thrust::device_ptr<float3> velocities(m_dVelocities);
    thrust::device_ptr<float3> newVelocitites(m_dNewVelocities);
    thrust::device_ptr<unsigned int> d_iid(m_dIid);

    thrust::sort_by_key(
        cellIds,
        cellIds + m_particlesNumber,
        thrust::make_zip_iterator(thrust::make_tuple(positions, velocities, newPositions, newVelocitites, d_iid)));

    const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);
    const int sharedMemorySize = sizeof(unsigned int) * (m_blockSize + 1);

    int cellsNumber = m_gridDimension.x * m_gridDimension.y * m_gridDimension.z;
    cudaMemset(m_dCellStarts, 0, sizeof(m_dCellStarts[0]) * cellsNumber);
    cudaMemset(m_dCellEnds, 0, sizeof(m_dCellEnds[0]) * cellsNumber);

    CalculateCellStartEnd<<<gridSize, m_blockSize, sharedMemorySize>>>(
        m_dCellIds, m_dCellStarts, m_dCellEnds, m_particlesNumber);

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::CorrectPosition() 
{
    bool writeToNewPositions = false;
    for (int i = 0; i < m_substepsNumber; ++i)
    {
        const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);
        CalculateLambda <<<gridSize, m_blockSize>>>(
            m_dLambdas,
            m_dDensities,
            m_dCellStarts,
            m_dCellEnds,
            m_gridDimension,
            m_dNewPositions,
            m_particlesNumber,
            m_pho0,
            m_lambda_eps,
            m_h,
            PositionToCellCoorinatesConverter(m_lowerBoundary, m_gridDimension, m_h),
            CellCoordinatesToCellIdConverter(m_gridDimension),
            Poly6Kernel(m_h),
            SpikyGradientKernel(m_h));

        m_coef_corr = -m_k_corr / powf(Poly6Kernel(m_h)(m_delta_q*m_delta_q), m_n_corr);

        CalculateNewPositions <<<gridSize, m_blockSize>>>(
            writeToNewPositions ? m_dTemporaryPositions : m_dNewPositions,
            writeToNewPositions ? m_dNewPositions : m_dTemporaryPositions,
            m_dCellStarts,
            m_dCellEnds,
            m_gridDimension,
            m_dLambdas,
            m_particlesNumber,
            m_pho0,
            m_h,
            m_coef_corr,
            m_n_corr,
            PositionToCellCoorinatesConverter(m_lowerBoundary, m_gridDimension, m_h),
            CellCoordinatesToCellIdConverter(m_gridDimension),
            m_upperBoundary,
            m_lowerBoundary,
            Poly6Kernel(m_h),
            SpikyGradientKernel(m_h));
        writeToNewPositions = !writeToNewPositions;
    }
    if (writeToNewPositions)
    {
        std::swap(m_dTemporaryPositions, m_dNewPositions);
    }
    // thrust::transform(
    //     thrust::make_zip_iterator(thrust::make_tuple(m_dNewPositions, m_dTemporaryPositions)),
    //     thrust::make_zip_iterator(
    //          thrust::make_tuple(m_dNewPositions + m_particlesNumber, m_dTemporaryPositions + m_particlesNumber)),
    //     m_dNewPositions,
    //     PositionUpdater(m_upperBoundary, m_lowerBoundary));
    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::UpdateVelocity()
{
    /* Warn: assume m_dPositions updates to m_dNewPositions after CorrectPosition() */
    thrust::device_ptr<float3> d_pos(m_dPositions);
    thrust::device_ptr<float3> d_npos(m_dNewPositions);
    thrust::device_ptr<float3> d_vel(m_dVelocities);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_npos)),
        thrust::make_zip_iterator(thrust::make_tuple(d_pos + m_particlesNumber, d_npos + m_particlesNumber)),
        d_vel,
        VelocityUpdater(m_deltaTime));

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::CorrectVelocity() {
    
    const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);
    const PositionToCellCoorinatesConverter positionToCellCoorinatesConverter(
        m_lowerBoundary, m_gridDimension, m_h);
    const CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter(m_gridDimension);

    CalculateVorticity<<<gridSize, m_blockSize>>>(
        m_dCellStarts,
        m_dCellEnds,
        m_gridDimension,
        m_dPositions,  // Need to determine which position (old or new to pass here). Same for velocity
        m_dVelocities,
        m_dCurl,
        m_particlesNumber,
        m_h,
        positionToCellCoorinatesConverter,
        cellCoordinatesToCellIdConverter,
        SpikyGradientKernel(m_h));
    
    ApplyVorticityConfinement<<<gridSize, m_blockSize>>> (
        m_dCellStarts,
        m_dCellEnds,
        m_gridDimension,
        m_dPositions,
        m_dCurl,
        m_dVelocities,
        m_particlesNumber,
        m_h,
        m_vorticityEpsilon,
        m_deltaTime,
        positionToCellCoorinatesConverter,
        cellCoordinatesToCellIdConverter,
        SpikyGradientKernel(m_h));

    if (m_c_XSPH > 0.5)
    {
        bool writeToNewVelocities = true;
        for (int i = 0; i < m_viscosityIterations; ++i)
        {
            ApplyXSPHViscosity<<<gridSize, m_blockSize>>>(
                m_dPositions,
                writeToNewVelocities ? m_dVelocities : m_dNewVelocities,
                m_dDensities,
                writeToNewVelocities ? m_dNewVelocities : m_dVelocities,
                m_dCellStarts,
                m_dCellEnds,
                m_gridDimension,
                m_particlesNumber,
                m_c_XSPH / m_viscosityIterations,
                m_h,
                positionToCellCoorinatesConverter,
                cellCoordinatesToCellIdConverter,
                Poly6Kernel(m_h));
            writeToNewVelocities = !writeToNewVelocities;
        }
        if (writeToNewVelocities)
        {
            std::swap(m_dVelocities, m_dNewVelocities);
        }
    }
    else
    {
        ApplyXSPHViscosity<<<gridSize, m_blockSize>>>(
            m_dPositions,
            m_dVelocities,
            m_dDensities,
            m_dNewVelocities,
            m_dCellStarts,
            m_dCellEnds,
            m_gridDimension,
            m_particlesNumber,
            m_c_XSPH,
            m_h,
            positionToCellCoorinatesConverter,
            cellCoordinatesToCellIdConverter,
            Poly6Kernel(m_h));
    }

    

    //thrust::device_ptr<float3> newVelocities(m_dNewVelocities), velocities(m_dTemporaryPositions);
    //thrust::copy_n(velocities, m_particlesNumber, newVelocities);
    cudaDeviceSynchronize();
}