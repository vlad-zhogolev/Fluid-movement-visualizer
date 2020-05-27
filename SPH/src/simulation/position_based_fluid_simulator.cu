#include <simulation/position_based_fluid_simulator.h>
#include <simulation/pbf_kernels.cuh>
#include <simulation/pbf_smoothing_kernels.cuh>
#include <simulation/converters.cuh>
#include <simulation/updaters.cuh>

#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <math_constants.h>

void PositionBasedFluidSimulator::PredictPositions()
{
    const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);

    pbf::cuda::kernels::PredictPositions<<<gridSize, m_blockSize>>>(
        m_dPositions,
        m_dVelocities,
        m_dNewPositions, 
        m_particlesNumber,
        make_float3(0, m_gravity, -9.8f),
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

    pbf::cuda::kernels::FindCellStartEnd<<<gridSize, m_blockSize, sharedMemorySize>>>(
        m_dCellIds, m_dCellStarts, m_dCellEnds, m_particlesNumber);

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::CorrectPosition() 
{
    //const Poly6Kernel poly6Kernel(m_h);
    //const SpikyGradientKernel spikyGradientKernel(m_h);
    const PositionToCellCoorinatesConverter positionToCellCoorinatesConverter(m_lowerBoundary, m_gridDimension, m_h);
    const CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter(m_gridDimension);

    bool writeToNewPositions = false;
    for (int i = 0; i < m_substepsNumber; ++i)
    {
        const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);
        pbf::cuda::kernels::CalculateLambda<<<gridSize, m_blockSize>>>(
            m_dLambdas,
            m_dDensities,
            m_dCellStarts,
            m_dCellEnds,
            m_gridDimension,
            m_dNewPositions,
            m_particlesNumber,
            1.0f / m_pho0,
            m_lambda_eps,
            m_h,
            positionToCellCoorinatesConverter,
            cellCoordinatesToCellIdConverter,
            m_poly6Kernel,
            m_spikyGradientKernel);

        m_coef_corr = -m_k_corr / powf(m_poly6Kernel(m_delta_q * m_delta_q), m_n_corr);

        pbf::cuda::kernels::CalculateNewPositions<<<gridSize, m_blockSize>>>(
            writeToNewPositions ? m_dTemporaryPositions : m_dNewPositions,
            writeToNewPositions ? m_dNewPositions : m_dTemporaryPositions,
            m_dCellStarts,
            m_dCellEnds,
            m_gridDimension,
            m_dLambdas,
            m_particlesNumber,
            1.0f / m_pho0,
            m_h,
            m_coef_corr,
            m_n_corr,
            positionToCellCoorinatesConverter,
            cellCoordinatesToCellIdConverter,
            m_upperBoundary,
            m_lowerBoundary,
            m_poly6Kernel,
            m_spikyGradientKernel);
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

    //const Poly6Kernel poly6Kernel(m_h);
    //const SpikyGradientKernel spikyGradientKernel(m_h);
    const PositionToCellCoorinatesConverter positionToCellCoorinatesConverter(
        m_lowerBoundary, m_gridDimension, m_h);
    const CellCoordinatesToCellIdConverter cellCoordinatesToCellIdConverter(m_gridDimension);

    pbf::cuda::kernels::CalculateVorticity<<<gridSize, m_blockSize>>>(
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
        m_spikyGradientKernel);
    
    pbf::cuda::kernels::ApplyVorticityConfinement<<<gridSize, m_blockSize>>> (
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
        m_spikyGradientKernel);

    if (m_c_XSPH > 0.5)
    {
        bool writeToNewVelocities = true;
        for (int i = 0; i < m_viscosityIterations; ++i)
        {
            pbf::cuda::kernels::ApplyXSPHViscosity<<<gridSize, m_blockSize>>>(
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
                m_poly6Kernel);
            writeToNewVelocities = !writeToNewVelocities;
        }
        if (writeToNewVelocities)
        {
            std::swap(m_dVelocities, m_dNewVelocities);
        }
    }
    else
    {
        pbf::cuda::kernels::ApplyXSPHViscosity<<<gridSize, m_blockSize>>>(
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
            m_poly6Kernel);
    }
    cudaDeviceSynchronize();
}