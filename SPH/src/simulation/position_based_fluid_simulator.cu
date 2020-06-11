#include <simulation/position_based_fluid_simulator.h>
#include <simulation/pbf_kernels.cuh>
#include <simulation/updaters.cuh>

#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
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
        m_gravity,
        m_deltaTime);

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::BuildUniformGrid()
{
    thrust::device_ptr<float3> positions(m_dPositions);
    thrust::device_ptr<float3> newPositions(m_dNewPositions);
    thrust::device_ptr<unsigned int> cellIds(m_dCellIds);

    thrust::transform(
        newPositions,
        newPositions + m_particlesNumber,
        cellIds,
        m_positionToCellIdConverter
    );

    thrust::device_ptr<float3> velocities(m_dVelocities);
    thrust::device_ptr<float3> newVelocitites(m_dNewVelocities);

    thrust::sort_by_key(
        cellIds,
        cellIds + m_particlesNumber,
        thrust::make_zip_iterator(thrust::make_tuple(positions, velocities, newPositions, newVelocitites)));

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
            m_positionToCellCoorinatesConverter,
            m_cellCoordinatesToCellIdConverter,
            m_poly6Kernel,
            m_spikyGradientKernel);

        const float deltaQ = m_delta_q * m_h;
        m_coef_corr = -m_k_corr / powf(m_poly6Kernel(deltaQ * deltaQ), m_n_corr);

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
            m_positionToCellCoorinatesConverter,
            m_cellCoordinatesToCellIdConverter,
            m_upperBoundary,
            m_lowerBoundary,
            m_poly6Kernel,
            m_spikyGradientKernel);
        writeToNewPositions = !writeToNewPositions;
    }
    if (writeToNewPositions)
    {
        // Can't just swap because m_dTemporaryPositions is not mapped OpenGL resource and m_dNewPositions is.
        thrust::device_ptr<float3> tmpPositions(m_dTemporaryPositions);
        thrust::device_ptr<float3> newPositions(m_dNewPositions);
        thrust::copy_n(tmpPositions, m_particlesNumber, newPositions);
        //std::swap(m_dTemporaryPositions, m_dNewPositions);
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
    thrust::device_ptr<float3> positions(m_dPositions);
    thrust::device_ptr<float3> newPositions(m_dNewPositions);
    thrust::device_ptr<float3> velocities(m_dVelocities);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(positions, newPositions)),
        thrust::make_zip_iterator(thrust::make_tuple(positions + m_particlesNumber, newPositions + m_particlesNumber)),
        velocities,
        VelocityUpdater(m_deltaTime));

    cudaDeviceSynchronize();
}

void PositionBasedFluidSimulator::CorrectVelocity() {
    
    const int gridSize = ceilDiv(m_particlesNumber, m_blockSize);

    pbf::cuda::kernels::CalculateVorticity<<<gridSize, m_blockSize>>>(
        m_dCellStarts,
        m_dCellEnds,
        m_gridDimension,
        m_dPositions,  // Need to determine which position (old or new to pass here). Same for velocity
        m_dVelocities,
        m_dCurl,
        m_particlesNumber,
        m_h,
        m_positionToCellCoorinatesConverter,
        m_cellCoordinatesToCellIdConverter,
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
        m_positionToCellCoorinatesConverter,
        m_cellCoordinatesToCellIdConverter,
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
                m_positionToCellCoorinatesConverter,
                m_cellCoordinatesToCellIdConverter,
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
            m_positionToCellCoorinatesConverter,
            m_cellCoordinatesToCellIdConverter,
            m_poly6Kernel);
    }
    cudaDeviceSynchronize();
}