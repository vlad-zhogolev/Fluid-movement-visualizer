#pragma once

#include <helper_math.h>

struct PositionToCellCoorinatesConverter
{
public:
    __host__ __device__
    explicit PositionToCellCoorinatesConverter(const float3 &lowerBoundary, const int3 &gridDimension, float cellLength)
        : m_lowerBoundary(lowerBoundary)
        , m_gridDimension(gridDimension)
        , m_cellLengthInverse(1.0f / cellLength) {}

    __host__ __device__ __forceinline__
    int3 operator()(float3 position) const
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
    explicit CellCoordinatesToCellIdConverter(const int3 &gridDimension) 
        : m_gridDimension(gridDimension) {}

    __host__ __device__ __forceinline__
    int operator()(int cellX, int cellY, int cellZ) const
    {
        int cellId = cellX * m_gridDimension.y * m_gridDimension.z + cellY * m_gridDimension.z + cellZ;
        return cellId;
    }

    __host__ __device__ __forceinline__
    int operator()(int3 cellCoordinates) const
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
    __host__ __device__ __forceinline__
    explicit PositionToCellIdConverter(
        const float3& lowerBoundary,
        const int3& gridDimension,
        float cellLength)
        : m_positionToCellCoordinatesConverter(lowerBoundary, gridDimension, cellLength)
        , m_cellCoordinatesToCellIdConverter(gridDimension) {}

    __host__ __device__ __forceinline__
    int operator()(float3 position) const
    {
        int3 cellIndices = m_positionToCellCoordinatesConverter(position);
        int cellId = m_cellCoordinatesToCellIdConverter(cellIndices.x, cellIndices.y, cellIndices.z);

        return cellId;
    }

private:
    PositionToCellCoorinatesConverter m_positionToCellCoordinatesConverter;
    CellCoordinatesToCellIdConverter m_cellCoordinatesToCellIdConverter;
};
