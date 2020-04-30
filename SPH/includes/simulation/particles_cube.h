#pragma once

#include <cstdlib>
#include <helper.h>

class ParticlesCube
{
public:
    ParticlesCube(float3 ulim, float3 llim, int3 ns);

    ~ParticlesCube();

	int initialize(unsigned int, unsigned int, unsigned int, int);
	int update(unsigned int, unsigned int, unsigned int, int);
	int reset(unsigned int, unsigned int, unsigned int, int);

private:
    float3 m_ulim;
    float3 m_llim;
	int3 m_ns;
	int m_count;

	float3 m_d;

    float3* m_pos;
    float3* m_vel;
    unsigned int *m_iid;
	int m_nallocated;

	void __realloc(int);
};

