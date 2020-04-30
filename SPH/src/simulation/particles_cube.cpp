#include <glad/glad.h>
#include <simulation/particles_cube.h>
#include <cstdlib>

ParticlesCube::ParticlesCube(float3 ulim, float3 llim, int3 ns)
    : m_ulim(ulim)
    , m_llim(llim)
    , m_ns(ns)
    , m_count(0)
    , m_pos(nullptr)
    , m_vel(nullptr)
    , m_iid(nullptr)
    , m_nallocated(0)
{
    m_d = ulim - llim;
    m_d.x /= ns.x;
    m_d.y /= ns.y;
    m_d.z /= ns.z;
};

int ParticlesCube::initialize(unsigned int pos, unsigned int vel, unsigned int iid, int max_nparticle) {
	
	srand(27);

	m_count = 0;
	__realloc(max_nparticle);
	float sx = m_llim.x + m_d.x / 2, sy = m_llim.y + m_d.y / 2, sz = m_llim.z + m_d.z / 2, x, y, z;

	x = sx;
	for (int i = 0; i < m_ns.x; i++, x += m_d.x) {
		y = sy;
		for (int j = 0; j < m_ns.y; j++, y += m_d.y) {
			z = sz;
			for (int k = 0; k < m_ns.z; k++, z += m_d.z, m_count++) {
				float r1 = 1.f * rand() / RAND_MAX, r2 = 1.f * rand() / RAND_MAX, r3 = 1.f * rand() / RAND_MAX;
				m_pos[m_count] = make_float3(x, y, z) + 0.1f * make_float3(sx * r1, sy * r2, sz * r3);
				m_vel[m_count] = make_float3(0.f, 0.f, 0.f);
				m_iid[m_count] = m_count;
			}
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, pos);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_pos[0]), m_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vel);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_vel[0]), m_vel);
	glBindBuffer(GL_ARRAY_BUFFER, iid);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_iid[0]), m_iid);

	return m_count;
}

int ParticlesCube::update(unsigned int pos, unsigned int vel, unsigned int iid, int max_nparticle) {
	return m_count;
}

int ParticlesCube::reset(unsigned int pos, unsigned int vel, unsigned int iid, int max_nparticle) {
	return initialize(pos, vel, iid, max_nparticle);
}

void ParticlesCube::__realloc(int max_nparticle)
{
	if (m_nallocated < max_nparticle) {
		if (m_pos) {
			m_pos = (float3*)realloc(m_pos, max_nparticle * sizeof(m_pos[0]));
			m_vel = (float3*)realloc(m_vel, max_nparticle * sizeof(m_vel[0]));
			m_iid = (unsigned int*)realloc(m_iid, max_nparticle * sizeof(m_iid[0]));
		}
		else {
			m_pos = (float3*)malloc(max_nparticle * sizeof(m_pos[0]));
			m_vel = (float3*)malloc(max_nparticle * sizeof(m_vel[0]));
			m_iid = (unsigned int*)malloc(max_nparticle * sizeof(m_iid[0]));
		}

		m_nallocated = max_nparticle;
	}
}


ParticlesCube::~ParticlesCube()
{
    if (m_pos)
    {
        free(m_pos);
        free(m_vel);
        free(m_iid);
    }
}
