#include <rendering/camera.h>
#include <input.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/common.hpp>
#include <glm/gtx/rotate_vector.hpp>

const float Camera::SCREEN_ROTATE_RATE = 0.005f;
const float Camera::SCREEN_PAN_RATE = 0.002f;
const float Camera::SCREEN_SCROLL_RATE = 0.1f;
const float Camera::MIN_SCROLL_DISTANCE = 0.1f;
const float Camera::MAX_SCROLL_DISTANCE = 10.0f;
const float Camera::NEAR_PLANE_DISTANCE = 0.1f;
const float Camera::FAR_PLANE_DISTANCE = 100.0f;

Camera::~Camera() {}

Camera::Camera(const glm::vec3& position, const glm::vec3& focus, float aspect)
{
    glm::vec3 front(focus - position), up = glm::vec3(0.f, 0.f, 1.f);
    up = glm::cross(front, glm::cross(up, front));
    *this = Camera(position, front, up, 60.f, aspect);
}

Camera::Camera(const glm::vec3& position, const glm::vec3 front, const glm::vec3 up, float fov, float aspect)
    : m_position(position)
    , m_front(front)
    , m_up(glm::normalize(up))
    , m_fov(fov)
    , m_aspect(aspect)
{
    m_rotx = glm::vec3(0.f, 0.f, 1.f);
    m_roty = glm::normalize(glm::cross(front, m_rotx));
}

void Camera::use(const Shader & shader, bool translate_invariant) const
{
    glm::mat4 view = glm::lookAt(m_position, m_position + m_front, m_up);
    if (translate_invariant)
    {
        view = glm::mat4(glm::mat3(view));
    }
    glm::mat4 pers = glm::perspective(glm::radians(m_fov), m_aspect, NEAR_PLANE_DISTANCE, FAR_PLANE_DISTANCE);
    shader.setUnif("view", view);
    shader.setUnif("proj", pers);
}

void Camera::rotate(const glm::vec2 dxy)
{
    glm::vec3 center = m_position + m_front;
    // Horizontal rotation
    if (dxy.x != 0)
    {
        const glm::vec3 &axis = m_rotx;
          
        m_front = glm::rotate(m_front, -dxy.x * SCREEN_ROTATE_RATE, axis);
        m_up = glm::rotate(m_up, -dxy.x * SCREEN_ROTATE_RATE, axis);
        m_position = center - m_front;

        m_roty = glm::rotate(m_roty, -dxy.x * SCREEN_ROTATE_RATE, axis);
    }
    // Vertical rotation
    if (dxy.y != 0) 
    {
        const glm::vec3& axis = m_roty;

        m_front = glm::rotate(m_front, -dxy.y * SCREEN_ROTATE_RATE, axis);
        m_up = glm::rotate(m_up, -dxy.y * SCREEN_ROTATE_RATE, axis);
        m_position = center - m_front;
    }

}

void Camera::pan(const glm::vec2 dxy)
{
    glm::vec3 normalizedRight = -glm::normalize(glm::cross(m_front, m_up));
    glm::vec3 cam_d = dxy.x * normalizedRight + dxy.y * glm::normalize(m_up);
    m_position += SCREEN_PAN_RATE * cam_d * glm::length(m_front);
}

void Camera::zoom(float diff)
{
    if (diff > 0)
    {
        if (m_front.length() < MIN_SCROLL_DISTANCE)
        {
            return;
        }
        m_position += m_front * SCREEN_SCROLL_RATE;
        m_front -= m_front * SCREEN_SCROLL_RATE;
    }
    else
    {
        if (m_front.length() > MAX_SCROLL_DISTANCE)
        {
            return;
        }
        m_position -= m_front * SCREEN_SCROLL_RATE;
        m_front += m_front * SCREEN_SCROLL_RATE;
    }
}

ProjectionInfo Camera::getProjectionInfo() const
{
    ProjectionInfo projectionInfo;
    float tanHalfFov = tan(glm::radians(m_fov) * 0.5f);
    projectionInfo.n = NEAR_PLANE_DISTANCE;
    projectionInfo.f = FAR_PLANE_DISTANCE;
    projectionInfo.t = tanHalfFov * projectionInfo.n;
    projectionInfo.b = -projectionInfo.t;
    projectionInfo.r = m_aspect * projectionInfo.t;
    projectionInfo.l = -projectionInfo.r;

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(m_fov), m_aspect, NEAR_PLANE_DISTANCE, FAR_PLANE_DISTANCE);
    projectionInfo.projectionXX = projectionMatrix[0][0];
    projectionInfo.projectionYY = projectionMatrix[1][1];
    projectionInfo.projectionZZ = projectionMatrix[2][2];
    projectionInfo.projectionZW = projectionMatrix[3][2];
    return projectionInfo;
}

glm::mat4 Camera::getInverseView() const
{
    glm::mat4 view = glm::lookAt(m_position, m_position + m_front, m_up);
    return glm::inverse(view);
}
