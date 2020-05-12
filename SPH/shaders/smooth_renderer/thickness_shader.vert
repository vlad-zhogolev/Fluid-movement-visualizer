# version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 proj;
uniform mat4 view;
uniform int windowHeight;
uniform float projectionTop;
uniform float projectionNear;
uniform float particleRadius; // TODO: describe more detailed

out vec4 viewPosition;

void main()
{
	viewPosition = view * vec4(aPos, 1.0);
	gl_Position = proj * viewPosition;
    // Determines side length of "particle" quad in pixels (particle diameter)
	gl_PointSize = particleRadius * projectionNear * windowHeight / (-viewPosition.z * projectionTop);
}