# version 330 core

uniform mat4 proj;
uniform float particleRadius;

in vec4 viewPosition;

out vec4 FragColor;

void main()
{
    vec3 viewSpaceSphereNormal;
    viewSpaceSphereNormal.xy = 2 * gl_PointCoord.xy - 1;
    float radius2 = dot(viewSpaceSphereNormal.xy, viewSpaceSphereNormal.xy);
    if (radius2 > 1.0f)
    {
        discard;
    }
    viewSpaceSphereNormal.z = sqrt(1 - radius2);

    float linearDepth = viewPosition.z + viewSpaceSphereNormal.z * particleRadius;
    FragColor.r = -linearDepth;
}