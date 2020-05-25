# version 330 core

uniform mat4 proj;
uniform float particleRadius;

in vec4 viewPosition;

out vec4 FragColor;

void main()
{
    vec3 viewSpaceSphereNormal;
    viewSpaceSphereNormal.xy = 2 * gl_PointCoord.xy - 1;
    float r2 = dot(viewSpaceSphereNormal.xy, viewSpaceSphereNormal.xy);
    if (r2 > 1.0f)
    {
        discard;
    }
    viewSpaceSphereNormal.z = sqrt(1 - r2);

    //vec4 pixelPosition = vec4(viewPosition.xyz + viewSpaceSphereNormal * particleRadius, 1);
    //FragColor.r = -pixelPosition.z;  // Store linear depth in texture
    // It's like having a directional light looking in z-axis;
    FragColor.r = 2 * particleRadius * viewSpaceSphereNormal.z;
}