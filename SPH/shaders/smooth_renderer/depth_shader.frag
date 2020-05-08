# version 330 core

uniform mat4 proj;
uniform float r;

in vec4 viewPosition;
in vec4 projectedPosition;

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

    vec4 pixelPosition = vec4(viewPosition.xyz + viewSpaceSphereNormal * r, 1);
    vec4 clipSpacePosition = proj * pixelPosition;
    gl_FragDepth = clipSpacePosition.z / clipSpacePosition.w;
}