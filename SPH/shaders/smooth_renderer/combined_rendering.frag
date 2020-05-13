# version 330 core

in vec2 ScreenCoordinates;

uniform float inverseProjectionXX;
uniform float inverseProjectionYY;
uniform float projectionZZ;
uniform float projectionZW;

uniform mat4 inverseView;

uniform float f_0;
uniform float fluidRefractionIndex;

uniform sampler2D depthTexture;
uniform sampler2D normalsTexture;
uniform sampler2D thicknessTexture;
uniform samplerCube skyboxTexture;

uniform bool change;

out vec4 FragColor;

const float airRefractionIndex = 1.0f;

float ProjectDepth(float linearDepth)
{
    // See formula (3): http://www.songho.ca/opengl/gl_projectionmatrix.html
    return -(projectionZZ + projectionZW / linearDepth);
}

float ConvertCoorinateDeviceToScreen(float coordinate)
{
    return 0.5f * (coordinate + 1.0f);
}

// Maps x, y from [0, 1] to [-1, 1]
vec2 ConvertCoordinatesScreenToDevice(vec2 coordinates)
{
    return 2 * coordinates - 1;
}

vec2 ConvertCoordinatesDeviceToView(vec2 coordinates, float linearDepth)
{
    coordinates.x *= linearDepth / inverseProjectionXX;
    coordinates.y *= linearDepth / inverseProjectionYY;
    return coordinates;
}

vec2 ConvertCoordinatesScreenToView(vec2 coordinates, float linearDepth)
{
    return ConvertCoordinatesDeviceToView(ConvertCoordinatesScreenToDevice(coordinates), linearDepth);
}

vec3 GetViewSpaceFragmentCoordinates()
{
    float linearDepth = texture(depthTexture, ScreenCoordinates).x;
    vec3 viewCoordinates;
    viewCoordinates.xy = ConvertCoordinatesScreenToView(ScreenCoordinates, linearDepth);
    viewCoordinates.z = -linearDepth; // Convert from left-hand to right-hand coordinate system
    return viewCoordinates;
}

vec3 GetSkyboxColor(vec3 viewDirection)
{
	vec3 worldDirection = mat3(inverseView) * viewDirection;
    return texture(skyboxTexture, worldDirection).rgb;
}

/*
    f_0 - base reflectance
    cosTheta - angle between fargment normal and direction to view
*/
float FreshnelFunction(float f_0, float cosTheta)
{
    return f_0 + (1 - f_0) * pow(1 - cosTheta, 5);
}

vec4 CalculateColor()
{
	vec3 viewFragmentCoordinates = GetViewSpaceFragmentCoordinates();
    vec3 directionToView = normalize(-viewFragmentCoordinates);
    vec3 normal = texture(normalsTexture, ScreenCoordinates).xyz;

    // Find reflected color from skybox
	vec3 reflectedDirection = reflect(-directionToView, normal);
    vec3 reflectedColor = GetSkyboxColor(reflectedDirection);

    // Find refracted color from skybox
	float thickness = texture(thicknessTexture, ScreenCoordinates).x;
    vec3 refractedDirection = refract(-directionToView, normal, airRefractionIndex / fluidRefractionIndex);
    vec3 refractedColor = GetSkyboxColor(refractedDirection);

    vec3 fluidColor = normalize(vec3(116,204,244));
    vec3 attenuation = max(exp(-vec3(0.3f) * thickness), 0.2f);
	refractedColor = mix(fluidColor, refractedColor, attenuation);

    float reflectedPart = FreshnelFunction(f_0, dot(normal, directionToView));
	return vec4(mix(refractedColor, reflectedColor, reflectedPart), 1);
}

void main()
{
	float linearDepth = texture(depthTexture, ScreenCoordinates).x;
	float projectedDepth = ProjectDepth(-linearDepth);
    gl_FragDepth = ConvertCoorinateDeviceToScreen(projectedDepth);

    FragColor = CalculateColor();
}