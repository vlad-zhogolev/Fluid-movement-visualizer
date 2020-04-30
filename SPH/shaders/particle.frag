# version 330 core

uniform vec4 color;
uniform uint hlIndex;
flat in uint iid;
out vec4 FragColor;

void OneColorShading()
{
    vec3 lightDir = normalize(vec3(1, -1, 1));
    float x = 2 * gl_PointCoord.x - 1;
    float y = 2 * gl_PointCoord.y - 1;
    float pho = x * x + y * y;
    float z = sqrt(1 - pho);
    if (pho > 1)
    {
        discard;
    }
    vec3 blueColor = vec3(0, 0, 1);
    FragColor = vec4(dot(lightDir, vec3(x, y, z)) * blueColor, 1);
}

void main()
{
    OneColorShading();	
}