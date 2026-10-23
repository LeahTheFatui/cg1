#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 mvp;
    vec4 tint;
} objectData;
layout(location = 0) out vec3 vertexColor;
void main() {
    gl_Position = objectData.mvp * vec4(inPosition, 1.0);
    vertexColor = inColor * objectData.tint.rgb;
}
