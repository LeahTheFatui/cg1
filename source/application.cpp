#include "application.hpp"

#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace application {
namespace {
constexpr float PI = 3.14159265358979323846f;

struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };
struct Mat4 { float m[16]{}; }; // column-major

Mat4 identity() {
    Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1.0f; return r;
}
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c=0;c<4;++c) for (int row=0;row<4;++row)
        for (int k=0;k<4;++k) r.m[c*4+row] += a.m[k*4+row]*b.m[c*4+k];
    return r;
}
Mat4 translate(float x,float y,float z) {
    Mat4 r=identity(); r.m[12]=x; r.m[13]=y; r.m[14]=z; return r;
}
Mat4 scale(float x,float y,float z) {
    Mat4 r=identity(); r.m[0]=x; r.m[5]=y; r.m[10]=z; return r;
}
Mat4 rotateX(float a) {
    Mat4 r=identity(); float c=std::cos(a),s=std::sin(a);
    r.m[5]=c; r.m[9]=-s; r.m[6]=s; r.m[10]=c; return r;
}
Mat4 rotateY(float a) {
    Mat4 r=identity(); float c=std::cos(a),s=std::sin(a);
    r.m[0]=c; r.m[8]=s; r.m[2]=-s; r.m[10]=c; return r;
}
Mat4 rotateZ(float a) {
    Mat4 r=identity(); float c=std::cos(a),s=std::sin(a);
    r.m[0]=c; r.m[4]=-s; r.m[1]=s; r.m[5]=c; return r;
}
Mat4 perspective(float fovy,float aspect,float nearZ,float farZ) {
    Mat4 r{}; const float f=1.0f/std::tan(fovy*0.5f);
    r.m[0]=f/aspect; r.m[5]=-f; // Vulkan framebuffer Y direction
    r.m[10]=farZ/(nearZ-farZ); r.m[11]=-1.0f;
    r.m[14]=(farZ*nearZ)/(nearZ-farZ);
    return r;
}
Mat4 orthographic(float height,float aspect,float nearZ,float farZ) {
    Mat4 r=identity(); const float width=height*aspect;
    r.m[0]=2.0f/width; r.m[5]=-2.0f/height;
    r.m[10]=1.0f/(nearZ-farZ); r.m[14]=nearZ/(nearZ-farZ);
    return r;
}
Vec3 normalize(Vec3 v) {
    float l=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
    return {v.x/l,v.y/l,v.z/l};
}
Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f=normalize({center.x-eye.x,center.y-eye.y,center.z-eye.z});
    Vec3 s=normalize({f.y*up.z-f.z*up.y,f.z*up.x-f.x*up.z,f.x*up.y-f.y*up.x});
    Vec3 u={s.y*f.z-s.z*f.y,s.z*f.x-s.x*f.z,s.x*f.y-s.y*f.x};
    Mat4 r=identity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;
    r.m[2]=-f.x; r.m[6]=-f.y; r.m[10]=-f.z;
    r.m[12]=-(s.x*eye.x+s.y*eye.y+s.z*eye.z);
    r.m[13]=-(u.x*eye.x+u.y*eye.y+u.z*eye.z);
    r.m[14]=(f.x*eye.x+f.y*eye.y+f.z*eye.z);
    return r;
}
struct Vertex { float pos[3]; float color[3]; };
struct Uniform { Mat4 mvp; Vec4 tint; };
struct Object {
    Vec3 position{0,0,0}, rotation{0,0,0}, size{1,1,1};
    float tint[4]{0.22f,0.62f,0.95f,1.0f};
    VkBuffer uniformBuffer=VK_NULL_HANDLE;
    VmaAllocation uniformAllocation=VK_NULL_HANDLE;
    void* mapped=nullptr;
    VkDescriptorSet descriptor=VK_NULL_HANDLE;
};
struct Renderer {
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout=VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout=VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool=VK_NULL_HANDLE;
    VkBuffer vertexBuffer=VK_NULL_HANDLE;
    VmaAllocation vertexAllocation=VK_NULL_HANDLE;
    void* vertexMapped=nullptr;
    VkBuffer indexBuffer=VK_NULL_HANDLE;
    VmaAllocation indexAllocation=VK_NULL_HANDLE;
    void* indexMapped=nullptr;
    uint32_t indexCount=0;
    bool initialized=false;
} g;
Object objects[2];
bool usePerspective=true, animate=true, showSecond=true, wireframe=false;
float speed=1.0f, radius=1.4f, orthoHeight=4.5f;
float clearColor[4]{0.075f,0.085f,0.11f,1.0f};
double previousTime=0.0;
float animationTime=0.0f;

bool check(VkResult r, const char* what) {
    if (r==VK_SUCCESS) return true;
    std::cerr << what << " failed, VkResult=" << int(r) << '\n'; return false;
}
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer,
                  VmaAllocation& allocation, void** mapped) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size=size; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{}; ai.usage=VMA_MEMORY_USAGE_AUTO;
    ai.flags=VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
             VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(graphics::internal::context.allocator,&bi,&ai,&buffer,&allocation,&info)!=VK_SUCCESS)
        return false;
    if (mapped) *mapped=info.pMappedData;
    return true;
}
std::vector<Vertex> makeVertices() {
    // Eight corners, expanded per face so each face can have a distinct procedural color.
    const Vec3 p[8]={{-0.5f,-0.5f,-0.5f},{0.5f,-0.5f,-0.5f},
      {0.5f,0.5f,-0.5f},{-0.5f,0.5f,-0.5f},{-0.5f,-0.5f,0.5f},
      {0.5f,-0.5f,0.5f},{0.5f,0.5f,0.5f},{-0.5f,0.5f,0.5f}};
    const int faces[6][4]={{0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}};
    std::vector<Vertex> out;
    for (int f=0;f<6;++f) for (int j : {0,1,2,0,2,3}) {
        Vec3 v=p[faces[f][j]];
        // Vertex colors are generated from local coordinates (RGB mapped from [-.5,.5] to [0,1]).
        out.push_back({{v.x,v.y,v.z},{v.x+0.5f,v.y+0.5f,v.z+0.5f}});
    }
    return out;
}
bool loadShader(const char* path, VkShaderModule& module) {
    FILE* file=std::fopen(path,"rb");
    if (!file) { std::cerr<<"Cannot open shader: "<<path<<'\n'; return false; }
    std::fseek(file,0,SEEK_END); long n=std::ftell(file); std::rewind(file);
    if (n<=0 || n%4!=0) { std::fclose(file); return false; }
    std::vector<uint32_t> code(size_t(n)/4);
    bool ok=std::fread(code.data(),1,size_t(n),file)==size_t(n); std::fclose(file);
    if (!ok) return false;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize=size_t(n); ci.pCode=code.data();
    return check(vkCreateShaderModule(graphics::internal::context.device,&ci,nullptr,&module),"vkCreateShaderModule");
}
bool createRenderer() {
    auto& c=graphics::internal::context;
    auto vertices=makeVertices(); g.indexCount=uint32_t(vertices.size());
    if (!createBuffer(sizeof(Vertex)*vertices.size(),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,g.vertexBuffer,g.vertexAllocation,&g.vertexMapped)) return false;
    std::memcpy(g.vertexMapped,vertices.data(),sizeof(Vertex)*vertices.size());
    // Non-indexed draw: vertex count is 36. Keep a tiny index buffer unnecessary.
    g.indexCount=uint32_t(vertices.size());
    if (!createBuffer(sizeof(Uniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,objects[0].uniformBuffer,objects[0].uniformAllocation,&objects[0].mapped)) return false;
    if (!createBuffer(sizeof(Uniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,objects[1].uniformBuffer,objects[1].uniformAllocation,&objects[1].mapped)) return false;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding=0; binding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount=1; binding.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount=1; dl.pBindings=&binding;
    if (!check(vkCreateDescriptorSetLayout(c.device,&dl,nullptr,&g.descriptorLayout),"descriptor set layout")) return false;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets=2; dp.poolSizeCount=1; dp.pPoolSizes=&ps;
    if (!check(vkCreateDescriptorPool(c.device,&dp,nullptr,&g.descriptorPool),"descriptor pool")) return false;
    VkDescriptorSetLayout layouts[]={g.descriptorLayout,g.descriptorLayout};
    VkDescriptorSet sets[2]{};
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool=g.descriptorPool; da.descriptorSetCount=2; da.pSetLayouts=layouts;
    if (!check(vkAllocateDescriptorSets(c.device,&da,sets),"allocate descriptor sets")) return false;
    for (int i=0;i<2;++i) {
        objects[i].descriptor=sets[i];
        VkDescriptorBufferInfo db{objects[i].uniformBuffer,0,sizeof(Uniform)};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=sets[i]; w.dstBinding=0; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo=&db;
        vkUpdateDescriptorSets(c.device,1,&w,0,nullptr);
    }
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount=1; pl.pSetLayouts=&g.descriptorLayout;
    if (!check(vkCreatePipelineLayout(c.device,&pl,nullptr,&g.pipelineLayout),"pipeline layout")) return false;

    VkShaderModule vs=VK_NULL_HANDLE, fs=VK_NULL_HANDLE;
    if (!loadShader("shaders/parallelepiped.vert.spv",vs) || !loadShader("shaders/parallelepiped.frag.spv",fs)) {
        if (vs) vkDestroyShaderModule(c.device,vs,nullptr);
        if (fs) vkDestroyShaderModule(c.device,fs,nullptr);
        std::cerr<<"Compile shaders with glslc (see README.md).\n"; return false;
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vs; stages[0].pName="main";
    stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=fs; stages[1].pName="main";
    VkVertexInputBindingDescription vb{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[2]={{0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,pos)},{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color)}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&vb; vi.vertexAttributeDescriptionCount=2; vi.pVertexAttributeDescriptions=attrs;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount=1; vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=1; blend.pAttachments=&cb;
    VkDynamicState dyns[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dyn.dynamicStateCount=2; dyn.pDynamicStates=dyns;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.stageCount=2; pi.pStages=stages; pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia; pi.pViewportState=&vp;
    pi.pRasterizationState=&rs; pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds; pi.pColorBlendState=&blend;
    pi.pDynamicState=&dyn; pi.layout=g.pipelineLayout; pi.renderPass=c.render_pass; pi.subpass=0;
    VkResult result=vkCreateGraphicsPipelines(c.device,VK_NULL_HANDLE,1,&pi,nullptr,&g.pipeline);
    vkDestroyShaderModule(c.device,vs,nullptr); vkDestroyShaderModule(c.device,fs,nullptr);
    if (!check(result,"graphics pipeline")) return false;
    g.initialized=true;
    objects[1].position={1.45f,0.0f,0.0f};
    objects[1].size={0.65f,0.65f,0.65f};
    objects[1].tint[0]=0.95f; objects[1].tint[1]=0.36f; objects[1].tint[2]=0.20f;
    return true;
}
void destroyRenderer() {
    auto& c=graphics::internal::context;
    if (g.pipeline) vkDestroyPipeline(c.device,g.pipeline,nullptr);
    if (g.pipelineLayout) vkDestroyPipelineLayout(c.device,g.pipelineLayout,nullptr);
    if (g.descriptorPool) vkDestroyDescriptorPool(c.device,g.descriptorPool,nullptr);
    if (g.descriptorLayout) vkDestroyDescriptorSetLayout(c.device,g.descriptorLayout,nullptr);
    for (auto& o:objects) if (o.uniformBuffer) vmaDestroyBuffer(c.allocator,o.uniformBuffer,o.uniformAllocation);
    if (g.vertexBuffer) vmaDestroyBuffer(c.allocator,g.vertexBuffer,g.vertexAllocation);
    g={};
}
Mat4 modelMatrix(const Object& o) {
    return multiply(translate(o.position.x,o.position.y,o.position.z),
           multiply(rotateZ(o.rotation.z),multiply(rotateY(o.rotation.y),
           multiply(rotateX(o.rotation.x),scale(o.size.x,o.size.y,o.size.z)))));
}
void updateUniforms(double time) {
    auto& c=graphics::internal::context;
    const float aspect=float(c.swapchain_extent.width)/float(std::max(1u,c.swapchain_extent.height));
    Mat4 proj=usePerspective?perspective(55.0f*PI/180.0f,aspect,0.1f,100.0f):orthographic(orthoHeight,aspect,0.1f,100.0f);
    Mat4 view=lookAt({0,0,5.5f},{0,0,0},{0,1,0});
    for (int i=0;i<2;++i) {
        Object& o=objects[i];
        Object animated=o;
        if (i==0 && animate) {
            float t=animationTime;
            animated.position.x=o.position.x+radius*std::cos(t);
            animated.position.y=o.position.y+0.45f*std::sin(2.0f*t);
            animated.position.z=o.position.z+0.35f*std::sin(t);
            animated.rotation.y+=t*0.65f; animated.rotation.x+=0.25f*std::sin(t);
        }
        Uniform u{}; u.mvp=multiply(proj,multiply(view,modelMatrix(animated)));
        u.tint={o.tint[0],o.tint[1],o.tint[2],o.tint[3]};
        std::memcpy(o.mapped,&u,sizeof(u));
    }
    (void)time;
}
} // namespace

bool initialize() {
    objects[0].position={-0.6f,0.0f,0.0f};
    if (!createRenderer()) { destroyRenderer(); return false; }
    return true;
}
void shutdown() {
    auto& c=graphics::internal::context;
    if (c.device) vkDeviceWaitIdle(c.device);
    destroyRenderer();
}
void update(double time) {
    if (previousTime==0.0) previousTime=time;
    float dt=float(std::clamp(time-previousTime,0.0,0.1)); previousTime=time;
    if (animate) animationTime+=dt*speed;

    ImGui::Begin("Lab work N1 -- 3D-graphics");
    ImGui::Text("Task 3: parallelepiped");
    ImGui::Separator();
    const char* projections[]={"Perspective","Ortographic"};
    int projection=usePerspective?0:1;
    if (ImGui::Combo("Projection",&projection,projections,2)) usePerspective=(projection==0);
    if (!usePerspective) ImGui::SliderFloat("Ortographic height",&orthoHeight,2.0f,10.0f);
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Transfiguring main object",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("POSITION XYZ",&objects[0].position.x,-3.0f,3.0f);
        ImGui::SliderFloat3("ROTATION XYZ",&objects[0].rotation.x,-PI,PI,"%.2f рад");
        ImGui::SliderFloat3("SCALE XYZ",&objects[0].size.x,0.1f,2.5f);
    }
    ImGui::Separator();
    if (ImGui::Button(animate?"Pause anim":"Continue anim")) animate=!animate;
    ImGui::SliderFloat("Speed",&speed,0.0f,3.0f);
    ImGui::SliderFloat("Tragectory raduis",&radius,0.0f,2.5f);
    ImGui::ColorEdit3("Main object color",objects[0].tint);
    ImGui::Separator();
    ImGui::Checkbox("Show secondary object", &showSecond);
    ImGui::ColorEdit3("Secondary object color",objects[1].tint);
    ImGui::SliderFloat3("Secondary objects' position",&objects[1].position.x,-3.0f,3.0f);
    ImGui::Text("Verticies colors are multiplied on selected color");
    ImGui::End();
    updateUniforms(time);
}
void render(const graphics::internal::FrameData& fd) {
    if (!g.initialized || fd.command_buffer==VK_NULL_HANDLE) return;
    auto& c=graphics::internal::context;
    VkCommandBuffer cmd=fd.command_buffer;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd,&begin)!=VK_SUCCESS) return;
    VkClearValue clears[2]{};
    clears[0].color={{clearColor[0],clearColor[1],clearColor[2],clearColor[3]}};
    clears[1].depthStencil={1.0f,0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass=c.render_pass; rp.framebuffer=fd.framebuffer;
    rp.renderArea.extent=c.swapchain_extent; rp.clearValueCount=2; rp.pClearValues=clears;
    vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0.0f,0.0f,float(c.swapchain_extent.width),float(c.swapchain_extent.height),0.0f,1.0f};
    VkRect2D scissor{{0,0},c.swapchain_extent};
    vkCmdSetViewport(cmd,0,1,&viewport); vkCmdSetScissor(cmd,0,1,&scissor);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,g.pipeline);
    VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(cmd,0,1,&g.vertexBuffer,&offset);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,g.pipelineLayout,0,1,&objects[0].descriptor,0,nullptr);
    vkCmdDraw(cmd,g.indexCount,1,0,0);
    if (showSecond) {
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,g.pipelineLayout,0,1,&objects[1].descriptor,0,nullptr);
        vkCmdDraw(cmd,g.indexCount,1,0,0);
    }
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}
} // namespace application
