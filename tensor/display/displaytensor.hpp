
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "display/materials/shader.hpp"
#include "display/materials/overlay.hpp"
#include "display/drawable/drawable.hpp"
#ifndef DISPLAYTENSOR_HPP
#define DISPLAYTENSOR_HPP

__weak RenderStruct<float32x2,float32x2>* cached_rect = nullptr;

__weak RenderStruct<float32x2,float32x2>* get_overlay_rect(){
    if (cached_rect == nullptr){
        cached_rect = new RenderStruct<float32x2,float32x2>(Shape{4});
        cached_rect->view<Hvec<float,16>,1>({1}) = Hvec<float,16>{-1, -1, 0, 1, 
                                                        1, -1, 1, 1, 
                                                        1, 1, 1, 0,
                                                        -1, 1, 0, 0};
        cached_rect->primitive_type = GL_TRIANGLE_FAN;
    }

    return cached_rect;
}




template <typename bufftype>
class VectorDisplay: public Tensor<bufftype,2>
{
    using Tensor<bufftype,2>::Tensor; // inherit constructors
    public:
    RenderStruct<float32x2,float32x2> rect;
    bool setup = false;
    GLuint framebuffer = 0; // if used as target for rendering
    GLuint depthbuffer = 0; // shared depth buffer for rendering

    VectorDisplay() : Tensor<bufftype,2>() {}

  
    VectorDisplay(Shape<2> shape, ComputeType compute_type = kOPENGL) : Tensor<bufftype,2>(shape, global_device_manager.get_compute_device(kOPENGL).default_memory_type, compute_type)
    {
    }
    

    VectorDisplay(AllocationMetadata m):Tensor<bufftype,2>(m){};


    void initialize_textures() {
        // This function can be used to initialize OpenGL textures if needed
        rect = *get_overlay_rect();
            if(this->storage_pointer->metadata.compute_device == ComputeType::kOPENGL){
                // Create buffer texture that references the buffer
                GLuint bufferTexture = 0;  // Buffer texture handle
                glGenTextures(1, &bufferTexture);
                glBindTexture(GL_TEXTURE_BUFFER, bufferTexture);

                size_t bytesize = this->bitsize;
                GLenum internalFormat;
                if(bytesize == 8){ //fp16x4
                    internalFormat = GL_RGBA16F;
                } else if(bytesize == 4){ //uint8x4
                    internalFormat = GL_RGBA8;
                } else if(bytesize == 6){ //fp16x3
                    internalFormat = GL_RGB16F;
                } else if(bytesize == 3){ //uint8x3
                    internalFormat = GL_RGB8;
                } else {
                    throw std::runtime_error("Unsupported buffer type for OpenGL display");
                }

                glTexBuffer(GL_TEXTURE_BUFFER, internalFormat, (GLuint)(size_t)this->storage_pointer->data);
                
                auto glErr = glGetError();
                if (glErr != GL_NO_ERROR) {
                    throw std::runtime_error("OpenGL error creating buffer texture: " + std::to_string(glErr));
                }
                
                glBindTexture(GL_TEXTURE_BUFFER, 0);
                glFinish();

                rect.material = new Shader<OverLayShader<true>>();
                rect.material->textures_ids["bufferTex"] = bufferTexture;
                rect.material->texture_types["bufferTex"] = GL_TEXTURE_BUFFER;
                rect.material->transparent = true;


            }else if (this->storage_pointer->metadata.compute_device == ComputeType::kOPENGLTEXTURE){

                rect.material = new Shader<OverLayShader<false>>();    
                rect.material->textures_ids["bufferTex"] = (GLuint)((size_t)this->storage_pointer->data);
                rect.material->texture_types["bufferTex"] = GL_TEXTURE_2D;
                rect.material->transparent = true;
            } else {
                throw ("Unsupported device");
            }
            setup = true;
        }

    void attach_depth_buffer(Tensor<float, 2> depthbuffer){
        if (depthbuffer.storage_pointer->data == 0){
            throw std::runtime_error("Invalid depth buffer provided");
        }
        this->depthbuffer = (GLuint)(size_t)depthbuffer.storage_pointer->data;
    }

    void initialize_framebuffer() {
        if (!setup) {
            initialize_textures();
        }


        glGenFramebuffers(1, &framebuffer);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        
        GLuint solidTexID = (GLuint)(size_t)this->storage_pointer->data;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, solidTexID, 0);
        
        // Attach SHARED depth buffer
        if (depthbuffer != 0) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer);
            // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthbuffer, 0);
        }
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Solid framebuffer not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }


    void present() {
        glViewport(0, 0, int(this->shape[0]), int(this->shape[1]));

        if(!setup){
            initialize_textures();
        }
        // Don't clear here - we're displaying, not rendering
        // glClear(GL_COLOR_BUFFER_BIT);
        rect.bind();
        
        glUniform2i(glGetUniformLocation(rect.material->shader_program, "dimensions"), int(this->shape[0]), int(this->shape[1]));
        

        rect.draw();

    }
    

    void bind_as_render_target() {
        if (framebuffer == 0){
            initialize_framebuffer();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }

    void unbind_render_target() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }


   
};


#endif