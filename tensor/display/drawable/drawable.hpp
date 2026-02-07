#ifndef DRAWABLE_HPP
#define DRAWABLE_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include "tensor.hpp"
#include "vector/vectors.hpp"
// #include "../vector/uint84.hpp"
#include "file_loaders/gltf.hpp"
#include "display/display.hpp"
#include "ops/ops.hpp"
#include "display/materials/materials.hpp"


template <typename... vertex_types>
struct RenderStruct : Tensor<mytuple<vertex_types...>, 1>
{
    
    GLenum primitive_type = GL_POINTS;
    mat4 model_matrix = mat4::identity();
    Material* material = nullptr;
    Skeleton bone_matrices;
    GLuint VAO;
    Tensor<int, 1> indices;
    int offset = 0;
    int count = -1;

    RenderStruct() : Tensor<mytuple<vertex_types...>, 1>() {}

    void setupVAO()
    {
        GLFuncs->glGenVertexArrays(1, &VAO);
        GLFuncs->glBindVertexArray(VAO);
        if (indices.data != nullptr){
            GLFuncs->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (unsigned long long)indices.storage_pointer);
        }
        GLFuncs->glBindBuffer(GL_ARRAY_BUFFER, (unsigned long long)this->storage_pointer);
        long offset = 0;
        for (int i = 0; i < VertexLayout<vertex_types...>::num_attributes; i++)
        {
            std::cout << "Setting up attribute " << i << ": size=" << VertexLayout<vertex_types...>::sizes[i] << ", type=" << VertexLayout<vertex_types...>::types[i] << ", normalized=" << VertexLayout<vertex_types...>::normalized[i] << ", attribute_size=" << VertexLayout<vertex_types...>::attribute_sizes[i] << std::endl;
            GLFuncs->glEnableVertexAttribArray(i);
            if (VertexLayout<vertex_types...>::types[i] == GL_INT)
            {
                GLFuncs->glVertexAttribIPointer(
                    i,
                    VertexLayout<vertex_types...>::sizes[i],
                    VertexLayout<vertex_types...>::types[i],
                    sizeof(VertexLayout<vertex_types...>),
                    (void*)(unsigned long)offset);
            }
            else
            {
                GLFuncs->glVertexAttribPointer(
                    i,
                    VertexLayout<vertex_types...>::sizes[i],
                    VertexLayout<vertex_types...>::types[i],
                    VertexLayout<vertex_types...>::normalized[i],
                    sizeof(VertexLayout<vertex_types...>),
                    (void*)(unsigned long)offset);
            };
            offset += VertexLayout<vertex_types...>::attribute_sizes[i];
        }
        GLFuncs->glBindBuffer(GL_ARRAY_BUFFER, 0);
        GLFuncs->glBindVertexArray(0);
    }

    RenderStruct(Tensor<vertex_types, 1>... tensors) : Tensor<mytuple<vertex_types...>, 1>(
        // shape of first tensor
        std::get<0>(std::make_tuple(tensors...)).shape,
        global_device_manager.get_compute_device(kOPENGL).default_memory_type,
        kOPENGL)
    {

        setupVAO();
        // copy data from tensors into this tensor
        CopyDataHelper<vertex_types...>::run(tensors.to(kCUDA_VRAM)..., this->to_compute(kCUDA));
    }

    RenderStruct(
        Shape<1> shape
    ) : Tensor<mytuple<vertex_types...>, 1>(
            shape,
            global_device_manager.get_compute_device(kOPENGL).default_memory_type,
            kOPENGL)
    {
        setupVAO();
    }

    RenderStruct(Skeleton bones, Tensor<int, 1> inindices, Tensor<vertex_types, 1>... inputs) : Tensor<mytuple<vertex_types...>, 1>(
        std::get<0>(std::make_tuple(inputs.shape...)), global_device_manager.get_compute_device(kOPENGL).default_memory_type, kOPENGL), bone_matrices(bones)
    {
        indices = inindices.to(global_device_manager.get_compute_device(kOPENGL).default_memory_type, kOPENGL);
        setupVAO();

        // indices = Tensor<int, 1>(inindices.shape, global_device_manager.get_compute_device(kOPENGL).default_memory_type, kOPENGL);
        std::cout << "Index storage pointer: " << indices << std::endl;
        this->device->synchronize_function();


        

        CopyDataHelper<vertex_types...>::run(inputs.to(kCUDA_VRAM)..., this->to_compute(kCUDA));

        this->primitive_type = GL_TRIANGLES;
    }

    // RenderStruct(
    //     const RenderStruct<vertex_types...>& other
    // ) :  Tensor<mytuple<vertex_types...>, 1>(other), primitive_type(other.primitive_type), model_matrix(other.model_matrix), material(other.material), bone_matrices(other.bone_matrices), indices(other.indices)
    // {
    //     setupVAO();
    // }

    void draw() const
    {
        if (material == nullptr)
        {
            std::cerr << "No material assigned to RenderStruct, cannot draw!" << std::endl;
            return;
        }

        GLFuncs->glBindVertexArray(VAO);
        GLFuncs->glBindBuffer(GL_ARRAY_BUFFER, (unsigned long long)this->storage_pointer);
        // set bone matrices
        if (bone_matrices.data != nullptr){
            GLint bonesLoc = GLFuncs->glGetUniformLocation(material->shader_program, "bone_matrices");
            GLFuncs->glUniformMatrix4fv(bonesLoc, bone_matrices.shape.A, GL_TRUE, (float *)(void *)bone_matrices.data); // Assuming 100 bones for simplicity
        }
        // Set double-sided rendering if needed
        if (material->double_sided)
        {
            glDisable(GL_CULL_FACE);
        }
        else
        {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }


        if (indices.data != nullptr)
        {
            GLFuncs->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (unsigned long long)indices.storage_pointer);
            glDrawElements(primitive_type, indices.shape[0], GL_UNSIGNED_INT,0);
        }
        else
        {
            // if offset and count are set, use them
            if (count > 0){
                glDrawArrays(primitive_type, offset, count);
                return;
            }
            glDrawArrays(primitive_type, 0, this->shape[0]);
        }
        GLFuncs->glBindVertexArray(0);
    }

    void bind()
    {
        if (material == nullptr)
        {
            std::cerr << "No material assigned to RenderStruct, cannot bind!" << std::endl;
            return;
        }
        material->bind();
        model_matrix.bind(material->shader_program, "model");
    }
};

#endif