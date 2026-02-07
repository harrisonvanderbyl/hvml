
#ifndef MAT4_HPP
#define MAT4_HPP
#include "vector/simplevec.hpp"
#include "device/device.hpp"
struct mat4 : public simpleVec<float32x4, 4>
{
    using simpleVec<float32x4, 4>::simpleVec;


    mat4  __host__ __device__  transpose()
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    };

    void  __host__ __device__  set_rotate(float angle, float32x3 axis)
    {
        float c = cos(angle);
        float s = sin(angle);
        float t = 1 - c;

        data[0] = float32x4(
            t * axis[0] * axis[0] + c,
            t * axis[0] * axis[1] - s * axis[2],
            t * axis[0] * axis[2] + s * axis[1],
            0
        );

        data[1] = float32x4(
            t * axis[0] * axis[1] + s * axis[2],
            t * axis[1] * axis[1] + c,
            t * axis[1] * axis[2] - s * axis[0],
            0
        );

        data[2] = float32x4(
            t * axis[0] * axis[2] - s * axis[1],
            t * axis[1] * axis[2] + s * axis[0],
            t * axis[2] * axis[2] + c,
            0
        );

        data[3] = float32x4(0, 0, 0, 1);
    };

    void  __host__ __device__  rotate(float angle, float32x3 axis)
    {
        mat4 rotation;
        rotation.set_rotate(angle, axis);
        
        // Multiply the current matrix by the rotation matrix
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.data[i][j] = data[i][0] * rotation.data[0][j] +
                                    data[i][1] * rotation.data[1][j] +
                                    data[i][2] * rotation.data[2][j] +
                                    data[i][3] * rotation.data[3][j];
            }
        }
        *this = result;
    };

    void  __host__ __device__  point_towards_direction(float32x3 direction, float32x3 up = float32x3(0, 1, -1))
    {
        // Normalize the direction vector
        float32x3 dir = direction.normalized();
        
        // Calculate the right and up vectors
        // float32x3 up = float32x3(0, 1, -1); // Assuming Y is up
        float32x3 right = up.cross(dir).normalized();
        up = dir.cross(right); // Recalculate up to ensure orthogonality

        // Set the rotation matrix
        data[0] = float32x4(right[0], up[0], -dir[0], 0);
        data[1] = float32x4(right[1], up[1], -dir[1], 0);
        data[2] = float32x4(right[2], up[2], -dir[2], 0);
        data[3] = float32x4(0, 0, 0, 1);

        
        // Set the translation part to zero
        data[3][0] = -(right[0] * 0 + up[0] * 0
                        - dir[0] * 0);   
        data[3][1] = -(right[1] * 0 + up[1] * 0
                        - dir[1] * 0);
        data[3][2] = right[2] * 0 + up[2] * 0
                        - dir[2] * 0;
        data[3][3] = 1;


    };

    mat4  __host__ __device__  pointed_towards_direction(float32x3 direction, float32x3 up = float32x3(0, 1, -1))
    {
        mat4 change = mat4::identity(); // Create an identity matrix for rotation
        
        mat4 myinverse = this->inverse(); // Get the inverse of the current matrix
        float32x3 relativedir =(myinverse * float32x4(direction[0], direction[1], direction[2], 0.0f)).xyz(); // Transform the direction to the local space

        change.point_towards_direction(relativedir, up); // Set the rotation

        mat4 result =  *this * change; // Multiply the current matrix by the rotation matrix
        return result;
    };

    void  __host__ __device__  set_translate(float32x3 translation)
    {
        data[0] = float32x4(1, 0, 0, translation[0]);
        data[1] = float32x4(0, 1, 0, translation[1]);
        data[2] = float32x4(0, 0, 1, translation[2]);
        data[3] = float32x4(0, 0, 0, 1);

    };

    void  __host__ __device__  translate(float32x3 translation)
    {
        mat4 translation_matrix;
        translation_matrix.set_translate(translation);
        
        // Multiply the current matrix by the translation matrix
        mat4 result = *this * translation_matrix;
        *this = result;
    };

    void   __host__ __device__  set_scale(float32x3 scale)
    {
        data[0] = float32x4(scale[0], 0, 0, 0);
        data[1] = float32x4(0, scale[1], 0, 0);
        data[2] = float32x4(0, 0, scale[2], 0);
        data[3] = float32x4(0, 0, 0, 1);
    };

    void  __host__ __device__  scale(float32x3 scale)
    {
        mat4 scale_matrix;
        scale_matrix.set_scale(scale);
        
        // Multiply the current matrix by the scale matrix
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.data[i][j] = data[i][0] * scale_matrix.data[0][j] +
                                    data[i][1] * scale_matrix.data[1][j] +
                                    data[i][2] * scale_matrix.data[2][j] +
                                    data[i][3] * scale_matrix.data[3][j];
            }
        }
        *this = result;
    };


    static  __host__ __device__  mat4 identity()
    {
        return mat4(
            float32x4(1, 0, 0, 0),
            float32x4(0, 1, 0, 0),
            float32x4(0, 0, 1, 0),
            float32x4(0, 0, 0, 1)
        );
    };

    mat4  __host__ __device__  translated(float32x3 translation)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for translation
        change.set_translate(translation); // Set the translation

        mat4 result = change* *this; // Multiply the current matrix by the translation matrix
        return result;
    };

    mat4  __host__ __device__  scaled(float32x3 scale)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for scaling
        change.set_scale(scale); // Set the scale

        mat4 result = change * *this; // Multiply the current matrix by the scale matrix
        return result;
    };

    mat4  __host__ __device__  rotated(float angle, float32x3 axis)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for rotation
        change.set_rotate(angle, axis); // Set the rotation

        mat4 result = change * *this; // Multiply the current matrix by the rotation matrix
        return result;
    };


    // matmul
    mat4  __host__ __device__  operator*(mat4 other)
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.data[i][j] = data[i][0] * other.data[0][j] +
                                    data[i][1] * other.data[1][j] +
                                    data[i][2] * other.data[2][j] +
                                    data[i][3] * other.data[3][j];
            }
        }
        return result;
    };

    mat4  __host__ __device__  inverse()
    {
        // Using the adjugate method for 4x4 matrix inversion
        mat4 inv;
        float det = data[0][0] * (data[1][1] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                                 data[1][2] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) +
                                 data[1][3] * (data[2][1] * data[3][2] - data[2][2] * data[3][1])) -
                    data[0][1] * (data[1][0] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                                 data[1][2] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                                 data[1][3] * (data[2][0] * data[3][2] - data[2][2] * data[3][0])) +
                    data[0][2] * (data[1][0] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) -
                                 data[1][1] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                                 data[1][3] * (data[2][0] * data[3][1] - data[2][1] * data[3][0])) -
                    data[0][3] * (data[1][0] * (data[2][1] * data[3][2] - data[2][2] * data[3][1]) -
                                 data[1][1] *(data[2][0]*data[3][2]-data[2][2]*data[3][0])+
                                 data[1][2]*(data[2][0]*data[3][1]-data[2][1]*data[3][0]));
        
        if (det == 0)
        {
            // throw "Matrix is singular and cannot be inverted";
        }

        // Calculate the inverse using the adjugate method
        // This is a simplified version and may not be numerically stable for all matrices
        inv = {
            float32x4(
                (data[1][1] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                 data[1][2] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) +
                 data[1][3] * (data[2][1] * data[3][2] - data[2][2] * data[3][1])) / det,
                -(data[0][1] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                  data[0][2] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) +
                  data[0][3] * (data[2][1] * data[3][2] - data[2][2] * data[3][1])) / det,
                (data[0][1] * (data[1][2] * data[3][3] - data[1][3] * data[3][2]) -
                 data[0][2] * (data[1][1] * data[3][3] - data[1][3] * data[3][1]) +
                 data[0][3] * (data[1][1] * data[3][2] - data[1][2] * data[3][1])) / det,
                -(data[0][1] * (data[1][2] * data[2][3] - data[1][3] * data[2][2]) -
                  data[0][2] * (data[1][1] * data[2][3] - data[1][3] * data[2][1]) +
                  data[0][3] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])) / det
            ),
            float32x4(
                -(data[1][0] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                  data[1][2] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                  data[1][3] * (data[2][0] * data[3][2] - data[2][2] * data[3][0])) / det,
                (data[0][0] * (data[2][2] * data[3][3] - data[2][3] * data[3][2]) -
                 data[0][2] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                 data[0][3] * (data[2][0] * data[3][2] - data[2][2] * data[3][0])) / det,
                -(data[0][0] * (data[1][2] * data[3][3] - data[1][3] * data[3][2]) -
                  data[0][2] * (data[1][0] * data[3][3] - data[1][3] * data[3][0]) +
                  data[0][3] * (data[1][0] * data[3][2] - data[1][2] * data[3][0])) / det,
                (data[0][0] * (data[1][2] * data[2][3] - data[1][3] * data[2][2]) -
                 data[0][2] * (data[1][0] * data[2][3] - data[1][3] * data[2][0]) +
                 data[0][3] * (data[1][0] * data[2][2] - data[1][2] * data[2][1])) / det
            ),
            float32x4(
                (data[1][0] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) -
                 data[1][1] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                 data[1][3] * (data[2][0] * data[3][1] - data[2][1] * data[3][0])) / det,
                -(data[0][0] * (data[2][1] * data[3][3] - data[2][3] * data[3][1]) -
                  data[0][1] * (data[2][0] * data[3][3] - data[2][3] * data[3][0]) +
                  data[0][3] * (data[2][0] * data[3][1] - data[2][1] * data[3][0])) / det,
                (data[0][0] * (data[1][1] * data[3][3] - data[1][3] * data[3][1]) -
                 data[0][1] * (data[1][0] * data[3][3] - data[1][3] * data[3][0]) +
                 data[0][3] * (data[1][0] * data[3][1] - data[1][1] * data[3][0])) / det,
                -(data[0][0] * (data[1][1] * data[2][3] - data[1][3] * data[2][1]) -
                  data[0][1] * (data[1][0] * data[2][3] - data[1][3] * data[2][0]) +
                  data[0][3] * (data[1][0] * data[2][1] - data[1][1] * data[2][0])) / det
            ),
            float32x4(
                -(data[1][0] * (data[2][1] * data[3][2] - data[2][2] * data[3][1]) -
                  data[1][1] * (data[2][0] * data[3][2] - data[2][2] * data[3][0]) +
                  data[1][2] * (data[2][0] * data[3][1] - data[2][1] * data[3][0])) / det,
                (data[0][0] * (data[2][1] * data[3][2] - data[2][2] * data[3][1]) -
                 data[0][1] * (data[2][0] * data[3][2] - data[2][2] * data[3][0]) +
                 data[0][2] * (data[2][0] * data[3][1] - data[2][1] * data[3][0])) / det,
                -(data[0][0] * (data[1][1] * data[3][2] - data[1][2] * data[3][1]) - 
                  data[0][1] * (data[1][0] * data[3][2] - data[1][2] * data[3][0]) +
                  data[0][2] * (data[1][0] * data[3][1] - data[1][1] * data[3][0])) / det,
                (data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1]) -
                 data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0]) +
                 data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0])) / det
            )
        };

        return inv;
    };

    float32x4  __host__ __device__  operator * (float32x4 vec)
    {
        float32x4 result;
        result[0] = data[0][0] * vec[0] + data[0][1] * vec[1] + data[0][2] * vec[2] + data[0][3] * vec[3];
        result[1] = data[1][0] * vec[0] + data[1][1] * vec[1] + data[1][2] * vec[2] + data[1][3] * vec[3];
        result[2] = data[2][0] * vec[0] + data[2][1] * vec[1] + data[2][2] * vec[2] + data[2][3] * vec[3];
        result[3] = data[3][0] * vec[0] + data[3][1] * vec[1] + data[3][2] * vec[2] + data[3][3] * vec[3];
        return result;
    };

    float32x3 __host__ __device__ operator * (float32x3 vec)
    {
        float32x4 result = *this * float32x4(vec[0], vec[1], vec[2], 1.0f);
        return float32x3(result[0], result[1], result[2]);
    };

    // print
    friend std::ostream &operator<<(std::ostream &os, mat4 a)
    {
        os << "[";
        for (int i = 0; i < 4; i++)
        {
            os << a.data[i];
            if (i != 3)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

    void bind(GLuint shader_program, std::string uniform_name = "model")
    {
        GLint matLoc = GLFuncs->glGetUniformLocation(shader_program, uniform_name.c_str());
        GLFuncs->glUniformMatrix4fv(matLoc, 1, GL_FALSE, (float*)&data);
    }
    
};

// vec4 * mat4
   __weak __host__ __device__ float32x4 operator*(float32x4 vec, mat4 mat)
    {
        float32x4 result;
        result[0] = vec[0] * mat.data[0][0] + vec[1] * mat.data[1][0] + vec[2] * mat.data[2][0] + vec[3] * mat.data[3][0];
        result[1] = vec[0] * mat.data[0][1] + vec[1] * mat.data[1][1] + vec[2] * mat.data[2][1] + vec[3] * mat.data[3][1];
        result[2] = vec[0] * mat.data[0][2] + vec[1] * mat.data[1][2] + vec[2] * mat.data[2][2] + vec[3] * mat.data[3][2];
        result[3] = vec[0] * mat.data[0][3] + vec[1] * mat.data[1][3] + vec[2] * mat.data[2][3] + vec[3] * mat.data[3][3];
        return result;
    };

    

#endif