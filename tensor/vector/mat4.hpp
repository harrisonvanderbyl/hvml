#include "float4.hpp"
#include "float3.hpp"
struct mat4
{
    float4 rows[4];

    mat4(float4 row0, float4 row1, float4 row2, float4 row3):
        rows{row0, row1, row2, row3}
    {
    };

    mat4():
        rows{float4(), float4(), float4(), float4()}
    {
    };

    mat4 transpose()
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.rows[j][i] = rows[i][j];
            }
        }
        return result;
    };

    void set_rotate(float angle, float3 axis)
    {
        float c = cos(angle);
        float s = sin(angle);
        float t = 1 - c;

        rows[0] = float4(
            t * axis.x * axis.x + c,
            t * axis.x * axis.y - s * axis.z,
            t * axis.x * axis.z + s * axis.y,
            0
        );

        rows[1] = float4(
            t * axis.x * axis.y + s * axis.z,
            t * axis.y * axis.y + c,
            t * axis.y * axis.z - s * axis.x,
            0
        );

        rows[2] = float4(
            t * axis.x * axis.z - s * axis.y,
            t * axis.y * axis.z + s * axis.x,
            t * axis.z * axis.z + c,
            0
        );

        rows[3] = float4(0, 0, 0, 1);
    };

    void rotate(float angle, float3 axis)
    {
        mat4 rotation;
        rotation.set_rotate(angle, axis);
        
        // Multiply the current matrix by the rotation matrix
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.rows[i][j] = rows[i].x * rotation.rows[0][j] +
                                    rows[i].y * rotation.rows[1][j] +
                                    rows[i].z * rotation.rows[2][j] +
                                    rows[i].w * rotation.rows[3][j];
            }
        }
        *this = result;
    };

    void point_towards_direction(float3 direction, float3 up = float3(0, 1, -1))
    {
        // Normalize the direction vector
        float3 dir = direction.normalize();
        
        // Calculate the right and up vectors
        // float3 up = float3(0, 1, -1); // Assuming Y is up
        float3 right = up.cross(dir).normalize();
        up = dir.cross(right); // Recalculate up to ensure orthogonality

        // Set the rotation matrix
        rows[0] = float4(right.x, up.x, -dir.x, 0);
        rows[1] = float4(right.y, up.y, -dir.y, 0);
        rows[2] = float4(right.z, up.z, -dir.z, 0);
        rows[3] = float4(0, 0, 0, 1);

        
        // Set the translation part to zero
        rows[3].x = -(right.x * 0 + up.x * 0
                        - dir.x * 0);   
        rows[3].y = -(right.y * 0 + up.y * 0
                        - dir.y * 0);
        rows[3].z = right.z * 0 + up.z * 0
                        - dir.z * 0;
        rows[3].w = 1;


    };

    mat4 pointed_towards_direction(float3 direction, float3 up = float3(0, 1, -1))
    {
        mat4 change = mat4::identity(); // Create an identity matrix for rotation
        
        mat4 myinverse = this->inverse(); // Get the inverse of the current matrix
        float3 relativedir =(myinverse * float4(direction.x, direction.y, direction.z, 0.0f)).xyz(); // Transform the direction to the local space

        change.point_towards_direction(relativedir, up); // Set the rotation

        mat4 result =  *this * change; // Multiply the current matrix by the rotation matrix
        return result;
    };

    void set_translate(float3 translation)
    {
        rows[0] = float4(1, 0, 0, translation.x);
        rows[1] = float4(0, 1, 0, translation.y);
        rows[2] = float4(0, 0, 1, translation.z);
        rows[3] = float4(0, 0, 0, 1);

    };

    void translate(float3 translation)
    {
        mat4 translation_matrix;
        translation_matrix.set_translate(translation);
        
        // Multiply the current matrix by the translation matrix
        mat4 result = *this * translation_matrix;
        *this = result;
    };

    void set_scale(float3 scale)
    {
        rows[0] = float4(scale.x, 0, 0, 0);
        rows[1] = float4(0, scale.y, 0, 0);
        rows[2] = float4(0, 0, scale.z, 0);
        rows[3] = float4(0, 0, 0, 1);
    };

    void scale(float3 scale)
    {
        mat4 scale_matrix;
        scale_matrix.set_scale(scale);
        
        // Multiply the current matrix by the scale matrix
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.rows[i][j] = rows[i].x * scale_matrix.rows[0][j] +
                                    rows[i].y * scale_matrix.rows[1][j] +
                                    rows[i].z * scale_matrix.rows[2][j] +
                                    rows[i].w * scale_matrix.rows[3][j];
            }
        }
        *this = result;
    };


    static mat4 identity()
    {
        return mat4(
            float4(1, 0, 0, 0),
            float4(0, 1, 0, 0),
            float4(0, 0, 1, 0),
            float4(0, 0, 0, 1)
        );
    };

    mat4 translated(float3 translation)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for translation
        change.set_translate(translation); // Set the translation

        mat4 result = change* *this; // Multiply the current matrix by the translation matrix
        return result;
    };

    mat4 scaled(float3 scale)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for scaling
        change.set_scale(scale); // Set the scale

        mat4 result = change * *this; // Multiply the current matrix by the scale matrix
        return result;
    };

    mat4 rotated(float angle, float3 axis)
    {
        mat4 change = mat4::identity(); // Create an identity matrix for rotation
        change.set_rotate(angle, axis); // Set the rotation

        mat4 result = change * *this; // Multiply the current matrix by the rotation matrix
        return result;
    };


    // matmul
    mat4 operator*(mat4 other)
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.rows[i][j] = rows[i].x * other.rows[0][j] +
                                    rows[i].y * other.rows[1][j] +
                                    rows[i].z * other.rows[2][j] +
                                    rows[i].w * other.rows[3][j];
            }
        }
        return result;
    };

    mat4 inverse()
    {
        // Using the adjugate method for 4x4 matrix inversion
        mat4 inv;
        float det = rows[0].x * (rows[1].y * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                                 rows[1].z * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) +
                                 rows[1].w * (rows[2].y * rows[3].z - rows[2].z * rows[3].y)) -
                    rows[0].y * (rows[1].x * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                                 rows[1].z * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                                 rows[1].w * (rows[2].x * rows[3].z - rows[2].z * rows[3].x)) +
                    rows[0].z * (rows[1].x * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) -
                                 rows[1].y * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                                 rows[1].w * (rows[2].x * rows[3].y - rows[2].y * rows[3].x)) -
                    rows[0].w * (rows[1].x * (rows[2].y * rows[3].z - rows[2].z * rows[3].y) -
                                 rows[1].y *(rows[2].x*rows[3].z-rows[2].z*rows[3].x)+
                                 rows[1].z*(rows[2].x*rows[3].y-rows[2].y*rows[3].x));
        
        if (det == 0)
        {
            throw "Matrix is singular and cannot be inverted";
        }

        // Calculate the inverse using the adjugate method
        // This is a simplified version and may not be numerically stable for all matrices
        inv = {
            float4(
                (rows[1].y * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                 rows[1].z * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) +
                 rows[1].w * (rows[2].y * rows[3].z - rows[2].z * rows[3].y)) / det,
                -(rows[0].y * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                  rows[0].z * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) +
                  rows[0].w * (rows[2].y * rows[3].z - rows[2].z * rows[3].y)) / det,
                (rows[0].y * (rows[1].z * rows[3].w - rows[1].w * rows[3].z) -
                 rows[0].z * (rows[1].y * rows[3].w - rows[1].w * rows[3].y) +
                 rows[0].w * (rows[1].y * rows[3].z - rows[1].z * rows[3].y)) / det,
                -(rows[0].y * (rows[1].z * rows[2].w - rows[1].w * rows[2].z) -
                  rows[0].z * (rows[1].y * rows[2].w - rows[1].w * rows[2].y) +
                  rows[0].w * (rows[1].y * rows[2].z - rows[1].z * rows[2].y)) / det
            ),
            float4(
                -(rows[1].x * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                  rows[1].z * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                  rows[1].w * (rows[2].x * rows[3].z - rows[2].z * rows[3].x)) / det,
                (rows[0].x * (rows[2].z * rows[3].w - rows[2].w * rows[3].z) -
                 rows[0].z * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                 rows[0].w * (rows[2].x * rows[3].z - rows[2].z * rows[3].x)) / det,
                -(rows[0].x * (rows[1].z * rows[3].w - rows[1].w * rows[3].z) -
                  rows[0].z * (rows[1].x * rows[3].w - rows[1].w * rows[3].x) +
                  rows[0].w * (rows[1].x * rows[3].z - rows[1].z * rows[3].x)) / det,
                (rows[0].x * (rows[1].z * rows[2].w - rows[1].w * rows[2].z) -
                 rows[0].z * (rows[1].x * rows[2].w - rows[1].w * rows[2].x) +
                 rows[0].w * (rows[1].x * rows[2].z - rows[1].z * rows[2].y)) / det
            ),
            float4(
                (rows[1].x * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) -
                 rows[1].y * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                 rows[1].w * (rows[2].x * rows[3].y - rows[2].y * rows[3].x)) / det,
                -(rows[0].x * (rows[2].y * rows[3].w - rows[2].w * rows[3].y) -
                  rows[0].y * (rows[2].x * rows[3].w - rows[2].w * rows[3].x) +
                  rows[0].w * (rows[2].x * rows[3].y - rows[2].y * rows[3].x)) / det,
                (rows[0].x * (rows[1].y * rows[3].w - rows[1].w * rows[3].y) -
                 rows[0].y * (rows[1].x * rows[3].w - rows[1].w * rows[3].x) +
                 rows[0].w * (rows[1].x * rows[3].y - rows[1].y * rows[3].x)) / det,
                -(rows[0].x * (rows[1].y * rows[2].w - rows[1].w * rows[2].y) -
                  rows[0].y * (rows[1].x * rows[2].w - rows[1].w * rows[2].x) +
                  rows[0].w * (rows[1].x * rows[2].y - rows[1].y * rows[2].x)) / det
            ),
            float4(
                -(rows[1].x * (rows[2].y * rows[3].z - rows[2].z * rows[3].y) -
                  rows[1].y * (rows[2].x * rows[3].z - rows[2].z * rows[3].x) +
                  rows[1].z * (rows[2].x * rows[3].y - rows[2].y * rows[3].x)) / det,
                (rows[0].x * (rows[2].y * rows[3].z - rows[2].z * rows[3].y) -
                 rows[0].y * (rows[2].x * rows[3].z - rows[2].z * rows[3].x) +
                 rows[0].z * (rows[2].x * rows[3].y - rows[2].y * rows[3].x)) / det,
                -(rows[0].x * (rows[1].y * rows[3].z - rows[1].z * rows[3].y) - 
                  rows[0].y * (rows[1].x * rows[3].z - rows[1].z * rows[3].x) +
                  rows[0].z * (rows[1].x * rows[3].y - rows[1].y * rows[3].x)) / det,
                (rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y) -
                 rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x) +
                 rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x)) / det
            )
        };

        return inv;
    };

    float4 operator * (float4 vec)
    {
        float4 result;
        result.x = rows[0].x * vec.x + rows[0].y * vec.y + rows[0].z * vec.z + rows[0].w * vec.w;
        result.y = rows[1].x * vec.x + rows[1].y * vec.y + rows[1].z * vec.z + rows[1].w * vec.w;
        result.z = rows[2].x * vec.x + rows[2].y * vec.y + rows[2].z * vec.z + rows[2].w * vec.w;
        result.w = rows[3].x * vec.x + rows[3].y * vec.y + rows[3].z * vec.z + rows[3].w * vec.w;
        return result;
    };

    float3 operator * (float3 vec)
    {
        float4 result = *this * float4(vec.x, vec.y, vec.z, 1.0f);
        return float3(result.x, result.y, result.z);
    };

    // print
    friend std::ostream &operator<<(std::ostream &os, mat4 a)
    {
        os << "[";
        for (int i = 0; i < 4; i++)
        {
            os << a.rows[i];
            if (i != 3)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
    
};



