
#include <iostream>
#include <initializer_list>
#include <vector>
#ifndef SHAPE
#define SHAPE
#define INT64MAX 9223372036854775807
template <int length = -1>
struct Shape
{
    long A = INT64MAX;
    long B = INT64MAX;
    long C = INT64MAX;
    long D = INT64MAX;
    long E = INT64MAX;
    long F = INT64MAX;
    long G = INT64MAX;

    template <typename T = long, typename TT = long, typename TTT = long, typename TTTT = long, typename TTTTT = long, typename TTTTTT = long, typename TTTTTTT = long>
    Shape(T A = INT64MAX, TT B = INT64MAX, TTT C = INT64MAX, TTTT D = INT64MAX, TTTTT E = INT64MAX, TTTTTT F = INT64MAX, TTTTTTT G = INT64MAX)
        : A(A), B(B), C(C), D(D), E(E), F(F), G(G)
    {
        
    }
   

    Shape(std::vector<size_t> a)
    {
        for (int i = 0; i < a.size(); i++)
        {
            if (i == 0)
            {
                A = a[i];
            }
            if (i == 1)
            {
                B = a[i];
            }
            if (i == 2)
            {
                C = a[i];
            }
            if (i == 3)
            {
                D = a[i];
            }
            if (i == 4)
            {
                E = a[i];
            }
            if (i == 5)
            {
                F = a[i];
            }
            if (i == 6)
            {
                G = a[i];
            }
        }
    }
   

    Shape()
    {
        
    }

    
    // Shape<std::max(length - 1, -1)> slice(){
    //     return *(Shape<std::max(length - 1, -1)>*)(((long*)this)+1);
    // }

    template <int numslice = 1> 
    Shape<std::max(length - numslice, -1)> slice(){
        return *(Shape<std::max(length - numslice, -1)>*)(((long*)this)+numslice);
    }

    inline Shape clone()
    {
        return *this;
    }

    

    long& operator[](int i)
    {      
        if (i < 0){
            auto ii = ndim();
            i = (i + ii)%ii;
        }
        return ((long *)this)[i];
    }

    operator std::string()
    {
        std::string s = "[";
        long *a = (long *)this;
        for (int i = 0; i<ndim(); i++)
        {
            s += std::to_string(a[i]);
            if (a[i + 1] != INT64MAX)
            {
                s += ", ";
            }
        }
        s += "]";
        return s;
    }

    size_t total_size() const
    {
        size_t total = 1;
        long *a = (long *)this;
        for (int i = 0; i < ndim(); i++)
        {
            total *= a[i];
        }
        return total;
    }

    size_t ndim() const
    {
        if (length != -1)
        {
            return length;
        }
        size_t total = 0;
        for (int i = 0; i < 10; i++)
        {
            if (((long *)this)[i] == INT64MAX)
            {
                break;
            }
            total++;
        }
        return total;
    }

    bool operator==(const Shape &other) const
    {
        for (int i = 0; i < ndim(); i++)
        {
            if (((long *)this)[i] != ((long *)&other)[i])
            {
                return false;
            }
        }
        return true;
    }

    
};

template <int rank>
std::ostream &operator<<(std::ostream &os, Shape<rank> a)
{
    os << a.operator std::string();
    return os;
}
#endif

// Shape a(1,2,3);