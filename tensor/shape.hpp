
#include <iostream>
#include <initializer_list>
#include <vector>

#ifndef __host__
#define __host__
#define __device__
#endif

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
    __host__ __device__ Shape(T A = INT64MAX, TT B = INT64MAX, TTT C = INT64MAX, TTTT D = INT64MAX, TTTTT E = INT64MAX, TTTTTT F = INT64MAX, TTTTTTT G = INT64MAX)
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

    template <int odim>
    Shape(const Shape<odim>& other)
    {
        if(odim != length && length != -1 && odim != -1){
            std::cerr << "Cannot assign shapes of different ranks, mrank: " << length << ",orank: " << odim << std::endl;
            throw std::runtime_error("Cannot assign shapes of different ranks");
        }

        if(length != -1){
            if (int(other.ndim())!=length){

                std::cerr << "Cannot assign shape of a different rank to a static typed rank shape" << std::endl;
                throw std::runtime_error("Cannot assign shapes of different ranks");
            }
        }

        for (int i = 0; i < other.ndim(); i++)
        {
            ((long *)this)[i] = ((long *)&other)[i];
        }
    }
   

     __host__ __device__ Shape()
    {
        
    }

    
    // Shape<std::max(length - 1, -1)> slice(){
    //     return *(Shape<std::max(length - 1, -1)>*)(((long*)this)+1);
    // }

    // template <int numslice = 1> 
    // Shape<std::max(length - numslice, -1)> slice(){
    //     return *(Shape<std::max(length - numslice, -1)>*)(((long*)this)+numslice);
    // }

    inline Shape clone()
    {
        return *this;
    }

    Shape<length> calc_strides() const
    {
        Shape<length> s;
        size_t ndim = this->ndim();
        long *a = (long *)this;
        s[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            s[i] = s[i + 1] * a[i + 1];
        }
        return s;
    }
    

    __host__ __device__ long& operator[](const int& i) const
    {      
        int d = i;
        if (d < 0){
            auto ii = ndim();
            d = (d + ii)%ii;
        }
        return ((long *)this)[d];
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

    __host__ __device__ size_t total_size() const
    {
        size_t total = 1;
        long *a = (long *)this;
        for (int i = 0; i < ndim(); i++)
        {
            total *= a[i];
        }
        return total;
    }

    __host__ __device__ size_t ndim() const
    {
        if (length != -1)
        {
            return (size_t)length;
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

    template <int newrank>
    bool operator==(const Shape<newrank> &other) const
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

    template <int newrank>
    bool operator!=(const Shape<newrank> &other) const
    {
        return !(*this == other);
    }

    template <int newrank>
    Shape operator=(const Shape<newrank> &other)
    {
        if(newrank != length && length != -1 && newrank != -1){
            std::cerr << "Cannot assign shapes of different ranks" << std::endl;
            throw std::runtime_error("Cannot assign shapes of different ranks");
        }

        for (int i = 0; i < other.ndim(); i++)
        {
            ((long *)this)[i] = ((long *)&other)[i];
        }
        return *this;
    }
    
    
};

template <int rank>
std::ostream &operator<<(std::ostream &os, Shape<rank> a)
{
    os << a.operator std::string();
    return os;
}


class DefaultSlice
{
    int Value;
    public:
        bool is_default = true;
        DefaultSlice(){};

        template <typename inttype = int, typename = std::enable_if_t<std::is_integral_v<inttype>>>
        DefaultSlice(inttype v) : Value(v), is_default(false) {};
        operator int() const {
            return Value;
        };
};

class Slice
{
    public:
    DefaultSlice start;
    DefaultSlice end;
    DefaultSlice step;
    bool is_slice;
    bool is_empty;

    
    // Constructor: full parameters

    Slice(DefaultSlice starti, DefaultSlice endi, DefaultSlice stepi = 1)
        : start(starti), end(endi), step(stepi), is_slice(true), is_empty(false) {};

    // Constructor: reduced slice
    Slice(DefaultSlice starti)
        : start(starti), end(-1), step(1), is_slice(false), is_empty(false) {};

    // Constructor: empty slice
    Slice()
        : start(0), end(-1), step(1) , is_slice(true), is_empty(true) {};

    // Slice(const Slice &other)
    //     : start(other.start), end(other.end), step(other.step), is_slice(other.is_slice), is_empty(other.is_empty) {};
};


template <int myreducedims = -1>
class SliceList {
public:
    static constexpr int reducedims = myreducedims;

    Slice A;
    Slice B;
    Slice C;
    Slice D;
    Slice E;
    Slice F;
    Slice G;
    // make sure amount of ints is equal to reducedims
    template <typename a = Slice, typename b = Slice, typename c = Slice, typename d = Slice, typename e = Slice, typename f = Slice, typename g = Slice, typename = std::enable_if_t<std::is_same_v<a,int> + std::is_same_v<b,int> + std::is_same_v<c,int> + std::is_same_v<d,int> + std::is_same_v<e,int> + std::is_same_v<f,int> + std::is_same_v<g,int> == reducedims>>
    SliceList(a aa=Slice(), b ba=Slice(), c ca=Slice(), d da=Slice(), e ea=Slice(), f fa=Slice(), g ga=Slice())
        : A(aa), B(ba), C(ca), D(da), E(ea), F(fa), G(ga) {};



    
    const Slice& operator[](int i) const
    {
        if (i == 0)
        {
            return A;
        }
        else if (i == 1)
        {
            return B;
        }
        else if (i == 2)
        {
            return C;
        }
        else if (i == 3)
        {
            return D;
        }
        else if (i == 4)
        {
            return E;
        }
        else if (i == 5)
        {
            return F;
        }
        else
        {
            return G;
        }
    }

    operator std::string() const
    {
        std::string result = "[";
        for (int i = 0; i < 7; i++)
        {
            if (i > 0)
            {
                result += ", ";
            }
            result += std::to_string((*this)[i].start) + ":" + std::to_string((*this)[i].end) + ":" + std::to_string((*this)[i].step);
        }
        result += "]";
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const SliceList& sl)
    {
        os << static_cast<std::string>(sl);
        return os;
    }

    SliceList operator,(const Slice& other) const
    {
        auto copy = *this;
        for (int i = 0; i < 7; i++)
        {
            if (copy[i].is_empty)
            {
                copy[i] = other;
                break;
            }
        }
        return copy;
    }

    SliceList<reducedims+1> operator,(const int& other) const
    {
        auto copy = *this;
        for (int i = 0; i < 7; i++)
        {
            if (copy[i].is_empty)
            {
                copy[i] = Slice(other);
                break;
            }
        }
        return SliceList<reducedims+1>(copy.A, copy.B, copy.C, copy.D, copy.E, copy.F, copy.G);
    }

};

__weak SliceList<0> operator,(const Slice& a, const Slice& b){
    return SliceList<0>{a,b};
};

__weak SliceList<1> operator,(const Slice& a, const int& b){
    return SliceList<1>{a,b};
};

__weak SliceList<1> operator,(const int& a, const Slice& b){
    return SliceList<1>{a,b};
};



#endif

// Shape a(1,2,3);