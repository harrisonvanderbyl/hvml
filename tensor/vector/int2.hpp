struct int2 
{
    int x;
    int y;
    int2(int x, int y){
        this->x = x;
        this->y = y;
    };
    int2(){
        this->x = 0;
        this->y = 0;
    };
};


struct int4 
{
    int x;
    int y;
    int z;
    int w;
    int4(int x, int y, int z, int w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    int4(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };
};

class uint84 
{
    public:
    uint8_t w;
    uint8_t z;
    uint8_t y;
    uint8_t x;
    
    
    uint84(uint8_t x, uint8_t y, uint8_t z, uint8_t w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    uint84(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };
    uint84 operator=(uint84 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };
    uint84 operator=(uint8_t other)
    {
        x = other;
        y = other;
        z = other;
        w = other;
        return *this;
    };
    uint84 operator=(uint84 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };
    
};



// uint8_t& uint84::r = &uint84::x;