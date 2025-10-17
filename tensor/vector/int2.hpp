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

struct uint84 
{
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t w;
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
    uint84(uint32_t v){
        this->x = (v >> 24) & 0xFF;
        this->y = (v >> 16) & 0xFF;
        this->z = (v >> 8) & 0xFF;
        this->w = v & 0xFF;
    };
};


// uint8_t& uint84::r = &uint84::x;