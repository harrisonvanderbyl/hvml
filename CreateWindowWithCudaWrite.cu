
#include "tensor.hpp"
#include "ops/ops.h"
#include "display/display.hpp"
#include <assert.h>



void setEnvsForNvidia(){
    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);
    setenv("SDL_DEBUG", "1", 1);
}


VectorDisplay<kCUDA>* DisplayPtr = nullptr;

void createDisplayPtr(int width, int height){
    DisplayPtr = new VectorDisplay<kCUDA>({height, width}, 0);
}

void* getDisplayPtrData(){
    return DisplayPtr->data;
}

void startDisplayLoop(){
    DisplayPtr->displayLoop();
}



extern "C" void init(int width, int height){
    setEnvsForNvidia();
    createDisplayPtr(width, height);
}

extern "C" void* getDataPtr(){
    return getDisplayPtrData();
}

extern "C" void updateDisplay(){
    DisplayPtr->updateDisplay();
}

extern "C" void copyDataToDisplay(void* src, size_t bytes){
    cudaMemcpy(DisplayPtr->data, src, bytes, cudaMemcpyDeviceToDevice);
}

// int main(int argc, char** argv){
//     init(800, 600);
//     runDisplayLoop();
//     return 0;
// }