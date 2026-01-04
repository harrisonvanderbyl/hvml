# nvcc ./CreateWindowWithCudaWrite.cu -I./tensor  -lSDL3 -lGLEW -lGL -std=c++20
# create shared library
nvcc \
  -Xcompiler -fPIC \
  -shared \
  CreateWindowWithCudaWrite.cu \
  -o CreateWindowWithCudaWrite.so \
  -lGL -lGLEW -lSDL3 -std=c++20 \
  -I./tensor