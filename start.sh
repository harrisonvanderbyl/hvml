# with and without simd
g++ ./qm.cpp -o a.out -I./tensor -lSDL2 -lGLEW -std=c++20 -ggdb
# hipcc ./qm.cpp -o hipa.out -I./tensor -lSDL2 -lGLEW -std=c++20 -ggdb
./a.out
# ./hipa.out
