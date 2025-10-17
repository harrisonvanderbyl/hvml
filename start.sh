# with and without simd
g++ ./qm.cpp -o a.out -I./tensor -lXrender -lXfixes -lXrandr -pthread -lXext -lSDL2 -lGLEW -lGL -lX11 -lXrender -lXcomposite -std=c++20 -ggdb
# hipcc ./qm.cpp -o hipa.out -I./tensor -lSDL2 -lGLEW -lX11 -std=c++20 -ggdb
./a.out
# ./hipa.out
