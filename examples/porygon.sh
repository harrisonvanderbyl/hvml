# with and without simd
g++ ./porygonpet.cpp -o a.out -I../tensor -lXrender -lXfixes -lXrandr -pthread -lXext -lSDL3 -lGLEW -lGL -lX11 -lXrender -lXcomposite -std=c++20 -fpermissive -ggdb
# hipcc ./qm.cpp -o hipa.out -I./tensor -lSDL2 -lGLEW -lX11 -std=c++20 -ggdb
sudo ./a.out
# ./hipa.out
