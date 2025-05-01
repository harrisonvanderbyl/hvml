#include <SDL2/SDL.h>
#include <iostream>
#include <cstring>
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include <thread>
#include <functional>

class VectorDisplay : public Tensor<uint84, 2> {
public:
    SDL_Texture* texture = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Window* window = nullptr;

    // functions to call during the display loop
    std::function<void()> on_update = nullptr;

    VectorDisplay(Shape<2> shape) : Tensor<uint84, 2>(shape) {
        // Initialize the display with a black background

        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }

        for (int y = 0; y < shape[0]; y++) {
            for (int x = 0; x < shape[1]; x++) {
                (*this)[y][x] = {0, 0, 0, 255}; // rgba
            }
        }

        // Create SDL window
        window = SDL_CreateWindow("512x512 Window with Writable Texture",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            shape[1], shape[0],
            SDL_WINDOW_SHOWN);

        if (window == nullptr) {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }

        // Create SDL renderer
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

        if (renderer == nullptr) {
            std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            return;
        }

        // Create a texture from this.data
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
            SDL_TEXTUREACCESS_STREAMING, shape[1], shape[0]);

        if (texture == nullptr) {
            std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            return;
        }

        
    }

    void updateTexture() {
        // Update the texture with the pixel data
        void* texturePixels = nullptr;
        int pitch = 0;
        SDL_LockTexture(texture, nullptr, &texturePixels, &pitch);
        memcpy(texturePixels, this->data, shape[1] * shape[0] * sizeof(uint32_t));
        SDL_UnlockTexture(texture);
    }

    void render() {
        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);

        // Present the renderer
        SDL_RenderPresent(renderer);
    }

    void displayLoop() {
        bool quit = false;
        SDL_Event e;

        while (!quit) {
            // Call the on_update functions if set
            if (on_update) {
                on_update();
            }
            // Event handling loop
            while (SDL_PollEvent(&e) != 0) {
                if (e.type == SDL_QUIT) {
                    quit = true;
                }
            }

            // Update the texture and render it
            updateTexture();
            render();
        }
    }

    ~VectorDisplay() {
        if (texture) {
            SDL_DestroyTexture(texture);
        }
        if (renderer) {
            SDL_DestroyRenderer(renderer);
        }
        if (window) {
            SDL_DestroyWindow(window);
        }
        SDL_Quit();
    }

    void add_on_update(std::function<void()> func) {
        // change the on_update function to first run the old one, then the new one
        auto old_on_update = on_update;
        on_update = [old_on_update, func]() {
            if (old_on_update) {
                old_on_update();
            }
            func();
        };
    }
};
