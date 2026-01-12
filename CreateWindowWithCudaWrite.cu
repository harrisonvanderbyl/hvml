#include "tensor.hpp"
#include "ops/ops.h"
#include "display/display.hpp"
#include <assert.h>
#include <map>
#include <set>
#include <string>

void setEnvsForNvidia(){
    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);
    setenv("SDL_DEBUG", "1", 1);
}

VectorDisplay<kCUDA>* DisplayPtr = nullptr;

void createDisplayPtr(int width, int height, int flags = 0){
    DisplayPtr = new VectorDisplay<kCUDA>({height, width}, flags);
}

void* getDisplayPtrData(){
    return DisplayPtr->data;
}

void startDisplayLoop(){
    DisplayPtr->displayLoop();
}

// ============================================================================
// PYGAME INITIALIZATION & DISPLAY
// ============================================================================

extern "C" void pygame_init(){
    // SDL is initialized when display is created
}

extern "C" void pygame_quit(){
    if (DisplayPtr) {
        delete DisplayPtr;
        DisplayPtr = nullptr;
    }
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

// Display mode setting (similar to pygame.display.set_mode)
extern "C" void* display_set_mode(int width, int height, int flags){
    if (DisplayPtr) {
        delete DisplayPtr;
    }
    createDisplayPtr(width, height, flags);
    return DisplayPtr->data;
}

extern "C" void display_set_caption(const char* title){
    if (DisplayPtr) {
        DisplayPtr->setWindowCaption(title);
    }
}

extern "C" void display_flip(){
    if (DisplayPtr) {
        DisplayPtr->updateDisplay();
    }
}

extern "C" void display_update(){
    if (DisplayPtr) {
        DisplayPtr->updateDisplay();
    }
}

extern "C" void get_surface_size(int* width, int* height){
    if (DisplayPtr) {
        auto size = DisplayPtr->getWindowSize();
        *width = size.first;
        *height = size.second;
    }
}

// ============================================================================
// PYGAME EVENT HANDLING
// ============================================================================

struct EventInfo {
    int type;
    int key;
    int button;
    int x, y;
    int xrel, yrel;
};

static std::vector<EventInfo> event_queue;
static bool quit_requested = false;

extern "C" int event_poll(EventInfo* event){
    if (!DisplayPtr) return 0;
    // Process SDL events and populate our queue
    if (event_queue.empty()) {

        event_queue.clear();
        quit_requested = false;
        
        SDL_Event sdl_event;
        while (SDL_PollEvent(&sdl_event)) {
            EventInfo info = {0};
            
            switch (sdl_event.type) {
                case SDL_EVENT_QUIT:
                    info.type = SDL_EVENT_QUIT;
                    quit_requested = true;
                    event_queue.push_back(info);
                    break;
                    
                case SDL_EVENT_KEY_DOWN:
                    info.type = SDL_EVENT_KEY_DOWN;
                    info.key = sdl_event.key.key;
                    event_queue.push_back(info);
                    break;
                    
                case SDL_EVENT_KEY_UP:
                    info.type = SDL_EVENT_KEY_UP;
                    info.key = sdl_event.key.key;
                    event_queue.push_back(info);
                    break;
                    
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    info.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
                    info.button = sdl_event.button.button;
                    info.x = sdl_event.button.x;
                    info.y = sdl_event.button.y;
                    event_queue.push_back(info);
                    break;
                    
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    info.type = SDL_EVENT_MOUSE_BUTTON_UP;
                    info.button = sdl_event.button.button;
                    info.x = sdl_event.button.x;
                    info.y = sdl_event.button.y;
                    event_queue.push_back(info);
                    break;
                    
                case SDL_EVENT_MOUSE_MOTION:
                    info.type = SDL_EVENT_MOUSE_MOTION;
                    info.x = sdl_event.motion.x;
                    info.y = sdl_event.motion.y;
                    info.xrel = sdl_event.motion.xrel;
                    info.yrel = sdl_event.motion.yrel;
                    DisplayPtr->current_screen_input_info.updateMouseMotion(info.xrel, info.yrel);
                    event_queue.push_back(info);
                    break;
            }
        }
    }
    
    if (!event_queue.empty()) {
        *event = event_queue.front();
        event_queue.erase(event_queue.begin());
        return 1;
    }
    
    return 0;
}

extern "C" void event_get(EventInfo* events, int* count, int max_events){
    *count = 0;
    while (*count < max_events) {
        if (!event_poll(&events[*count])) {
            break;
        }
        (*count)++;
    }
}

extern "C" void event_clear(){
    event_queue.clear();
    SDL_Event e;
    while (SDL_PollEvent(&e)) {}
}

// ============================================================================
// PYGAME MOUSE
// ============================================================================

extern "C" void mouse_get_pos(int* x, int* y){
    if (DisplayPtr) {
        auto pos = DisplayPtr->current_screen_input_info.getMousePosition();
        *x = (int)pos.x;
        *y = (int)pos.y;
    }
}

extern "C" void mouse_get_rel(int* dx, int* dy){
    if (DisplayPtr) {
        auto rel = DisplayPtr->current_screen_input_info.getMouseRel();
        *dx = rel.first;
        *dy = rel.second;
    }
    DisplayPtr->current_screen_input_info.updateMouseMotion(0, 0);
}

extern "C" int mouse_get_pressed(int button){
    if (!DisplayPtr) return 0;
    
    auto buttons = DisplayPtr->current_screen_input_info.getMouseButtonsPressed();
    return buttons.find(button) != buttons.end() ? 1 : 0;
}

extern "C" void mouse_get_pressed_array(int* buttons){
    if (!DisplayPtr) {
        buttons[0] = buttons[1] = buttons[2] = 0;
        return;
    }
    
    auto pressed = DisplayPtr->current_screen_input_info.getMouseButtonsPressed();
    buttons[0] = pressed.find(SDL_BUTTON_LEFT) != pressed.end() ? 1 : 0;
    buttons[1] = pressed.find(SDL_BUTTON_MIDDLE) != pressed.end() ? 1 : 0;
    buttons[2] = pressed.find(SDL_BUTTON_RIGHT) != pressed.end() ? 1 : 0;
}

extern "C" void mouse_set_visible(int visible){
    if (DisplayPtr) {
        DisplayPtr->setMouseVisible(visible != 0);
    }
}

extern "C" void event_set_grab(int grab){
    if (DisplayPtr) {
        DisplayPtr->setMouseGrab(grab != 0);
    }
}

// ============================================================================
// PYGAME KEY
// ============================================================================

// Key code mapping similar to pygame
extern "C" int key_code(const char* key_name){
    return SDL_GetKeyFromName(key_name);
}

extern "C" int key_get_pressed(int keycode){
    if (!DisplayPtr) return 0;
    return DisplayPtr->current_screen_input_info.isKeyPressed(keycode) ? 1 : 0;
}

// Key constants (matching pygame's naming)
extern "C" int get_K_ESCAPE() { return SDLK_ESCAPE; }
extern "C" int get_K_SPACE() { return SDLK_SPACE; }
extern "C" int get_K_RETURN() { return SDLK_RETURN; }
extern "C" int get_K_LSHIFT() { return SDLK_LSHIFT; }
extern "C" int get_K_RSHIFT() { return SDLK_RSHIFT; }
extern "C" int get_K_w() { return SDLK_W; }
extern "C" int get_K_a() { return SDLK_A; }
extern "C" int get_K_s() { return SDLK_S; }
extern "C" int get_K_d() { return SDLK_D; }
extern "C" int get_K_q() { return SDLK_Q; }
extern "C" int get_K_e() { return SDLK_E; }
extern "C" int get_K_u() { return SDLK_U; }
extern "C" int get_K_INSERT() { return SDLK_INSERT; }
extern "C" int get_K_DELETE() { return SDLK_DELETE; }
extern "C" int get_K_MINUS() { return SDLK_MINUS; }
extern "C" int get_K_EQUALS() { return SDLK_EQUALS; }
extern "C" int get_K_LEFTBRACKET() { return SDLK_LEFTBRACKET; }
extern "C" int get_K_RIGHTBRACKET() { return SDLK_RIGHTBRACKET; }
extern "C" int get_K_BACKSLASH() { return SDLK_BACKSLASH; }
extern "C" int get_K_SEMICOLON() { return SDLK_SEMICOLON; }
extern "C" int get_K_QUOTE() { return SDLK_APOSTROPHE; }
extern "C" int get_K_COMMA() { return SDLK_COMMA; }
extern "C" int get_K_PERIOD() { return SDLK_PERIOD; }
extern "C" int get_K_SLASH() { return SDLK_SLASH; }

// Number keys
extern "C" int get_K_0() { return SDLK_0; }
extern "C" int get_K_1() { return SDLK_1; }
extern "C" int get_K_2() { return SDLK_2; }
extern "C" int get_K_3() { return SDLK_3; }
extern "C" int get_K_4() { return SDLK_4; }
extern "C" int get_K_5() { return SDLK_5; }
extern "C" int get_K_6() { return SDLK_6; }
extern "C" int get_K_7() { return SDLK_7; }
extern "C" int get_K_8() { return SDLK_8; }
extern "C" int get_K_9() { return SDLK_9; }

// ============================================================================
// PYGAME SURFACE & TRANSFORM
// ============================================================================

struct SurfaceHandle {
    int width;
    int height;
    std::vector<uint32_t> data;
};

static std::map<int, SurfaceHandle> surfaces;
static int next_surface_id = 1;

extern "C" int surface_create(int width, int height){
    int id = next_surface_id++;
    surfaces[id] = SurfaceHandle{width, height, std::vector<uint32_t>(width * height, 0)};
    return id;
}

extern "C" void surface_destroy(int surface_id){
    surfaces.erase(surface_id);
}

extern "C" void* surface_get_data(int surface_id){
    if (surfaces.find(surface_id) != surfaces.end()) {
        return surfaces[surface_id].data.data();
    }
    return nullptr;
}

extern "C" void surface_get_size(int surface_id, int* width, int* height){
    if (surfaces.find(surface_id) != surfaces.end()) {
        *width = surfaces[surface_id].width;
        *height = surfaces[surface_id].height;
    }
}

extern "C" int surfarray_make_surface(const uint8_t* array, int width, int height){
    int id = surface_create(width, height);
    if (surfaces.find(id) != surfaces.end()) {
        // Copy RGB data (array is HxWx3)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                uint8_t r = array[idx + 0];
                uint8_t g = array[idx + 1];
                uint8_t b = array[idx + 2];
                surfaces[id].data[y * width + x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            }
        }
    }
    return id;
}

extern "C" int transform_scale(int surface_id, int new_width, int new_height){
    if (surfaces.find(surface_id) == surfaces.end()) return -1;
    
    auto& src = surfaces[surface_id];
    int new_id = surface_create(new_width, new_height);
    auto& dst = surfaces[new_id];
    
    // Simple nearest neighbor scaling
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = x * src.width / new_width;
            int src_y = y * src.height / new_height;
            dst.data[y * new_width + x] = src.data[src_y * src.width + src_x];
        }
    }
    
    return new_id;
}

extern "C" void blit_surface(int src_id, int x, int y){
    if (!DisplayPtr || surfaces.find(src_id) == surfaces.end()) return;
    
    auto& src = surfaces[src_id];
    DisplayPtr->blitSurface(
        VectorDisplay<kCUDA>::Surface{src.width, src.height, src.data.data()},
        x, y
    );
}

// ============================================================================
// PYGAME FONT
// ============================================================================

struct FontHandle {
    int size;
    std::string name;
};

static std::map<int, FontHandle> fonts;
static int next_font_id = 1;

extern "C" int font_SysFont(const char* name, int size){
    int id = next_font_id++;
    fonts[id] = FontHandle{size, name ? name : "default"};
    return id;
}

extern "C" void font_destroy(int font_id){
    fonts.erase(font_id);
}

extern "C" int font_render(int font_id, const char* text, int antialias, uint32_t color){
    // Placeholder: return a surface with text rendered
    // In real implementation, would use SDL_ttf
    if (fonts.find(font_id) == fonts.end()) return -1;
    
    int width = strlen(text) * fonts[font_id].size / 2;
    int height = fonts[font_id].size;
    
    int surf_id = surface_create(width, height);
    // Fill with text color (simplified)
    if (surfaces.find(surf_id) != surfaces.end()) {
        for (auto& pixel : surfaces[surf_id].data) {
            pixel = color;
        }
    }
    
    return surf_id;
}

// ============================================================================
// PYGAME TIME/CLOCK
// ============================================================================

struct ClockHandle {
    std::chrono::steady_clock::time_point last_tick;
    ClockHandle() : last_tick(std::chrono::steady_clock::now()) {}
};

static std::map<int, ClockHandle> clocks;
static int next_clock_id = 1;

extern "C" int time_Clock(){
    int id = next_clock_id++;
    clocks[id] = ClockHandle();
    return id;
}

extern "C" void clock_destroy(int clock_id){
    clocks.erase(clock_id);
}

extern "C" void clock_tick(int clock_id, int fps){
    if (clocks.find(clock_id) == clocks.end()) return;
    
    auto& clock = clocks[clock_id];
    auto target_duration = std::chrono::microseconds(1000000 / fps);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - clock.last_tick);
    
    if (elapsed < target_duration) {
        auto sleep_time = target_duration - elapsed;
        // std::this_thread::sleep_for(sleep_time);
    }
    
    clock.last_tick = std::chrono::steady_clock::now();
}

extern "C" int clock_get_fps(int clock_id){
    if (clocks.find(clock_id) == clocks.end()) return 0;
    
    auto& clock = clocks[clock_id];
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - clock.last_tick);
    if (elapsed.count() == 0) return 0;
    return 1000000 / elapsed.count();
}

extern "C" void time_delay(int milliseconds){
    // std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// ============================================================================
// WINDOW FLAGS (matching pygame constants)
// ============================================================================

extern "C" int get_RESIZABLE() { return WP_RESIZABLE; }
extern "C" int get_FULLSCREEN() { return WP_FULLSCREEN; }
extern "C" int get_BORDERLESS() { return WP_BORDERLESS; }

// ============================================================================
// BUTTON CONSTANTS
// ============================================================================

extern "C" int get_BUTTON_LEFT() { return SDL_BUTTON_LEFT; }
extern "C" int get_BUTTON_MIDDLE() { return SDL_BUTTON_MIDDLE; }
extern "C" int get_BUTTON_RIGHT() { return SDL_BUTTON_RIGHT; }

// ============================================================================
// HELPER FUNCTIONS FOR ASYNC/COROUTINE USAGE
// ============================================================================

extern "C" int should_quit(){
    return quit_requested ? 1 : 0;
}

extern "C" void process_events(){
    if (!DisplayPtr) return;
    DisplayPtr->processEvents();
}

// ============================================================================
// TESTING/DEBUG
// ============================================================================

// int main(int argc, char** argv){
//     init(800, 600);
//     startDisplayLoop();
//     return 0;
// }