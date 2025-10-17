
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/extensions/Xrender.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/shape.h>
#include <X11/extensions/Xrandr.h>
#include <linux/input.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <poll.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <functional>
#include <atomic>
#include <map>
#include <string>
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "display/input.hpp"

#ifndef DISPLAY_MANAGER_HPP
#define DISPLAY_MANAGER_HPP
class DisplayInfo {
public:
    int x, y;
    int width, height;
    int index;
    std::string name;
    bool is_primary;
    
    DisplayInfo(int x = 0, int y = 0, int w = 0, int h = 0, int idx = 0, 
                const std::string& n = "", bool primary = false)
        : x(x), y(y), width(w), height(h), index(idx), name(n), is_primary(primary) {}
    
    bool containsPoint(int px, int py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
    
    // Convert global coordinates to display-relative coordinates
    std::pair<int, int> globalToLocal(int global_x, int global_y) const {
        return {global_x - x, global_y - y};
    }
    
    // Convert display-relative coordinates to global coordinates
    std::pair<int, int> localToGlobal(int local_x, int local_y) const {
        return {local_x + x, local_y + y};
    }
};

class MultiDisplayManager {
private:
    Display* display;
    std::vector<DisplayInfo> displays;
    int current_display_index = 0;
    
public:
    MultiDisplayManager(Display* dpy) : display(dpy) {
        if(dpy){
            detectDisplays();
        }
    }
    
    void detectDisplays() {
        displays.clear();
        
        // Check if Xrandr extension is available
        int xrandr_event_base, xrandr_error_base;
        if (!XRRQueryExtension(display, &xrandr_event_base, &xrandr_error_base)) {
            std::cerr << "Xrandr extension not available, falling back to single display" << std::endl;
            // Fallback to single display
            int screen = DefaultScreen(display);
            int width = DisplayWidth(display, screen);
            int height = DisplayHeight(display, screen);
            displays.emplace_back(0, 0, width, height, 0, "Display0", true);
            return;
        }
        
        int screen = DefaultScreen(display);
        Window root = RootWindow(display, screen);
        
        XRRScreenResources* screen_resources = XRRGetScreenResources(display, root);
        if (!screen_resources) {
            std::cerr << "Failed to get screen resources" << std::endl;
            return;
        }
        
        // Get primary output
        RROutput primary = XRRGetOutputPrimary(display, root);
        
        int display_index = 0;
        for (int i = 0; i < screen_resources->noutput; i++) {
            XRROutputInfo* output_info = XRRGetOutputInfo(display, screen_resources, 
                                                          screen_resources->outputs[i]);
            
            if (output_info->connection == RR_Connected && output_info->crtc) {
                XRRCrtcInfo* crtc_info = XRRGetCrtcInfo(display, screen_resources, 
                                                        output_info->crtc);
                
                if (crtc_info && crtc_info->width > 0 && crtc_info->height > 0) {
                    bool is_primary = (screen_resources->outputs[i] == primary);
                    std::string name = std::string(output_info->name);
                    
                    displays.emplace_back(crtc_info->x, crtc_info->y, 
                                        crtc_info->width, crtc_info->height,
                                        display_index, name, is_primary);
                    
                    std::cout << "Found display " << display_index << ": " << name 
                              << " (" << crtc_info->x << "," << crtc_info->y 
                              << " " << crtc_info->width << "x" << crtc_info->height << ")"
                              << (is_primary ? " [PRIMARY]" : "") << std::endl;
                    
                    display_index++;
                }
                
                XRRFreeCrtcInfo(crtc_info);
            }
            
            XRRFreeOutputInfo(output_info);
        }
        
        XRRFreeScreenResources(screen_resources);
        
        // Sort displays by position (left to right, top to bottom)
        std::sort(displays.begin(), displays.end(), 
                 [](const DisplayInfo& a, const DisplayInfo& b) {
                     if (a.y != b.y) return a.y < b.y;
                     return a.x < b.x;
                 });
        
        // Reassign indices after sorting
        for (size_t i = 0; i < displays.size(); i++) {
            displays[i].index = i;
        }
        
        // Find primary display index after sorting
        for (size_t i = 0; i < displays.size(); i++) {
            if (displays[i].is_primary) {
                current_display_index = i;
                break;
            }
        }
    }
    
    const std::vector<DisplayInfo>& getDisplays() const {
        return displays;
    }
    
    const DisplayInfo* getCurrentDisplay() const {
        if (current_display_index >= 0 && current_display_index < displays.size()) {
            return &displays[current_display_index];
        }
        return nullptr;
    }
    
    const DisplayInfo* getDisplayContaining(int x, int y) const {
        for (const auto& display : displays) {
            if (display.containsPoint(x, y)) {
                return &display;
            }
        }
        return nullptr;
    }
    
    const DisplayInfo* getDisplay(int index) const {
        if (index >= 0 && index < displays.size()) {
            return &displays[index];
        }
        return nullptr;
    }
    
    int getDisplayCount() const {
        return displays.size();
    }
    
    void setCurrentDisplay(int index) {
        if (index >= 0 && index < displays.size()) {
            current_display_index = index;
        }
    }
    
    int getCurrentDisplayIndex() const {
        return current_display_index;
    }

    // get global size, ie, the bounding box of all displays
    float4 getGlobalSize() const {
        if (displays.empty()) return float4(0, 0, 0, 0);
        
        int min_x = displays[0].x;
        int min_y = displays[0].y;
        int max_x = displays[0].x + displays[0].width;
        int max_y = displays[0].y + displays[0].height;
        
        for (const auto& display : displays) {
            min_x = std::min(min_x, display.x);
            min_y = std::min(min_y, display.y);
            max_x = std::max(max_x, display.x + display.width);
            max_y = std::max(max_y, display.y + display.height);
        }
        
        return float4(min_x, min_y, max_x - min_x, max_y - min_y);
    }
};

#endif // DISPLAY_MANAGER_HPP