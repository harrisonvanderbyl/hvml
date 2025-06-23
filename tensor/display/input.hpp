
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


// NLONGS
#define NLONGS(x) (((x) + (sizeof(unsigned long) * 8) - 1) / (sizeof(unsigned long) * 8))
#ifndef DIRECT_INPUT_READER_HPP
#define DIRECT_INPUT_READER_HPP

class DirectInputReader {
private:
    struct InputDevice {
        int fd;
        std::string path;
        std::string name;
        bool is_keyboard;
        bool is_mouse;
        bool is_touchpad;
    };
    
    std::vector<InputDevice> devices;
    std::thread input_thread;
    std::atomic<bool> running{false};
    
public:
    DirectInputReader() {
        discoverInputDevices();
    }
    
    ~DirectInputReader() {
        stop();
    }
    
    void discoverInputDevices() {
        DIR* input_dir = opendir("/dev/input");
        if (!input_dir) {
            std::cerr << "Cannot open /dev/input directory. Run with appropriate permissions." << std::endl;
            return;
        }
        
        struct dirent* entry;
        while ((entry = readdir(input_dir)) != nullptr) {
            if (strncmp(entry->d_name, "event", 5) == 0) {
                std::string device_path = "/dev/input/" + std::string(entry->d_name);
                
                int fd = open(device_path.c_str(), O_RDONLY | O_NONBLOCK);
                if (fd < 0) {
                    continue; // Skip devices we can't open
                }
                
                // Get device name and capabilities
                char device_name[256] = "Unknown";
                ioctl(fd, EVIOCGNAME(sizeof(device_name)), device_name);
                
                // Check device capabilities
                unsigned long evbit[NLONGS(EV_MAX)] = {0};
                ioctl(fd, EVIOCGBIT(0, EV_MAX), evbit);
                
                bool is_keyboard = test_bit(EV_KEY, evbit);
                bool is_mouse = test_bit(EV_REL, evbit) && test_bit(EV_KEY, evbit);
                bool is_touchpad = test_bit(EV_ABS, evbit) && test_bit(EV_KEY, evbit);
                
                // Filter out devices that don't provide input we care about
                if (is_keyboard || is_mouse || is_touchpad) {
                    InputDevice device;
                    device.fd = fd;
                    device.path = device_path;
                    device.name = std::string(device_name);
                    device.is_keyboard = is_keyboard;
                    device.is_mouse = is_mouse;
                    device.is_touchpad = is_touchpad;
                    
                    devices.push_back(device);
                    std::cout << "Found input device: " << device.name << " (" << device_path << ")" << std::endl;
                } else {
                    close(fd);
                }
            }
        }
        closedir(input_dir);
    }
    
    void start(std::function<void(const input_event&, const std::string&)> callback) {
        if (running) return;
        
        running = true;
        input_thread = std::thread([this, callback]() {
            readInputEvents(callback);
        });
    }
    
    void stop() {
        running = false;
        if (input_thread.joinable()) {
            input_thread.join();
        }
        
        for (auto& device : devices) {
            close(device.fd);
        }
        devices.clear();
    }
    
private:
    static bool test_bit(int bit, const unsigned long* array) {
        return (array[bit / (sizeof(unsigned long) * 8)] >> (bit % (sizeof(unsigned long) * 8))) & 1;
    }
    
    void readInputEvents(std::function<void(const input_event&, const std::string&)> callback) {
        std::vector<pollfd> poll_fds;
        
        for (const auto& device : devices) {
            pollfd pfd;
            pfd.fd = device.fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            poll_fds.push_back(pfd);
        }
        
        while (running) {
            int poll_result = poll(poll_fds.data(), poll_fds.size(), 100); // 100ms timeout
            
            if (poll_result > 0) {
                for (size_t i = 0; i < poll_fds.size(); i++) {
                    if (poll_fds[i].revents & POLLIN) {
                        input_event event;
                        ssize_t bytes_read = read(devices[i].fd, &event, sizeof(event));
                        
                        if (bytes_read == sizeof(event)) {
                            callback(event, devices[i].name);
                        }
                    }
                }
            }
        }
    }
};

#endif // DIRECT_INPUT_READER_HPP