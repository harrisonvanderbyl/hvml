#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <string>
#include <iostream>
#include <vector>

#ifndef SelectedTextReader_hpp
#define SelectedTextReader_hpp
class SelectedTextReader {
private:
    Display* display;
    Window window;
    
public:
    SelectedTextReader(Display* disp, Window win) : display(disp), window(win) {}
    
    // Method 1: Get PRIMARY selection (text selected with mouse)
    std::string getPrimarySelection() {
        Atom primary = XA_PRIMARY;
        Atom utf8_string = XInternAtom(display, "UTF8_STRING", False);
        Atom compound_text = XInternAtom(display, "COMPOUND_TEXT", False);
        Atom text_plain = XInternAtom(display, "text/plain", False);
        
        // Request the selection
        XConvertSelection(display, primary, utf8_string, primary, window, CurrentTime);
        XFlush(display);
        
        // Wait for SelectionNotify event
        XEvent event;
        while (true) {
            XNextEvent(display, &event);
            if (event.type == SelectionNotify) {
                if (event.xselection.property == None) {
                    // Try with COMPOUND_TEXT if UTF8_STRING failed
                    XConvertSelection(display, primary, compound_text, primary, window, CurrentTime);
                    XFlush(display);
                    continue;
                }
                break;
            }
        }
        
        // Get the selection data
        Atom actual_type;
        int actual_format;
        unsigned long nitems, bytes_after;
        unsigned char* data = nullptr;
        
        int result = XGetWindowProperty(display, window, primary,
                                      0, 1024, False, AnyPropertyType,
                                      &actual_type, &actual_format,
                                      &nitems, &bytes_after, &data);
        
        std::string selected_text;
        if (result == Success && data) {
            selected_text = std::string((char*)data);
            XFree(data);
        }
        
        // Clean up
        XDeleteProperty(display, window, primary);
        return selected_text;
    }
};

#endif // SelectedTextReader_hpp