#include <iostream>
#include <unordered_map>
#include <tuple>
#include <functional>

// Define a 3D coordinate structure
struct Point3D {
    int x, y, z;
    
    // Constructor
    Point3D(int x_val = 0, int y_val = 0, int z_val = 0) 
        : x(x_val), y(y_val), z(z_val) {}
    
    // Equality operator for comparison
    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Custom hash function for Point3D
struct Point3DHash {
    std::size_t operator()(const Point3D& point) const {
        // Combine the hash of the three coordinates
        std::size_t h1 = std::hash<int>{}(point.x);
        std::size_t h2 = std::hash<int>{}(point.y);
        std::size_t h3 = std::hash<int>{}(point.z);
        
        // Combine hashes using a technique from Boost
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2)) ^ 
               (h3 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2));
    }
};

// Define our 3D HashMap class
template <typename T>
class HashMap3D {
private:
    std::unordered_map<Point3D, T, Point3DHash> map;
    
public:
    // Insert a value at the specified coordinates
    void insert(int x, int y, int z, const T& value) {
        map[Point3D(x, y, z)] = value;
    }
    
    // Get the value at the specified coordinates
    // Returns a pointer to the value if found, nullptr otherwise
    T* get(int x, int y, int z) {
        Point3D point(x, y, z);
        auto it = map.find(point);
        if (it != map.end()) {
            return &(it->second);
        }
        return nullptr;
    }
    
    // Check if a key exists
    bool contains(int x, int y, int z) {
        return map.find(Point3D(x, y, z)) != map.end();
    }
    
    // Remove a value at the specified coordinates
    // Returns true if something was removed, false otherwise
    bool remove(int x, int y, int z) {
        Point3D point(x, y, z);
        return map.erase(point) > 0;
    }
    
    // Get the number of elements in the map
    size_t size() const {
        return map.size();
    }
    
    // Clear all elements
    void clear() {
        map.clear();
    }
    
    // Iterate through all elements
    void forEach(std::function<void(int x, int y, int z, T& value)> callback) {
        for (auto& pair : map) {
            callback(pair.first.x, pair.first.y, pair.first.z, pair.second);
        }
    }
};

// // Example usage
// int main() {
//     // Create a 3D HashMap to store string values
//     HashMap3D<std::string> worldMap;
    
//     // Insert some values
//     worldMap.insert(0, 0, 0, "Origin");
//     worldMap.insert(1, 2, 3, "Point A");
//     worldMap.insert(-5, 10, 7, "Point B");
    
//     // Retrieve and print values
//     std::string* value = worldMap.get(1, 2, 3);
//     if (value) {
//         std::cout << "Value at (1,2,3): " << *value << std::endl;
//     }
    
//     // Check if a point exists
//     if (worldMap.contains(0, 0, 0)) {
//         std::cout << "Origin exists in the map" << std::endl;
//     }
    
//     // Iterate through all points
//     std::cout << "All points in the map:" << std::endl;
//     worldMap.forEach([](int x, int y, int z, std::string& value) {
//         std::cout << "(" << x << "," << y << "," << z << "): " << value << std::endl;
//     });
    
//     // Remove a point
//     if (worldMap.remove(-5, 10, 7)) {
//         std::cout << "Point B removed successfully" << std::endl;
//     }
    
//     // Print the size
//     std::cout << "Map size: " << worldMap.size() << std::endl;
    
//     return 0;
// }