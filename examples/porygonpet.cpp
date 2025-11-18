#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
#include "file_loaders/gltf.hpp"
#include <ops/ops.h>
#include <string>
#include "display/display.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/rwkv7.hpp"
#include "tensor/display/opengl_renderer.hpp"

class pixel {
    float32x4 offset;
    mat4* parent;
};

class eventQueue {
    
};

int main(){
    int objects = 4;
    Tensor<float, 2> myobjects({objects, 4}, DeviceType::kCPU);
    Tensor vecview = myobjects.view<float32x4,1>({objects});
    for (int i = 0; i < objects; i++) {
        vecview[i] = float32x4(i%512, i/512, 0, 1); // x, y, z, w
    }

    gltf d3model = gltf("porygon","scene.gltf");
    std::cout << d3model << std::endl;

    OpenGLRenderer display(512,512);
    display.setCamera({0, 0, 1.0}, {0, 0, 0}, {0, 1, 0});
    display.loadGLTF(d3model);

    // d3model.skeletons[0][1].rotate(3.1415/2, float32x3(1, 0, 0)); // Rotate the first skeleton
    // d3model.skeletons[0][2].rotate(3.1415/2, float32x3(1, 0, 0)); // Rotate the first skeleton
    // d3model.skeletons[0][3].rotate(3.1415/2, float32x3(1, 0, 0)); // Rotate the first skeleton
    // d3model.skeletons[0][4].rotate(3.1415/2, float32x3(1, 0, 0)); // Rotate the first skeleton
    // d3model.skeletons[0][5].rotate(3.1415/2, float32x3(1, 0, 0)); // Rotate the first skeleton
    float32x3 duckposition = {0, 0, 0};
    float32x3 acceldir = {0, 0,0};
    float32x3 forcedir = {0, 0,0};
    float32x32x2 mouseposition = {0, 0};
    float32x3 taildir = {0, 0, 0};
    float randomwalk = 0.2;
    float legwalk = 0.0;
    float distanceScreen = 1.0;


    display.add_on_update([&](CurrentScreenInputInfo& input_info) {
        // Update the model's position or any other properties here
        if(input_info.isMouseLeftButtonPressed()){
            distanceScreen = distanceScreen*0.9 + 0.5;
        }else{
            distanceScreen = distanceScreen*0.9 + 0.1;
        }
        float32x32x2 ss = input_info.getDisplayManager()->getGlobalSize().zw(); // Get the screen size
        mouseposition = (((input_info.getSelectedArea().xy() / ss)) - float32x32x2(0.5, 0.5)) * 2.0;
        float32x3 mouseposition3 = float32x3(mouseposition.x, mouseposition.y, 0.0); // Convert to 3D position
        // use inverse camera matrix to convert mouse position to world coordinates
        mouseposition3 = (display.getCamera().getViewProjectionMatrix().inverse() * float32x4(mouseposition3.x, mouseposition3.y, mouseposition3.z, 0.0)).xyz();
        // mouseposition3.z = 
        float32x3 dirdir = normalize(mouseposition3 - duckposition); // Calculate the direction vector
        float randomnumber = (rand() % 10000) / 10000.0 ; // Generate a random number between -0.5 and 0.5
        
        randomwalk = randomwalk*0.9 + randomnumber*0.1; // Update the random walk value

        mat4 rotateAroundSlightly = mat4::identity().rotated(3.1415/8*randomwalk , float32x3(0, 0, 1)); // Rotate around the x-axis slightly

        float32x3 newdirdir = rotateAroundSlightly * dirdir; // Apply the rotation to the direction vector
        
        float32x3 direction = (mouseposition3 - duckposition) - newdirdir*0.1; // Calculate the direction vector
        

        acceldir = acceldir * 0.9 + direction * 0.1; // Apply some acceleration towards the mouse position
        forcedir = forcedir * 0.9 + acceldir * 0.1; // Apply some force towards the mouse position
        taildir = taildir * 0.6 + forcedir * 0.4; // Apply some force towards the mouse position
        duckposition = duckposition + forcedir * 0.1; // Move towards the mouse position
        legwalk = legwalk + sqrt(forcedir.x * forcedir.x + forcedir.y * forcedir.y + forcedir.z * forcedir.z) * 0.1; // Update the leg walk value

        mat4 base_transform = mat4::identity().scaled(float32x3(0.1, 0.1, 0.1)).rotated(3.1415, float32x3(0, 0, -1));

        mat4 accelturn = base_transform.pointed_towards_direction(-forcedir, float32x3(0.0,1.0,1.0));
        mat4 directturn = base_transform.pointed_towards_direction(-dirdir, float32x3(0.0,1.0,1.0));

        mat4 legleft  = accelturn * mat4::identity().rotated(0.3*sin(legwalk*25.0), float32x3(1.0,0.0,0.0)) ;
        mat4 legright = accelturn * mat4::identity().rotated(0.3*cos(legwalk*25.0), float32x3(1.0,0.0,0.0)) ;
        mat4 inbetweenturn = base_transform.pointed_towards_direction(-acceldir, float32x3(0.0,1.0,1.0)) * 
        mat4::identity().rotated(0.1*-sin(legwalk*25.0), float32x3(0.0,0.0,1.0));
        mat4 tailturn = base_transform.pointed_towards_direction(-taildir, float32x3(0.0,1.0,1.0)) *
        mat4::identity().rotated(0.1*-sin(legwalk*25.0), float32x3(0.0,1.0,0.0));
        
        
        

        int i= 0;
        d3model.skeletons[0][i++] = directturn.translated(duckposition);
        // one leg
        d3model.skeletons[0][i++] = legleft.translated(duckposition);
        // tail
        d3model.skeletons[0][i++] = tailturn.translated(duckposition);
        // otherleg
        d3model.skeletons[0][i++] = legright.translated(duckposition);
        // head
        d3model.skeletons[0][i++] = directturn.translated(duckposition);
        // main body
        d3model.skeletons[0][i++] = inbetweenturn.translated(duckposition);


        float32x4 duckpositionscreen = display.getCamera().getProjectionViewMatrix().transpose() * float32x4(duckposition.x, duckposition.y, duckposition.z, 1.0f); // Convert to screen coordinates
        duckpositionscreen = duckpositionscreen / duckpositionscreen.w;
        duckpositionscreen = duckpositionscreen * float32x4(0.5, 0.5, 0.5, 1.0); // Convert to normalized device coordinates
        duckpositionscreen = duckpositionscreen + float32x4(0.5, 0.5, 0.5, 0.0); // Convert to normalized device coordinates
        duckpositionscreen = duckpositionscreen * float32x4(ss.x,ss.y, 1.0, 1.0); // Scale to screen size
        // duckpositionscreen = duckpositionscreen * float32x4(input_info.getCurrentDisplayInfo()->width, input_info.getCurrentDisplayInfo()->height, 1.0, 1.0); // Scale to screen size
        // float32x32x2 duckpositionscreenpixel = float32x32x2(duckpositionscreen.x, duckpositionscreen.y) ; // Convert to screen coordinates
        auto duckondisplay = input_info.getDisplayManager()->getDisplayContaining(int(duckpositionscreen.x), int(duckpositionscreen.y));
        if(duckondisplay == nullptr){
            duckondisplay = input_info.getCurrentDisplayInfo();
        }else if (duckondisplay->index != input_info.getCurrentDisplayInfo()->index) {
            display.moveToDisplay(duckondisplay->index);
        }
        
    });
  
    display.displayLoop();
}
