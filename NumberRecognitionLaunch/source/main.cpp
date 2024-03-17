#include <torch/script.h> // One-stop header.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <memory>

class ModuleLoader{
  public:
    torch::jit::script::Module module;
  ModuleLoader(const char loc[]){
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(loc);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    exit(-1);
  }
  std::cout << "Module loaded\n";
  }

};
class ImageLoader{
  public:
    int width, height, channels;
    unsigned char* image_data;
    ImageLoader(const char file[]) : image_data(stbi_load(file, &width, &height, &channels, 1)){
      if (image_data == nullptr) {
        // Error occurred during image loading
        std::cerr << "Error: " << stbi_failure_reason() << std::endl;
        exit(-1);
      }
    }
    ~ImageLoader(){
      stbi_image_free(image_data); // Free memory allocated by stbi_load
    }
    void InvertPixels(){
      for(int i=0;i<width*height*channels;i++){
        image_data[i] = ((unsigned char) 255) - image_data[i];
      }
    }
    void resize(int target_width, int target_height){
      // Allocate memory for the resized image
      unsigned char* target_data = (unsigned char*)malloc(target_width * target_height * channels);
      int result = stbir_resize_uint8(image_data, width , height , 0, target_data, target_width, target_height, 0, 1);
      stbi_image_free(image_data);
      image_data = target_data;
      width=target_width;
      height=target_height;
    }
    void write(char name[]){
      stbi_write_png(name, width, height, channels, image_data, width * channels);
    }
    int OutputModel(torch::jit::script::Module& module){
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::from_blob(image_data, {1, height, width}, torch::kByte).to(torch::kFloat32));
      return(torch::argmax(module.forward(inputs).toTensor()).item<int64_t>());
    }
};

int main(int argc, const char* argv[]) {
  if (argc == 1) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  ImageLoader im_loader = ImageLoader(argv[2]);
  im_loader.InvertPixels();
  im_loader.resize(56,56);

  ModuleLoader mod_loader = ModuleLoader(argv[1]);
  
  // Execute the model and turn its output into a number.
  std::cout << im_loader.OutputModel(mod_loader.module) << '\n';
  
}