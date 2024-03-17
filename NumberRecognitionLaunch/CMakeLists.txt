cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Find Torch
set(DCMAKE_TOOLCHAIN_FILE "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(Torch_DIR "C:/Program Files/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(main source/main.cpp)
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")  # Use the imported target
set_property(TARGET main PROPERTY CXX_STANDARD 17)
