cmake_minimum_required(VERSION 3.16)

project(visual LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 未指定时默认 Debug，方便 F5 单步调试
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 生成 compile_commands.json 供 IntelliSense 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

# 收集 src/ 下所有源文件，新增/删除文件后会自动重新配置
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
)

if(NOT PROJECT_SOURCES)
    message(FATAL_ERROR "在 src/ 下没有找到任何 C++ 源文件")
endif()

add_executable(visual ${PROJECT_SOURCES})

target_include_directories(visual PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visual PRIVATE ${OpenCV_LIBS})

message(STATUS "visual 源文件: ${PROJECT_SOURCES}")
