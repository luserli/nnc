cmake_minimum_required (VERSION 3.0) # 规定cmake的最低版本要求
project(run) # 项目名称
set(CMAKE_CXX_COMPILER "gcc") # 设置编译器

include_directories(${PROJECT_SOURCE_DIR}/Inc) # 添加头文件的搜索路径
aux_source_directory(./Src SrcFiles) # 将源文件列表写在变量SrcFiles中
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build) # 设置可执行文件输出路径
add_executable(run ${SrcFiles}) # 设置可执行文件的名称，make之后bin目录下出现
target_link_libraries(run m)