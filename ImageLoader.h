#pragma once

#include <stdint.h>
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

struct Image {
        int        width;
        int        height;
        int        stride;
        std::vector<uint8_t>    data;
};


Image load_png(fs::path file_name);
Image load_jpeg(fs::path file_name);
