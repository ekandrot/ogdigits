#pragma once

#include <stdint.h>

struct image {
        int        width;
        int        height;
        int        stride;
        uint8_t    *data;
};


uint8_t* load_png(const char *file_name);
image load_jpeg(const char *file_name);

