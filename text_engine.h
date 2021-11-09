#pragma once

#include <string>

enum TEXT_OFFSET {
        WINDOW_OFFSET,
        PIXEL_OFFSET
};

void load_text_engine();

float render_text(const std::string str, int pixels, float x, float y, TEXT_OFFSET which=WINDOW_OFFSET);
