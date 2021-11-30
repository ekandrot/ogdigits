#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <charconv>

#include "data.h"

//-------------------------------------------------------------------------------------------

void get_all_values(const char *first, const char *last, std::vector<float> &x) {
        int value(0);
        auto res = std::from_chars(first, last, value);
        while (res.ec == std::errc()) {
                float fvalue = value / 255.0;
                fvalue = fvalue * 2 - 1;
                x.push_back(fvalue);
                res = std::from_chars(res.ptr+1, last, value);
        }
}

void TrainingData::load_data(const std::string& fname)
{
        std::ifstream file(fname);
        label = fname;

        std::string str;
        std::getline(file, str);

        // x.reserve(32928000);
        while (std::getline(file, str)) {
                int value(0);
                auto res = std::from_chars(str.c_str(), str.c_str()+str.size(), value);
                Y.push_back(value);
                ++num_examples;
                get_all_values(res.ptr+1, str.c_str()+str.size(), x);
        }
        num_features = x.size() / num_examples;

        std::cout << "Training data, examples:  " << num_examples << std::endl;
        std::cout << "Training data, targets:  " << Y.size() << std::endl;
        std::cout << "Training data, data elements:  " << x.size() << std::endl;
        std::cout << "num of features:  " << num_features << std::endl;
        std::cout << "Training data, data elements, num of features:  " << x.size() / num_examples << std::endl;
}

void TestingData::load_data(const std::string& fname)
{
        std::ifstream file(fname);
        label = fname;

        std::string str;
        std::getline(file, str);

        while (std::getline(file, str)) {
                ++num_examples;
                get_all_values(str.c_str(), str.c_str()+str.size(), x);
        }
        num_features = x.size() / num_examples;

        std::cout << std::endl;
        std::cout << "Test data, examples:  " << num_examples << std::endl;
        std::cout << "Test data, data elements:  " << x.size() << std::endl;
        std::cout << "num of features:  " << num_features << std::endl;
        std::cout << "Test data, data elements, num of features:  " << x.size() / num_examples << std::endl;
}

//-------------------------------------------------------------------------------------------
