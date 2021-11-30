#pragma once

#include <vector>
#include <string>

struct Data {
        Data() : num_examples(0), num_features(0), displayed_index(0) {}

        virtual void load_data(const std::string& fname) = 0;

        std::vector<int> x;
        int num_examples;
        int num_features;

        std::string label;
        int displayed_index;
};


struct TrainingData : public virtual Data {
        virtual void load_data(const std::string& fname);

        std::vector<int> Y;
};

struct TestingData : public virtual Data {
        virtual void load_data(const std::string& fname);
};

