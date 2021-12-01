#pragma once

#include <vector>
#include <string>

struct Data {
        Data() : num_examples(0), num_features(0) {}

        virtual void load_data(const std::string& fname) = 0;

        inline float* row(int r)  {return &x[r * num_features];}
        inline const float* row(int r) const {return &x[r * num_features];}

        std::vector<float> x;
        int num_examples;
        int num_features;

        std::string label;
};


struct TrainingData : public virtual Data {
        void load_data(const std::string& fname) override;

        std::vector<int> Y;
};

struct TestingData : public virtual Data {
        void load_data(const std::string& fname) override;
};

