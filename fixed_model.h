#pragma once
#include <vector>
#include "ModelStats.h"
#include "data.h"

//-------------------------------------------------------------------------------------------

struct FixedModel {
        FixedModel(int num_features);
        ~FixedModel();

        void learn(const TrainingData *data);
        void predict(const Data *data, std::vector<int> &guesses);
        int predict_one(const Data *data, int selector);
        void eval(const TrainingData *data, ModelStats &stats);


        float* layer1;          // input to hidden 1
        float* layer2;          // layer 1 to layer 2
        float* layer_out;       // layer 2 to output

        int num_features;

        // stats during learning
        int16_t last_training_time_ms;
};

//-------------------------------------------------------------------------------------------
