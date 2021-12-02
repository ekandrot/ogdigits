#pragma once

#include <vector>
#include <mutex>
#include <chrono>

#include "ModelStats.h"
#include "data.h"

//-------------------------------------------------------------------------------------------

struct FixedModel {
        FixedModel(int num_features);

        void learn(const TrainingData *data);
        void predict(const Data *data, std::vector<int> &guesses);
        int predict_one(const Data *data, int selector);
        double eval(const TrainingData *data, ModelStats &stats);

        std::mutex mtx;
        std::mutex learning_mtx;                // only one call into learn should ever been running (it can spawn multithread)
        std::chrono::_V2::system_clock::time_point  time_stamp;     // time stamp for last time model was updated

        std::vector<float> layer1;          // input to hidden 1
        std::vector<float> layer2;          // layer 1 to layer 2
        std::vector<float> layer_out;       // layer 2 to output

        std::vector<float> b_hidden1;           // values added to hidden1 after matmul
        std::vector<float> b_hidden2;           // values added to hidden2 after matmul
        std::vector<float> b_output;            // values added to output after matmul

        int num_features;

        // stats during learning
        int16_t last_training_time_ms;
        int epoch;

// private:
//         void predict_one_core();
};

//-------------------------------------------------------------------------------------------
