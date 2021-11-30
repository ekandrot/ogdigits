#pragma once
#include <vector>
#include "data.h"

//-------------------------------------------------------------------------------------------

struct FixedModel {
        // TrainingData *train;

        void learn(const TrainingData *data);
        void predict(const Data *data, std::vector<int> &guesses);
};

//-------------------------------------------------------------------------------------------
