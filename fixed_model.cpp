#include <math.h>
#include "fixed_model.h"

//-------------------------------------------------------------------------------------------

float activation(float x)
{
        return tanh(x);
}

void init_matrix(float *m, int values)
{
        for (int i=0; i<values; ++i) {
                m[i] = (rand() / (double)RAND_MAX) * 2 - 1;
        }
}


void FixedModel::learn(const TrainingData *data)
{

}

void FixedModel::predict(const Data *data, std::vector<int> &guesses)
{
        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        float* layer1 = new float[num_features * 25];  // input to hidden 1
        float* layer2 = new float[25 * 25];            // layer 1 to layer 2
        float* layer_out = new float[25 * 10];         // layer 2 to output

        init_matrix(layer1, num_features * 25);
        init_matrix(layer2, 25 * 25);
        init_matrix(layer_out, 25 * 10);

        // int* output = new int[num_examples];

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                for (int j=0; j<25; ++j) {
                        float value{0};
                        for (int i=0; i<num_features; ++i) {
                                value += example[i] * layer1[i + num_features * j];
                        }
                        hidden1[j] = value;
                }
                for (int i=0; i<25; ++i) {
                        hidden1[i] = activation(hidden1[i]);
                }

                float hidden2[25];
                for (int j=0; j<25; ++j) {
                        float value{0};
                        for (int i=0; i<25; ++i) {
                                value += hidden1[i] * layer2[i + 25 * j];
                        }
                        hidden2[j] = value;
                }
                for (int i=0; i<25; ++i) {
                        hidden1[i] = activation(hidden1[i]);
                }

                float output[10];
                for (int j=0; j<10; ++j) {
                        float value{0};
                        for (int i=0; i<25; ++i) {
                                value += hidden2[i] * layer_out[i + 25 * j];
                        }
                        output[j] = value;
                }

                float max_value{output[0]};
                int max_index{0};
                for (int i=1; i<10; ++i) {
                        if (output[i] > max_value) {
                                max_value = output[i];
                                max_index = i;
                        }
                }

                guesses[r] = max_index;
        }


        delete [] layer1;
        delete [] layer2;
        delete [] layer_out;
        // delete [] output;
}

//-------------------------------------------------------------------------------------------
