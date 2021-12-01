#include <math.h>
#include <iostream>
#include <chrono>
#include "fixed_model.h"

//-------------------------------------------------------------------------------------------
inline auto now() noexcept { return std::chrono::high_resolution_clock::now(); }


void activation(float *x, int len)
{
        for (int i=0; i<len; ++i) {
                x[i] = tanh(x[i]);
        }
}

void softmax(float *x, int len)
{
        float sum{0};
        for (int i=0; i<len; ++i) {
                x[i] = exp(x[i]);
                sum += x[i];
        }
        for (int i=0; i<len; ++i) {
                x[i] /= sum;
        }
}

int max_index(float *x, int len)
{
        float max_value{x[0]};
        int max_index{0};
        for (int i=1; i<10; ++i) {
                if (x[i] > max_value) {
                        max_value = x[i];
                        max_index = i;
                }
        }
        return max_index;
}

// matrix is M rows by N columns
// input is M long
// output is N long
void matmul(const float *input, float *output, float *matrix, int M, int N)
{
        for (int j=0; j<N; ++j) {
                float value{0};
                for (int i=0; i<M; ++i) {
                        value += input[i] * matrix[i + M * j];
                }
                output[j] = value;
        }
}

void init_matrix(float *m, int values)
{
        for (int i=0; i<values; ++i) {
                m[i] = (rand() / (double)RAND_MAX) * 2 - 1;
        }
}

//-------------------------------------------------------------------------------------------

FixedModel::FixedModel(int _num_features) : num_features(_num_features), last_training_time_ms(0)
{
        layer1 = new float[num_features * 25];
        layer2 = new float[25 * 25];
        layer_out = new float[25 * 10];

        init_matrix(layer1, num_features * 25);
        init_matrix(layer2, 25 * 25);
        init_matrix(layer_out, 25 * 10);
}

FixedModel::~FixedModel()
{
        delete [] layer1;
        delete [] layer2;
        delete [] layer_out;
}


void FixedModel::learn(const TrainingData *data)
{
        const float learn_rate = 0.001;

        auto start_time = now();

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                matmul(example, hidden1, layer1, num_features, 25);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2, 25, 25);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out, 25, 10);
                softmax(output, 10);
                int ans = max_index(output, 10);


                int t[10] = {0};
                t[data->Y[r]] = 1;

                for (int j=0; j<10; ++j) {
                        for (int i=0; i<25; ++i) {
                                layer_out[i + j*25] += (t[j] - output[j]) * hidden2[i] * learn_rate;
                        }
                }
        }

        auto end_time = now();
        last_training_time_ms = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

void FixedModel::eval(const TrainingData *data, ModelStats &stats)
{
        const float learn_rate = 0.001;

        auto start_time = now();

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        // int* output = new int[num_examples];

        stats.num_correct = 0;
        stats.num_incorrect = 0;

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                matmul(example, hidden1, layer1, num_features, 25);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2, 25, 25);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out, 25, 10);
                softmax(output, 10);
                int ans = max_index(output, 10);


                int t[10] = {0};
                t[data->Y[r]] = 1;

                if (data->Y[r] ==  ans) {
                        ++stats.num_correct;
                } else {
                        ++stats.num_incorrect;
                }
        }

        auto end_time = now();
        stats.execution_time = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

int FixedModel::predict_one(const Data *data, int selector)
{
        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        const float *example = data->row(selector);
        float hidden1[25];
        matmul(example, hidden1, layer1, num_features, 25);
        activation(hidden1, 25);

        float hidden2[25];
        matmul(hidden1, hidden2, layer2, 25, 25);
        activation(hidden2, 25);

        float output[10];
        matmul(hidden2, output, layer_out, 25, 10);
        softmax(output, 10);
        int ans = max_index(output, 10);

        return ans;

}

void FixedModel::predict(const Data *data, std::vector<int> &guesses)
{
        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        // int* output = new int[num_examples];

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                matmul(example, hidden1, layer1, num_features, 25);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2, 25, 25);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out, 25, 10);
                softmax(output, 10);
                int ans = max_index(output, 10);

                guesses[r] = ans;
        }


        // delete [] output;
}

//-------------------------------------------------------------------------------------------
