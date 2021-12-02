#include <math.h>
#include <iostream>
#include <chrono>
#include "fixed_model.h"

//-------------------------------------------------------------------------------------------
inline auto now() noexcept { return std::chrono::high_resolution_clock::now(); }


void activation(float *h, const float *z, int len)
{
        for (int i=0; i<len; ++i) {
                h[i] = tanh(z[i]);
        }
}

void activation(float *x, int len)
{
        for (int i=0; i<len; ++i) {
                x[i] = tanh(x[i]);
        }
}

void relu(float *h, const float *z, int len)
{
        for (int i=0; i<len; ++i) {
                if (z[i] < 0) {
                        h[i] = 0;
                } else {
                        h[i] = z[i];
                }
        }
}

void relu(float *x, int len)
{
        for (int i=0; i<len; ++i) {
                if (x[i] < 0) x[i] = 0;
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


void vect_add(float *output, const std::vector<float> &input)
{
        for (int i=0; i<input.size(); ++i) {
                output[i] += input[i];
        }
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

void init_vector(std::vector<float> &m)
{
        for (int i=0; i<m.size(); ++i) {
                m[i] = (rand() / (double)RAND_MAX) * 2 - 1;
        }
}

//-------------------------------------------------------------------------------------------

FixedModel::FixedModel(int _num_features) : num_features(_num_features), last_training_time_ms(0)
{
        layer1.resize(num_features * 25);
        layer2.resize(25 * 25);
        layer_out.resize(25 * 10);

        b_hidden1.resize(25);
        b_hidden2.resize(25);
        b_output.resize(10);

        init_vector(layer1);
        init_vector(layer2);
        init_vector(layer_out);

        init_vector(b_hidden1);
        init_vector(b_hidden2);
        init_vector(b_output);

        epoch = 0;

        time_stamp = now();
}

void FixedModel::learn(const TrainingData *data)
{
        std::lock_guard<std::mutex> lck(mtx);

        auto start_time = now();

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        const float learn_rate = 1.0/num_examples;

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                float z1[25];
                matmul(example, z1, layer1.data(), num_features, 25);
                vect_add(z1, b_hidden1);
                activation(hidden1, z1, 25);

                float hidden2[25];
                float z2[25];
                matmul(hidden1, z2, layer2.data(), 25, 25);
                vect_add(z2, b_hidden2);
                activation(hidden2, z2, 25);

                float output[10];
                matmul(hidden2, output, layer_out.data(), 25, 10);
                vect_add(output, b_output);
                softmax(output, 10);


                int t[10] = {0};
                t[data->Y[r]] = 1;

                for (int j=0; j<25; ++j) {
                        for (int i=0; i<25; ++i) {
                                for (int k=0; k<10; ++k) {
                                        layer2[i + j*25] += (t[k] - output[k]) * layer_out[j + k*25] * (1 - hidden2[j]*hidden2[j]) * hidden1[i] * learn_rate * 0.01;
                                }
                        }
                }

                for (int j=0; j<10; ++j) {
                        for (int i=0; i<25; ++i) {
                                layer_out[i + j*25] += (t[j] - output[j]) * hidden2[i] * learn_rate;
                        }
                        b_output[j] += (t[j] - output[j]) * learn_rate;
                }
        }

        auto end_time = now();
        last_training_time_ms = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        time_stamp = now();

        // for (int j=0; j<25; ++j) {
        //         for (int i=0; i<25; ++i) {
        //                 std::cout << layer2[i + j*25] << ", ";
        //         }
        //         std::cout << std::endl;
        // }

        ++epoch;
}

double FixedModel::eval(const TrainingData *data, ModelStats &stats)
{
        std::lock_guard<std::mutex> lck(mtx);
        
        auto start_time = now();

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        stats.num_correct = 0;
        stats.num_incorrect = 0;

        double L{0};

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                matmul(example, hidden1, layer1.data(), num_features, 25);
                vect_add(hidden1, b_hidden1);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2.data(), 25, 25);
                vect_add(hidden2, b_hidden2);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out.data(), 25, 10);
                vect_add(output, b_output);
                softmax(output, 10);
                int ans = max_index(output, 10);

                L += -log(output[data->Y[r]]);

                if (data->Y[r] ==  ans) {
                        ++stats.num_correct;
                } else {
                        ++stats.num_incorrect;
                }
        }

        auto end_time = now();
        stats.execution_time = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        return L / num_examples;
}

int FixedModel::predict_one(const Data *data, int selector)
{
        std::unique_lock<std::mutex> lock(mtx, std::try_to_lock);
        if (lock.owns_lock()) {
                const int num_features = data->num_features;
                const int num_examples = data->num_examples;

                const float *example = data->row(selector);
                float hidden1[25];
                matmul(example, hidden1, layer1.data(), num_features, 25);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2.data(), 25, 25);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out.data(), 25, 10);
                // softmax(output, 10);
                int ans = max_index(output, 10);

                return ans;
        }

        return -1;
}

void FixedModel::predict(const Data *data, std::vector<int> &guesses)
{

        return;
/*

        std::lock_guard<std::mutex> lck(mtx);

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        for (int r=0; r<num_examples; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                matmul(example, hidden1, layer1.data(), num_features, 25);
                activation(hidden1, 25);

                float hidden2[25];
                matmul(hidden1, hidden2, layer2.data(), 25, 25);
                activation(hidden2, 25);

                float output[10];
                matmul(hidden2, output, layer_out.data(), 25, 10);
                // softmax(output, 10);
                int ans = max_index(output, 10);

                guesses[r] = ans;
        }
*/
}

//-------------------------------------------------------------------------------------------
