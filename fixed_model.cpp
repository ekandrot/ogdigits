#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
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
void matmul(const float *input, float *output, const float *matrix, int M, int N)
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

// !!! assumes data is locked or local copy outside of this block !!!
//  private, internal only function for the above reason(s)
static void predict_one_core(const Data *data, int index,
        const std::vector<float> &W1,
        const std::vector<float> &W2,
        const std::vector<float> &W3,
        const std::vector<float> &b1,
        const std::vector<float> &b2,
        const std::vector<float> &b3,
        float *output
        )
{
        const int num_features = data->num_features;

        const float *example = data->row(index);
        float hidden1[25];
        matmul(example, hidden1, W1.data(), num_features, 25);
        vect_add(hidden1, b1);
        activation(hidden1, 25);

        float hidden2[25];
        matmul(hidden1, hidden2, W2.data(), 25, 25);
        vect_add(hidden2, b2);
        activation(hidden2, 25);

        matmul(hidden2, output, W3.data(), 25, 10);
        vect_add(output, b3);
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

struct UpdateData {
        std::vector<float> W1;
        std::vector<float> W2;
        std::vector<float> W3;

        std::vector<float> b1;
        std::vector<float> b2;
        std::vector<float> b3;

};


void learn_thread(const TrainingData *data, int first, int last, UpdateData &update)
{
        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        const float learn_rate = 1.0/num_examples;

        for (int r=first; r<last; ++r) {
                const float *example = data->row(r);
                float hidden1[25];
                float z1[25];
                matmul(example, z1, update.W1.data(), num_features, 25);
                vect_add(z1, update.b1);
                activation(hidden1, z1, 25);

                float hidden2[25];
                float z2[25];
                matmul(hidden1, z2, update.W2.data(), 25, 25);
                vect_add(z2, update.b2);
                activation(hidden2, z2, 25);

                float output[10];
                matmul(hidden2, output, update.W3.data(), 25, 10);
                vect_add(output, update.b3);
                softmax(output, 10);

                int t[10] = {0};
                t[data->Y[r]] = 1;

                for (int k=0; k<10; ++k) {
                        float t_output = (t[k] - output[k]);
                        for (int j=0; j<25; ++j) {
                                float dtanh2 = (1 - hidden2[j]*hidden2[j]) * update.W3[j + k*25];
                                for (int i=0; i<25; ++i) {
                                        update.W2[i + j*25] += t_output * dtanh2 * hidden1[i] * learn_rate * 0.1;
                                }
                                update.b2[j] += t_output * dtanh2 * learn_rate * 0.1;
                        }
                }

                for (int j=0; j<10; ++j) {
                        for (int i=0; i<25; ++i) {
                                update.W3[i + j*25] += (t[j] - output[j]) * hidden2[i] * learn_rate;
                        }
                        update.b3[j] += (t[j] - output[j]) * learn_rate;
                }
        }
}


void vec_sum(std::vector<float> &a, const std::vector<float> &b)
{
        for (int i=0; i<a.size(); ++i) {
                a[i] += b[i];
        }
}

void vec_div(std::vector<float> &a, float b)
{
        for (int i=0; i<a.size(); ++i) {
                a[i] /= b;
        }
}


void FixedModel::learn(const TrainingData *data)
{
        // only one learner, since the data is updated at the end
        std::lock_guard<std::mutex> lck(learning_mtx);
        auto start_time = now();

        std::vector<std::thread> pool;
        std::vector<UpdateData> updates(10);

        UpdateData update;
        update.W1 = layer1;
        update.W2 = layer2;
        update.W3 = layer_out;
        update.b1 = b_hidden1;
        update.b2 = b_hidden2;
        update.b3 = b_output;

        int amount_per_thread = data->num_examples / 10;
        for (int i=0; i<10; ++i) {
                updates[i] = update;
                pool.push_back(std::thread(learn_thread, data, amount_per_thread * i, amount_per_thread * (i + 1), std::ref(updates[i])));
        }

        for (int i=0; i<10; ++i) {
                pool[i].join();
        }

        // update the main data from what we've learned
        {
                std::lock_guard<std::mutex> lck(mtx);

                layer1 = updates[0].W1;
                layer2 = updates[0].W2;
                layer_out = updates[0].W3;

                b_hidden1 = updates[0].b1;
                b_hidden2 = updates[0].b2;
                b_output = updates[0].b3;

                for (int i=1; i<10; ++i) {
                        vec_sum(layer1, updates[i].W1);
                        vec_sum(layer2, updates[i].W2);
                        vec_sum(layer_out, updates[i].W3);

                        vec_sum(b_hidden1, updates[i].b1);
                        vec_sum(b_hidden2, updates[i].b2);
                        vec_sum(b_output, updates[i].b3);
                }

                vec_div(layer1, 10);
                vec_div(layer2, 10);
                vec_div(layer_out, 10);

                vec_div(b_hidden1, 10);
                vec_div(b_hidden2, 10);
                vec_div(b_output, 10);

        }

        ++epoch;

        auto end_time = now();
        last_training_time_ms = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        time_stamp = now();
}


// this function can take a second or so, so maybe copy the data locally inside of a lock?
double FixedModel::eval(const TrainingData *data, ModelStats &stats)
{
        // since we only lock around the model object's data read/write, we need a function lock above
        std::unique_lock<std::mutex> lock(mtx);

        // make local copies so that others can acquire the lock on the main model data
        std::vector<float> W1(layer1);
        std::vector<float> W2(layer2);
        std::vector<float> W3(layer_out);

        std::vector<float> b1(b_hidden1);
        std::vector<float> b2(b_hidden2);
        std::vector<float> b3(b_output);

        lock.unlock();

        auto start_time = now();

        const int num_features = data->num_features;
        const int num_examples = data->num_examples;

        stats.num_correct = 0;
        stats.num_incorrect = 0;

        double L{0};

        for (int r=0; r<num_examples; ++r) {
                float output[10];
                predict_one_core(data, r, W1, W2, W3, b1, b2, b3, output);
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

// this function should be very fast, so any data locks shouldn't block real work noticable
int FixedModel::predict_one(const Data *data, int selector)
{
        std::lock_guard<std::mutex> lck(mtx);

        float output[10];
        predict_one_core(data, selector, layer1, layer2, layer_out, b_hidden1, b_hidden2, b_output, output);
        int ans = max_index(output, 10);

        return ans;
}

// this function can take a second or so, so maybe copy the data locally inside of a lock?
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
