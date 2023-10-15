#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <iomanip> // for std::setw
#include <cstdlib> // for std::atoi and std::atof


void printResults(const std::vector<std::vector<float>>& inputs, const std::vector<float>& correct_outputs, const std::vector<float>& model_outputs) {
    int colWidth = 20;
    std::cout << std::left << std::setw(colWidth) << "Input"
              << std::setw(colWidth) << "Correct Output"
              << std::setw(colWidth) << "Model Output" << std::endl;

    std::cout << std::string(colWidth * 3, '-') << std::endl;

    for (int i = 0; i < inputs.size(); ++i) {
        std::string input_str;
        for (const auto& val : inputs[i]) {
            input_str += std::to_string(val) + " ";
        }
        std::cout << std::left << std::setw(colWidth) << input_str
                  << std::setw(colWidth) << correct_outputs[i]
                  << std::setw(colWidth) << model_outputs[i] << std::endl;
    }
}

// Function to reshape 1D vector to 2D vector
std::vector<std::vector<float>> reshape(const std::vector<float>& flatArray, int rows, int cols) {
    std::vector<std::vector<float>> reshaped(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            reshaped[i][j] = flatArray[i * cols + j];
        }
    }
    return reshaped;
}


float* uploadToGPU(const std::vector<float>& data, size_t dataSize) {
    float* gpu_data;
    cudaMalloc((void**)&gpu_data, dataSize);
    cudaMemcpy(gpu_data, data.data(), dataSize, cudaMemcpyHostToDevice);
    return gpu_data;
}

// Function to initialize and return weights
std::vector<float> initWeights(int rows, int cols) {
    std::vector<float> weights_flat(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier initialization
    float variance = 1.0 / static_cast<float>(rows);
    float stddev = std::sqrt(variance);
    std::normal_distribution<float> d(0, stddev);

    // Initialize weights in flat array
    for (int i = 0; i < rows * cols; ++i) {
        weights_flat[i] = d(gen);
    }

    return weights_flat;
}


// CUDA kernel for matrix addition (for bias)
__global__ void matAdd(float* A, float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < M) {
        C[row * M + col] = A[row * M + col] + B[col];
    }
}

// CUDA kernel for naive matrix multiplication
__global__ void matMul(float* A, float* B, float* C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < M) {
        float val = 0.0;
        for(int i = 0; i < K; ++i) {
            val += A[row * K + i] * B[i * M + col];
        }
        C[row * M + col] = val;
    }
}

// CUDA kernel for sigmoid activation
__global__ void sigmoidActivation(float* A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        A[idx] = 1.0 / (1.0 + exp(-A[idx]));
    }
}

// Forward pass for 2-layer network
void forwardNet(float* gpu_inputData, float* gpu_outputLayerData, float* gpu_hiddenLayerData,
                float* gpu_W1, float* gpu_B1, float* gpu_W2, float* gpu_B2,
                int num_samples, int input_nodes, int hidden_nodes, int output_nodes)
{
    // Forward pass
    dim3 blockDim(16, 16);
    dim3 gridDim((hidden_nodes + blockDim.x - 1) / blockDim.x, (num_samples + blockDim.y - 1) / blockDim.y);

    // Input to Hidden Layer
    matMul<<<gridDim, blockDim>>>(gpu_inputData, gpu_W1, gpu_hiddenLayerData, num_samples, input_nodes, hidden_nodes);
    matAdd<<<gridDim, blockDim>>>(gpu_hiddenLayerData, gpu_B1, gpu_hiddenLayerData, num_samples, hidden_nodes);

    // Apply Activation Function
    int totalHiddenElements = num_samples * hidden_nodes;
    sigmoidActivation<<<(totalHiddenElements + 255)/256, 256>>>(gpu_hiddenLayerData, totalHiddenElements);

    // Hidden to Output
    gridDim = dim3((output_nodes + blockDim.x - 1) / blockDim.x, (num_samples + blockDim.y - 1) / blockDim.y);
    matMul<<<gridDim, blockDim>>>(gpu_hiddenLayerData, gpu_W2, gpu_outputLayerData, num_samples, hidden_nodes, output_nodes);
    matAdd<<<gridDim, blockDim>>>(gpu_outputLayerData, gpu_B2, gpu_outputLayerData, num_samples, output_nodes);

    // Apply Activation Function on Output Layer
    int totalOutputElements = num_samples * output_nodes;
    sigmoidActivation<<<(totalOutputElements + 255)/256, 256>>>(gpu_outputLayerData, totalOutputElements);
}

// CUDA kernel for binary cross-entropy loss calcuation
__global__ void crossEntropyLoss(float* predictions, float* labels, float* loss, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        loss[idx] = -labels[idx] * logf(predictions[idx] + 1e-5) - (1 - labels[idx]) * logf(1 - predictions[idx] + 1e-5);
    }
}

// CUDA kernel for gradient of the loss with respect to the output (dL/dy)
__global__ void gradientOutput(float* d_outputLayerData, float* d_outputLabels, float* d_gradientOutput, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_gradientOutput[idx] = d_outputLayerData[idx] - d_outputLabels[idx];
    }
}

// CUDA kernel for computing the gradients with respect to the hidden layer (dL/dh)
__global__ void gradientHidden(float* d_gradientOutput, float* d_W2, float* d_hiddenLayerData, float* d_gradientHidden, int N, int M, int O) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float val = 0.0;
        for (int o = 0; o < O; ++o) {
            val += d_W2[col * O + o] * d_gradientOutput[row * O + o];
        }
        val *= d_hiddenLayerData[row * M + col] * (1 - d_hiddenLayerData[row * M + col]);  // Sigmoid derivative
        d_gradientHidden[row * M + col] = val;
    }
}

// CUDA kernel for weight updates
__global__ void updateWeights(float* d_W, float* d_inputs, float* d_gradients, float learningRate, int N, int M, int O) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float val = 0.0;
        for (int o = 0; o < O; ++o) {
            val += d_inputs[row * O + o] * d_gradients[o * M + col];
        }
        d_W[row * M + col] -= learningRate * val;
    }
}

// CUDA kernel for bias updates
__global__ void updateBiases(float* d_B, float* d_gradients, float learningRate, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        float val = 0.0;
        val = d_gradients[idx];
        d_B[idx] -= learningRate * val;
    }
}

// Function for backward pass of the neural network
void backwardNet(float* gpu_outputLayerData, float* gpu_outputData, float* gpu_hiddenLayerData, float* gpu_W1, float* gpu_W2, float* gpu_B1, float* gpu_B2, int num_samples, int input_nodes, int hidden_nodes, int output_nodes, float learningRate) {
    // Allocate GPU memory for gradients
    float* d_gradientOutput;
    float* d_gradientHidden;
    cudaMalloc((void**)&d_gradientOutput, sizeof(float) * num_samples * output_nodes);
    cudaMalloc((void**)&d_gradientHidden, sizeof(float) * num_samples * hidden_nodes);

    // Compute gradient of loss with respect to output
    int totalOutputElements = num_samples * output_nodes;
    gradientOutput<<<(totalOutputElements + 255)/256, 256>>>(gpu_outputLayerData, gpu_outputData, d_gradientOutput, totalOutputElements);

    // Compute gradient of loss with respect to hidden layer
    dim3 blockDim(16, 16);
    dim3 gridDim((hidden_nodes + blockDim.x - 1) / blockDim.x, (num_samples + blockDim.y - 1) / blockDim.y);
    gradientHidden<<<gridDim, blockDim>>>(d_gradientOutput, gpu_W2, gpu_hiddenLayerData, d_gradientHidden, num_samples, hidden_nodes, output_nodes);

    // Update weights and biases
    updateWeights<<<gridDim, blockDim>>>(gpu_W2, gpu_hiddenLayerData, d_gradientOutput, learningRate, hidden_nodes, output_nodes, num_samples);
    updateBiases<<<(output_nodes + 255)/256, 256>>>(gpu_B2, d_gradientOutput, learningRate, output_nodes);

    gridDim = dim3((input_nodes + blockDim.x - 1) / blockDim.x, (num_samples + blockDim.y - 1) / blockDim.y);
    updateWeights<<<gridDim, blockDim>>>(gpu_W1, gpu_hiddenLayerData, d_gradientHidden, learningRate, input_nodes, hidden_nodes, num_samples);
    updateBiases<<<(hidden_nodes + 255)/256, 256>>>(gpu_B1, d_gradientHidden, learningRate, hidden_nodes);

    // Cleanup for gradients
    cudaFree(d_gradientOutput);
    cudaFree(d_gradientHidden);
}


// Function to compute binary cross-entropy loss
float computeLoss(std::vector<float>& trueValues, std::vector<float>& predictedValues) {
    float loss = 0.0f;
    for (size_t i = 0; i < trueValues.size(); ++i) {
        float y = trueValues[i];
        float p = predictedValues[i];
        loss -= y * std::log(p + 1e-5) + (1 - y) * std::log(1 - p + 1e-5);
    }
    return loss / trueValues.size();
}



int main(int argc, char *argv[]) {
    // Check if enough arguments are provided
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [learningRate] [hiddenNodes] [num_batches]\n";
        return 1;
    }
    // Initialize architecture, weights, and biases
    int input_nodes = 2;
    int output_nodes = 1;
    int num_samples = 4;
    // Convert command line arguments to variables
    float learningRate = std::stof(argv[1]); // Convert string to float
    int hidden_nodes = std::stoi(argv[2]);   // Convert string to integer
    int num_batches = std::stoi(argv[3]);  // convert string to integer

    // Initialize weights & biases
    auto W1 = initWeights(input_nodes, hidden_nodes);
    auto W2 = initWeights(hidden_nodes, output_nodes);
    std::vector<float> B1(hidden_nodes, 0.0f);  // Bias for hidden layer
    std::vector<float> B2(output_nodes, 0.0f);  // Bias for output layer

    // Upload weights and biases to GPU
    float* gpu_W1 = uploadToGPU(W1, sizeof(float) * input_nodes * hidden_nodes);
    float* gpu_W2 = uploadToGPU(W2, sizeof(float) * hidden_nodes * output_nodes);

    // Upload to GPU
    float* gpu_B1 = uploadToGPU(B1, sizeof(float) * hidden_nodes);
    float* gpu_B2 = uploadToGPU(B2, sizeof(float) * output_nodes);

    // Prepare data and upload to GPU
    float inputData[4][2] = { {0, 0}, {1, 1}, {0, 1}, {1, 0} };
    float outputData[4] = {0, 0, 1, 1};
    float* gpu_inputData;
    float* gpu_outputData;
    cudaMalloc((void**)&gpu_inputData, sizeof(float) * 4 * 2);
    cudaMalloc((void**)&gpu_outputData, sizeof(float) * 4);
    cudaMemcpy(gpu_inputData, inputData, sizeof(float) * 4 * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_outputData, outputData, sizeof(float) * 4, cudaMemcpyHostToDevice);

    // Prepare and allocate hidden and output layer data
    float* gpu_hiddenLayerData;
    float* gpu_outputLayerData;
    cudaMalloc((void**)&gpu_hiddenLayerData, sizeof(float) * num_samples * hidden_nodes);
    cudaMalloc((void**)&gpu_outputLayerData, sizeof(float) * num_samples * output_nodes);

    int totalOutputElements = num_samples * output_nodes;
    std::vector<float> correct_outputs = {0, 0, 1, 1}; // Ground truth

    for (int i = 1; i <= num_batches; ++i) {
        // Forward pass
        forwardNet(gpu_inputData, gpu_outputLayerData, gpu_hiddenLayerData,
               gpu_W1, gpu_B1, gpu_W2, gpu_B2,
               num_samples, input_nodes, hidden_nodes, output_nodes);

        // Backward pass
        backwardNet(gpu_outputLayerData, gpu_outputData, gpu_hiddenLayerData,
               gpu_W1, gpu_W2, gpu_B1, gpu_B2,
               num_samples, input_nodes, hidden_nodes, output_nodes, learningRate);

        // Download and check output
        float outputLayerData[4];  // Adjust size accordingly
        cudaMemcpy(outputLayerData, gpu_outputLayerData, sizeof(float) * totalOutputElements, cudaMemcpyDeviceToHost);

        // Convert the array back to a vector for easier handling
        std::vector<float> model_outputs(outputLayerData, outputLayerData + totalOutputElements);

        // Compute loss
        float loss = computeLoss(correct_outputs, model_outputs);

        // Print current iteration and loss every 10 batches
        if (i % 10 == 0) {
            std::cout << "Batch " << i << ", Loss: " << loss << std::endl;
        }
    }

    // Download and check output
    float outputLayerData[4];  // Adjust size accordingly
    cudaMemcpy(outputLayerData, gpu_outputLayerData, sizeof(float) * totalOutputElements, cudaMemcpyDeviceToHost);
    // Convert the array back to a vector for easier handling
    std::vector<float> model_outputs(outputLayerData, outputLayerData + totalOutputElements);
    // Define your input data and correct output data as vectors of vectors and vectors, respectively
    std::vector<std::vector<float>> inputDataVec = { {0, 0}, {1, 1}, {0, 1}, {1, 0} };
    // std::vector<float> correct_outputs = {0, 0, 1, 1};
    // Call the function to print the results
    printResults(inputDataVec, correct_outputs, model_outputs);
    // Cleanup
    cudaFree(gpu_inputData);
    cudaFree(gpu_outputData);
    cudaFree(gpu_W1);
    cudaFree(gpu_W2);
    cudaFree(gpu_hiddenLayerData);
    cudaFree(gpu_outputLayerData);

    return 0;
}

