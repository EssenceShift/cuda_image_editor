
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bit"
#include <stdio.h>
#include "bitmap.cpp"
#include <time.h>
#include <ratio>
#include <chrono>

using namespace std::chrono;

cudaError_t addWithCuda(unsigned int* resultCuda, unsigned int* intensity1, unsigned int* intensity2, int width, int height);

__global__ void addKernel(unsigned int* res, unsigned int* int1, unsigned int* int2, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    res[j * width + i] = 255 * (int1[j * width + i] + int2[j * width + i]) / 510;
}

void getBestParam(int& blockSize, int& gridSize, int size) {
    int minGridSize; // The minimum grid size needed to achieve the 
                     // maximum occupancy for a full device launch 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addKernel, 0, 0);
    gridSize = (size + blockSize - 1) / blockSize;

}
unsigned int* withoutCuda(unsigned int* intensity1, unsigned int* intensity2, int height, int width) {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    unsigned int* result = new unsigned int[height * width];

    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            result[j * width + i] = 255 * (intensity1[j * width + i] + intensity2[j * width + i]) / 510;
        }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> time_span = t2 - t1;

    cout << "Without Cuda: " << time_span.count() << " milliseconds.\n";
    return result;
}

int main()
{
    string path1 = "image_1.bmp";
    string path2 = "image_2.bmp";
    int height, width;

    bitmap_image bitmap1{ path1 };
    bitmap_image bitmap2{ path2 };
    height = bitmap1.height();
    width = bitmap1.width();
    unsigned int* intensity1 = new unsigned int[height * width];
    unsigned int* intensity2 = new unsigned int[height * width];
    unsigned int* resultCuda = new unsigned int[height * width];

    unsigned char red, green, blue;
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            bitmap1.get_pixel(i, j, red, green, blue);
            intensity1[j * width + i] = (int)(red + green + blue) / 3;
            bitmap2.get_pixel(i, j, red, green, blue);
            intensity2[j * width + i] = (int)(red + green + blue) / 3;
        }


    unsigned int* result = withoutCuda(intensity1, intensity2, height, width);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            unsigned char rgb = (unsigned char)result[j * width + i];
            bitmap2.set_pixel(i, j, rgb, rgb, rgb);
        }
    }
    bitmap2.save_image("result_withoutCuda.bmp");



    cudaError_t cudaStatus = addWithCuda(resultCuda, intensity1, intensity2, width, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            unsigned char rgb = (unsigned char)resultCuda[j * width + i];
            bitmap2.set_pixel(i, j, rgb, rgb, rgb);
        }
    }
    bitmap2.save_image("result_Cuda.bmp");



    free(intensity1);
    free(intensity2);


    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(unsigned int* resultCuda, unsigned int* intensity1, unsigned int* intensity2, int width, int height)
{
    int size = width * height;
    unsigned int* dev_int1 = 0;
    unsigned int* dev_int2 = 0;
    unsigned int* dev_result = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_result, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_int1, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_int2, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_int1, intensity1, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_int2, intensity2, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    int blockSize;   // The launch configurator returned block size 
    int gridSize;    // The actual grid size needed, based on input size 

    getBestParam(blockSize, gridSize, size);


    high_resolution_clock::time_point t_Cuda1 = high_resolution_clock::now();

    addKernel << < gridSize, blockSize >> > (dev_result, dev_int1, dev_int2, width);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }



    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(resultCuda, dev_result, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    high_resolution_clock::time_point t_Cuda2 = high_resolution_clock::now();
    duration<double, std::milli> time_cuda = t_Cuda2 - t_Cuda1;
    cout << "With Cuda: " << time_cuda.count() << " milliseconds.";

Error:
    cudaFree(dev_int1);
    cudaFree(dev_int2);
    cudaFree(dev_result);

    return cudaStatus;
}
