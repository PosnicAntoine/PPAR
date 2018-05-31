#include "utils.h"
#include <stdlib.h>

#include "summation_kernel.cu"

// CPU implementation
float log2_series(int n)
{
	float sum = 0.0;
	for(int i=0; i<n; i++){
		sum += (i%2==0 ? 1.0 : -1.0) / (i + 1.0);
	}
	return sum;
}

//same backward
float log2_series_backward(int n){
	float sum = 0.0;
	for(int i=n-1; i>=0; i--){ 
		sum += (i%2==0 ? 1.0 : -1.0) / (i + 1.0);
	}
	return sum;
}

int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    float log2 = log2_series(data_size);
    double start_time = getclock();
    float log22 = log2_series_backward(data_size);
    double end_time = getclock();
    
    printf("CPU result: %f\n", log2);
    printf("CPU result backward: %f\n", log22);

    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);
    
    // Parameter definition
    int threads_per_block = 4 * 32;
    int blocks_in_grid = 8;
    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    
    //Allocating cpu output
    float table[results_size] = {} ;
    float * data_out_cpu;
    
    // Allocating output data on GPU
    cudaMalloc((void **)&data_out_cpu, results_size * sizeof(float));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
    int gap = data_size/num_threads;
    process<<<blocks_in_grid, threads_per_block>>>(data_out_cpu, gap);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
    cudaMemcpy(table, data_out_cpu, (results_size * sizeof(float)),cudaMemcpyDeviceToHost);

    
    // Finish reduction
    float sum = 0.;
    for(int i = results_size -1; i >= 0 ;i--){
        sum = sum + table[i];
    }
    
    // Cleanup
    cudaFree(data_out_cpu);
    
    printf("GPU results:\n");
    printf(" Sum: %f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",	total_time,time_per_iter * 1.e9,bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}

