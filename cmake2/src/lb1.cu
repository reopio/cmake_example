#include <iostream>

#include "lb1.h"

#include <cublas.h>

__global__ void add_vec(float *a, float *b, float *c, int32_t n){

    int32_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<n){
        c[i] = a[i] + b[i];
    }

}


void test_cuda(){

    std::cout<<"this is a cuda test for cmake"<<std::endl;
    std::cout<<"this example calculate vec + vec using gpu"<<std::endl
    <<"with both kernel and cublas"<<std::endl;

    int32_t n=3;

    float *a,*b,*c,*ad,*bd,*cd;
    a=new float[n];
    b=new float[n];
    c=new float[n];

    a[0] = 0.1;
    a[1] = 0.2;
    a[2] = 0.3;
    b[0] = 0.4;
    b[1] = 0.5;
    b[2] = 0.6;

    cudaMalloc((void**)&ad,sizeof(float)*n);
    cudaMalloc((void**)&bd,sizeof(float)*n);
    cudaMalloc((void**)&cd,sizeof(float)*n);

    cudaMemcpy(ad,a,sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,sizeof(float)*n,cudaMemcpyHostToDevice);

    add_vec<<<1,512>>>(ad,bd,cd,n);

    cudaDeviceSynchronize();

    cudaMemcpy(c,cd,sizeof(float)*n,cudaMemcpyDeviceToHost);

    std::cout<<"vec a is [0.1, 0.2, 0.3]"<<std::endl;
    std::cout<<"vec b is [0.4, 0.5, 0.6]"<<std::endl;
    std::cout<<"a + b is [" <<c[0]<<", "
                            <<c[1]<<", "
                            <<c[2]
                            <<"]"<<std::endl;
    std::cout<<"calculated by cuda kernel"  <<std::endl
                                            <<std::endl
                                            <<std::endl;
    
    float alpha=1.0;

    cublasSaxpy(n,
                alpha,
                ad,1,
                bd,1);

    cudaMemcpy(b,bd,sizeof(float)*n,cudaMemcpyDeviceToHost);
    std::cout<<"vec a is [0.1, 0.2, 0.3]"<<std::endl;
    std::cout<<"vec b is [0.4, 0.5, 0.6]"<<std::endl;
    std::cout<<"a + b is [" <<b[0]<<", "
                            <<b[1]<<", "
                            <<b[2]
                            <<"]"<<std::endl;
    std::cout<<"calculated by cublas"   <<std::endl;

    delete(a);
    delete(b);
    delete(c);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

}