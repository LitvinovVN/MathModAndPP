#include <iostream>
__global__ void myKernel(void) {
    printf("Hello CUDA!\n");
}


int main(void) {
	myKernel <<<1, 1>>>();
	
	return 0;
}