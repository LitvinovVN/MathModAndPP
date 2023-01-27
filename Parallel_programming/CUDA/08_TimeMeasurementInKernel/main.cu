#include <stdio.h>

const long long delay_clock_cycles = 1000000000;
const int nthr = 3;
const int nTPB = 256;

__global__ void kernel(long long *clocks)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  long long start=clock64();
  

  while (clock64() < start + delay_clock_cycles);
  if (idx < nthr) clocks[idx] = clock64()-start;
}

int main(){

  int peak_clk = 1;
  int device = 0;
  long long *clock_data;
  long long *host_data;
  host_data = (long long *)malloc(nthr*sizeof(long long));
  cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
  if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}
  err = cudaMalloc(&clock_data, nthr*sizeof(long long));
  if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

  kernel<<<(nthr+nTPB-1)/nTPB, nTPB>>>(clock_data);

  err = cudaMemcpy(host_data, clock_data, nthr*sizeof(long long), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}

  for(int i = 0; i < nthr; i++)
  {
    printf("delay clock cycles: %ld, measured clock cycles: %ld, peak clock rate: %dkHz, elapsed time: %fms\n", delay_clock_cycles, host_data[i], peak_clk, host_data[i]/(float)peak_clk);
  }
  return 0;
}