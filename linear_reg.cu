#include <stdio.h>
#include <time.h>

__global__ void computeSigma(float *d_sigma, float *d_x, float *d_y, float *d_theta) {
	// compute sigma for linear regression

	// initialize values
	int idx = threadIdx.x;
	float x = d_x[idx], y = d_y[idx];
	// printf("%d: x = %f y = %f\n", idx, x, y);
	
	// compute sigma using the formula given
	d_sigma[idx] = ((d_theta[0] + d_theta[1] * x) - y) * ((d_theta[0] + d_theta[1] * x) - y);

}

long timediff(clock_t t1, clock_t t2) {
	long elapsed;
	elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
	return elapsed;
}

int main() {
	// load input data on the host
	FILE *fp;
	fp = fopen("ex1data1.txt", "r");
	if (fp == NULL) {
		printf("Couldn't open file\n");
		return 1;
	}

	float h_x[97], h_y[97];

	for (int i = 0; i < 97; i++) {
		fscanf(fp, "%f,%f", &h_x[i], &h_y[i]);
	}

	fclose(fp);

	// generate other data on the host
	int h_size = sizeof(h_y) / sizeof(float);
	float h_sigma[h_size];
	float h_theta[2] = {0, 0};
	clock_t start, end;

	// declare GPU memory pointers
	float * d_sigma, * d_x, * d_y, * d_theta;

	// allocate GPU memory
	cudaMalloc((void **) &d_sigma, sizeof(h_sigma));
	cudaMalloc((void **) &d_x, sizeof(h_x));
	cudaMalloc((void **) &d_y, sizeof(h_y));
	cudaMalloc((void **) &d_theta, sizeof(h_theta));

	// transfer the data to the GPU
	cudaMemcpy(d_sigma, h_sigma, sizeof(h_sigma), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_theta, h_theta, sizeof(h_theta), cudaMemcpyHostToDevice);
	
	start = clock();
	// launch the kernel
	computeSigma<<<1, h_size>>>(d_sigma, d_x, d_y, d_theta);

	// copy back the result to the CPU
	cudaMemcpy(h_sigma, d_sigma, sizeof(h_sigma), cudaMemcpyDeviceToHost);

	// add each value in sigma to compute cost
	float J = 0;
	for (int j = 0; j < h_size; j++) {
		J += h_sigma[j];
	}
	J = J / (2 * h_size);
	end = clock();	
	
	// print out the result
	printf("Initial cost = %f Elapsed time is %lu ms\n", J, (end - start));

	return 0;
}
