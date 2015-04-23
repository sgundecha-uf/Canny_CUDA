//Group no : 6
//Contact email : karan.magiya@ufl.edu

//Data precision(input and output) : 8 - bit unsigned array with binary values
//Data size range : from 2x2 matrix to 2^15x2^15

//Description of code : Implementing Image processing Canny kernel using Nvidia NPP library. The sobel kernel size is 3x3 and the final step, Non-maximum supression is implemented via custom kernels
//Half of the image is set to '1' value.
//The kernel is then executed for 1000 times. And the kernel execution time is averaged over the 1000 values.

#include <memory>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdio.h> 
#include <stdlib.h> 
#include <fstream> 
#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

using namespace std;

__global__ void sobel(Npp8u *s_x, Npp8u *s_y, Npp8u *d_o, int size) // Calculating the final sobel output
{
	int Idx = threadIdx.x + blockIdx.x * 1024; //Get the thread Id
	float sobel;
	int pointer = Idx * size;
	for(int i = 0; i < size; i++) 
	{
		sobel = sqrt(s_x[pointer]*s[pointer] + s_y[pointer]*s_y[pointer]); 
		d_o[pointer] = sobel;
	}
}

__global__ void canny(Npp8u *d_o, int size) // Non-maximum supression
{
	int Idx = threadIdx.x + blockIdx.x * 1024; //Get the thread Id
	int lowThresh = 50;	// Lower threshold
	int highThresh = 140; // Higher threshold
	int m = sqrt(Idx + 1);  // Get the number elements in one row	

	int pointer = Idx * size;

	//Check for the non-maximum supression
	for(int i = 0; i < size; i++) 
	{
		if(d_o[Idx] > highThresh) 
			d_o[Idx] = 255;
		elseif(d_o[Idx] < lowThresh)
			d_o[Idx] = 0;
		else
		{
			if(d_o[Idx - 1] > highThresh || d_o[Idx + 1] > highThresh || d_o[Idx + m] > highThresh || d_o[Idx - m] > highThresh)
				d_o[Idx] = 255;
			else	
				d_o[Idx] = 0;
		}
	}
}

int main()
{

	//Host and Device Array pointers
	Npp8u * h_i;
	Npp8u * h_o;
	Npp8u * d_i;
	Npp8u * d_o_s_x;	   
	Npp8u * d_o_s_y;
	Npp8u * d_o;
	int * size;	
	//Timing varibales
	cudaEvent_t start,stop;
	long double elapsedTime;

	//CREATING EVENTS
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	long double sum[15];

	int count = 0;
	int n;
	for(int m = 0; m < 32767; m++)
	{
		sum[count] = 0;	
		
		//Host and Device memory allocation
		h_i = (Npp8u*)malloc(m * m * sizeof(Npp8u));
		h_o = (Npp8u*)malloc(m * m * sizeof(Npp8u));
		cudaMalloc((Npp8u*) &d_i, m * m * sizeof(Npp8u));
		cudaMalloc((Npp8u*) &d_o_s_x, m * m * sizeof(Npp8u));
		cudaMalloc((Npp8u*) &d_o_s_y, m * m * sizeof(Npp8u));
		cudaMalloc((Npp8u*) &d_o, m * m * sizeof(Npp8u));
		cudaMalloc((int*) &size, sizeof(int));
		//Initializing the input image data
		int x;
		for (int i = 0; i < m * m; i++)
		{
			h_i[i] = 100;              //Input Gray scale image 
			if (i % m < (m/2))
				h_i[i] = 200;
		}

		// Copying the input image to the Device
		cudaMemcpy(d_i, h_i, sizeof(Npp8u) * m * m, cudaMemcpyHostToDevice);

		// Copying the size of the input image to the Device
		cudaMemcpy(size, m, sizeof(int), cudaMemcpyHostToDevice);

		// Defining the number of Blocks and blocks per thread for the custom kernels
		int NUM_BLOCKS, BLOCK_WIDTH;
		BLOCK_WIDTH = m;		
		NUM_BLOCK = m / 1024;
`		if(m % 1024 != 0) // One block can support only 1024 threads. Hence increase number of blocks if needed
			NUM_BLOCK += 1;
		
		NppStatus canny_status;
		
		NppiSize oSrcSize = {m, m}; // ROI of source
		NppiPoint oSrcOffset = {0, 0}; // Offset for the sobel kernel
		
		NppiSize oSizeROI = {m, m};
		
		cudaEventRecord(start,0);

		for (int i = 0; i < 1000; i++)
		{
			//cudaEventRecord(start,0);
			canny_status = nppiFilterSobelHorizBorder_8u_C1R (d_i, m, oSrcSize, oSrcOffset, d_o_s_x, m, oSizeROI, NPP_BORDER_REPLICATE);  // SObel x
			canny_status = nppiFilterSobelVertBorder_8u_C1R (d_i, m, oSrcSize, oSrcOffset, d_o_s_y, m, oSizeROI, NPP_BORDER_REPLICATE); // SObel y
			sobel<<<NUM_BOCKS,BLOCK_WIDTH>>>(d_o_s_x,d_o_s_y,d_o,size); // Sobel final
			canny<<<NUM_BLOCKS,BLOCK_WIDTH>>>(d_o,size); // Canny final
			
		}
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime,start,stop);
			sum[count] = (elapsedTime/1000);		// Final average computed and saved to an array location
		
		//Copy back the output to the host		
		cudaMemcpy(h_o, d_o, sizeof(Nppu8) * n * n, cudaMemcpyDeviceToHost);

		count++;
		
		//Free the allocated memory
		cudaFree(d_o);
		cudaFree(d_o_s_x);
		cudaFree(d_o_s_y);
		cudaFree(d_i);
		cudaFree(size);
		Free(h_i);
		Free(h_o);
	}

	ofstream file;
	file.open("resize_up_result_npp.txt");
	for (int i = 0; i < 15; i++)
	{
		file << "Average time taken for 2^" << i << " = " << sum[i] << endl;
	}
	file.close();
	return 0;
}
	
