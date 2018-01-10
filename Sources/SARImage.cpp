/*
 * numerical_t.h
 *
 *  Created on: Sep 28, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cerrno>
#include <ctime>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <CL/cl.h>

#include "support.h"
#include "../OpenCL/numerical_t.h"
#include "numerical_s.h"
#include "../Test/test_algs.h"
#include "../../home/diego/softwares/eigen3.3/Eigen/Dense"

/*
 * Dear reviewer, if you want compile the code in your machine (besides having all the OpenCL libraries set up in your computer),
 * please, edit the data file location depending on where you put it in your computer. The String object storing the data file
 * location is the Rfilename in the main function.
 */

//This determines the number of replicates
const long unsigned int replicates = 1000;

//This determines if we will use real data or synthetic data
const bool SYNTHETIC = true;

/*
 * The testing functions for the kernels were already verified. See the folder Test.
 */

int main(int argc, char* argv[]){


	//Array of input matrices, synthetic or to be read from R file
	mcmatrix* inputMatrices = NULL;
	//The variable that will store the length of the inputMatrices array
	long unsigned int nMatrices = 0;

	if(!SYNTHETIC){
		/*
		 * Reading real data from R file. TO REVIEWER: Edit here to insert the location od the data file in your computer.
		 */
		std::string Rfilename = "/home/diego/isync/dfgc1/Coelho-Cintra-Frery-Dimitrov/data/SanFranciscoImage.rdata";
		read_r_data(Rfilename, &inputMatrices, &nMatrices);
		std::cout << "File data converted to C++ format. We have a total of " << nMatrices << " matrices." << std::endl;

	} else {
		/*
		 * Creating synthetic data to be used for simulation and setting the number of matrices
		 */
		//CL_DEVICE_MAX_MEM_ALLOC returned 525058048 (read in a different application).
		nMatrices = (525058048/sizeof(mcmatrix));//450*600;


		//Allocating space for the inputMatrices array
		inputMatrices = new mcmatrix[nMatrices];

		//Setting up the input matrices with random entries.
		for(unsigned int i = 0; i < nMatrices; i++){
			//Defining test matrix
			Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(3,3);
			//Forcing it to be Hermitian
			A = A*A.transpose().conjugate();

			mcomplex mctemp;
			mctemp.a = A(0,0).real();mctemp.b = A(0,0).imag();
			setmcm(&inputMatrices[i].a, &mctemp);
			mctemp.a = A(0,1).real();mctemp.b = A(0,1).imag();
			setmcm(&inputMatrices[i].b, &mctemp);
			mctemp.a = A(0,2).real();mctemp.b = A(0,2).imag();
			setmcm(&inputMatrices[i].c, &mctemp);
			mctemp.a = A(1,1).real();mctemp.b = A(1,1).imag();
			setmcm(&inputMatrices[i].d, &mctemp);
			mctemp.a = A(1,2).real();mctemp.b = A(1,2).imag();
			setmcm(&inputMatrices[i].e, &mctemp);
			mctemp.a = A(2,2).real();mctemp.b = A(2,2).imag();
			setmcm(&inputMatrices[i].f, &mctemp);
		}

	}


	/*
	 * Specify the files to be created for storing the runtimes
	 * for the Cholesky based approach and the proposed method.
	 */
	std::stringstream nss, repss;
	nss << nMatrices;
	repss << replicates;
	std::string ftimes_lu_name("tLUn");
	ftimes_lu_name = ftimes_lu_name+nss.str()+"rep"+repss.str()+".txt";
	std::string ftimes_fast_name("tfastn");
	ftimes_fast_name = ftimes_fast_name+nss.str()+"rep"+repss.str()+".txt";
	std::ofstream ftimes_lu(ftimes_lu_name);
	std::ofstream ftimes_fast(ftimes_fast_name);

	srand((unsigned int) time(0));

	cl_platform_id platform_id;//stores one platform only
	cl_int erri;
	cl_uint nPlatforms;
	cl_device_id device_id;//stores one device only
	cl_uint nDevices;
	cl_context context;
	cl_context_properties contextProp[3];
	cl_program program;
	cl_command_queue command_queue;
	cl_kernel kernel;
	cl_mem inputMatricesBuffer, inputDetsBuffer, outputMatricesBuffer, outputDetsBuffer;
	size_t clDATA_SIZE = static_cast<size_t>(nMatrices);
	char buildOptions[] = "-I /home/diego/cuda-workspace/SARImageOpenCL/OpenCL";

	timespec time_init, time_end;
	double time_lu = 0.0, time_fast = 0.0;
	std::vector<double> times_lu(replicates, 0.0), times_fast(replicates, 0.0);

	std::string kernel_string_name_lu = "inv_det_lu";
	std::string kernel_string_name_fast = "inv_det_fast";
	std::vector<std::string> kernel_strings(2);
	kernel_strings[0] = kernel_string_name_lu;
	kernel_strings[1] = kernel_string_name_fast;

	mcmatrix* outputMatricesFast = new mcmatrix[nMatrices];
	mcmatrix* outputMatricesLU = new mcmatrix[nMatrices];
	mcomplex* outputDetsLU = new mcomplex[nMatrices];
	mcomplex* outputDetsFast = new mcomplex[nMatrices];




	//Get the number of platforms
	erri = clGetPlatformIDs(0, NULL, &nPlatforms);

	if(erri == CL_INVALID_VALUE){
		std::cout << "Error during the reading of the number of platforms." << std::endl;
		exit(1);
	} else {
		std::cout << "We read a total of " << nPlatforms << " platform(s).\nWe will use only one, however." << std::endl;
	}

	//Get the first platform
	erri = clGetPlatformIDs(nPlatforms, &platform_id, NULL);
	if(erri){
		std::cout << "Unable to retrieve the available platform." << std::endl;
		exit(1);
	}


	//Get the number of GPU devices
	erri = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &nDevices);

	switch (erri) {
		case CL_INVALID_PLATFORM:
			std::cout << "Error! Invalid platform_id argument." << std::endl;
			exit(1);
		case CL_INVALID_DEVICE_TYPE:
			std::cout << "Error! Invalid device type. Most likely, this machine does not possess this particular device (eg., GPU)." << std::endl;
			exit(1);
		case CL_INVALID_VALUE:
			std::cout << "Error! The list of devices is either or the number of devices passes is null." << std::endl;
			exit(1);
		case CL_DEVICE_NOT_FOUND:
			std::cout << "Error! This machine does not possess devices in the requested category." << std::endl;
			exit(1);
		case CL_SUCCESS:
			std::cout << "The number of devices was loaded successfully." << std::endl;
			break;
		default:
			std::cout << "Error! But we could not detect what is the problem with the device allocation." << std::endl;
			exit(1);
	}

	//Get the first GPU device
	erri = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(erri){
		std::cout << "Unable to retrieve a supported GPU device." << std::endl;
		exit(1);
	}


	//Setting he context property
	contextProp[0] = CL_CONTEXT_PLATFORM;
	contextProp[1] = (cl_context_properties) platform_id;
	contextProp[2] = 0;

	context = clCreateContext(contextProp, 1, &device_id, NULL, NULL, &erri);

	switch (erri){
		case CL_INVALID_PLATFORM:
			std::cout << "Error! The platform you passed as argument is invalid." << std::endl;
			exit(1);
		case CL_INVALID_VALUE:
			std::cout << "Error! One of the parameters such as the context property is not valid." << std::endl;
			exit(1);
		case CL_DEVICE_NOT_AVAILABLE:
			std::cout << "Error! The device you requested is currently not available." << std::endl;
			exit(1);
		case CL_OUT_OF_HOST_MEMORY:
			std::cout << "Error! The host CPU could not allocate enough resources in order to create the context." << std::endl;
			exit(1);
		case CL_SUCCESS:
			std::cout << "The context was created successfully." << std::endl;
			break;
		default:
			std::cout << "Error! But we could not detect what is wrong with the context creation." << std::endl;
			exit(1);
	}

	//Create the command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &erri);

	switch(erri){
		case CL_INVALID_CONTEXT:
			std::cout << "Error! Invalid context." << std::endl;
			exit(1);
		case CL_INVALID_DEVICE:
			std::cout << "Error! Invalid device." << std::endl;
			exit(1);
		case CL_INVALID_VALUE:
			std::cout << "Error! Invalid property value." << std::endl;
			exit(1);
		case CL_INVALID_QUEUE_PROPERTIES:
			std::cout << "Error! Properties not supported." << std::endl;
			exit(1);
		case CL_OUT_OF_HOST_MEMORY:
			std::cout << "Error! The host CPU could not allocate enough resources to create the queue." << std::endl;
			exit(1);
		case CL_SUCCESS:
			std::cout << "Command queue created successfully." << std::endl;
			break;
		default:
			std::cout << "Error! But we could not detect what is wrong with the command queue creation." << std::endl;
			exit(1);
	}

	//Reading the kernel from the file Sources/mkernel.cl
	std::ifstream kernel_file("./OpenCL/mkernel.cl");
	std::stringstream kernel_stream_content;
	kernel_stream_content << kernel_file.rdbuf();
	std::string kernel_content = kernel_stream_content.str();
	const char* kernel_chars = kernel_content.c_str();

	//Create program with the kernel source
	program = clCreateProgramWithSource(context, 1, (const char**) &kernel_chars, NULL, &erri);

	switch(erri){
		case CL_INVALID_CONTEXT:
			std::cout << "Error! The context is invalid." << std::endl;
			exit(1);
		case CL_INVALID_VALUE:
			std::cout << "Error! Most likely, the string is empty or not null terminated." << std::endl;
			exit(1);
		case CL_OUT_OF_HOST_MEMORY:
			std::cout << "Error! The host CPU could not allocate enough resources to create the queue." << std::endl;
			exit(1);
		case CL_SUCCESS:
			std::cout << "Program created successfully." << std::endl;
			break;
		default:
			std::cout << "Error! But we could not detect what is wrong with the program creation." << std::endl;
			exit(1);
	}

	//Compile the program
	erri = clBuildProgram(program, 0, NULL, (const char*) &buildOptions, NULL, NULL);

	switch(erri){
	case CL_INVALID_PROGRAM:
		std::cout << "Error! Invalid program." << std::endl;
		exit(1);
	case CL_INVALID_VALUE:
		std::cout << "Error! Invalid value for device list." << std::endl;
		exit(1);
	case CL_INVALID_DEVICE:
		std::cout << "Error! Invalid devices." << std::endl;
		exit(1);
	case CL_INVALID_BINARY:{
		std::cout << "Error! Invalid binary. The log message is ..." << std::endl;
		char *p_log_message, *p_option_message;
		size_t err_size;
		//Getting the log message
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &err_size);
		p_log_message = new char[err_size];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, err_size, p_log_message, NULL);
		std::string err_message(p_log_message);
		std::cout << err_message << std::endl;
		//Getting the option message
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &err_size);
		p_option_message = new char[err_size];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, err_size, p_option_message, NULL);
		std::string opt_message(p_option_message);
		std::cout << "\n\nThe option passed to build is...\n\n" << opt_message << std::endl;
		delete p_option_message;
		exit(1);
	}
	case CL_INVALID_BUILD_OPTIONS:
		std::cout << "Error! Invalid build options." << std::endl;
		exit(1);
	case CL_INVALID_OPERATION:
		std::cout << "Error! Invalid operation." << std::endl;
		exit(1);
	case CL_COMPILER_NOT_AVAILABLE:
		std::cout << "Error! Compiler not available." << std::endl;
		exit(1);
	case CL_BUILD_PROGRAM_FAILURE:{
		std::cout << "Error! The build failed. The log message is ..." << std::endl;
		char *p_log_message, *p_option_message;
		size_t err_size;
		//Getting the log message
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &err_size);
		p_log_message = new char[err_size];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, err_size, p_log_message, NULL);
		std::string err_message(p_log_message);
		std::cout << err_message << std::endl;
		//Getting the option message
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &err_size);
		p_option_message = new char[err_size];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, err_size, p_option_message, NULL);
		std::string opt_message(p_option_message);
		std::cout << "\n\nThe option passed to build is...\n\n" << opt_message << std::endl;
		delete p_option_message;
		exit(1);}
	case CL_OUT_OF_RESOURCES:
		std::cout << "Error! The host is out of resources." << std::endl;
		exit(1);
	case CL_OUT_OF_HOST_MEMORY:
		std::cout << "Error! The host can not allocate memory for this program." << std::endl;
		exit(1);
	case CL_SUCCESS:
		std::cout << "Program created successfully." << std::endl;
		break;
	default:
		std::cout << "Error! But we could not detect what is wrong with the program building." << std::endl;
		exit(1);

	}

	/*
	 * Create buffer for inputMatrices.
	 */

	inputMatricesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(mcmatrix)*nMatrices, NULL, &erri);

	switch(erri){
	case CL_INVALID_CONTEXT:
		std::cout << "Error! Invalid context." << std::endl;
		exit(1);
	case CL_INVALID_VALUE:
		std::cout << "Error! Invalid cl_mem_flag value." << std::endl;
		exit(1);
	case CL_INVALID_BUFFER_SIZE:
		std::cout << "Error! Either the buffer size is 0 or beyond the limit of the computing device." << std::endl;
		exit(1);
	case CL_INVALID_HOST_PTR:
		std::cout << "Error! Invalid host pointer." << std::endl;
		exit(1);
	case CL_INVALID_MEM_OBJECT:
		std::cout << "Error! Unable to allocate memory object." << std::endl;
		exit(1);
	case CL_OUT_OF_HOST_MEMORY:
		std::cout << "Error! Unable to allocate OpenCL resources." << std::endl;
		exit(1);
	case CL_SUCCESS:
		std::cout << "Input buffer allocated successfully." << std::endl;
		break;
	default:
		std::cout << "Error! But we could not detect what is wrong with the input buffer creation." << std::endl;
		exit(1);
	}

	/*
	 * Create buffer for outputMatrices.
	 */

	outputMatricesBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(mcmatrix)*nMatrices, NULL, NULL);


	switch(erri){
	case CL_INVALID_CONTEXT:
		std::cout << "Error! Invalid context." << std::endl;
		exit(1);
	case CL_INVALID_VALUE:
		std::cout << "Error! Invalid cl_mem_flag value." << std::endl;
		exit(1);
	case CL_INVALID_BUFFER_SIZE:
		std::cout << "Error! Either the buffer size is 0 or beyond the limit of the computing device." << std::endl;
		exit(1);
	case CL_INVALID_HOST_PTR:
		std::cout << "Error! Invalid host pointer." << std::endl;
		exit(1);
	case CL_INVALID_MEM_OBJECT:
		std::cout << "Error! Unable to allocate memory object." << std::endl;
		exit(1);
	case CL_OUT_OF_HOST_MEMORY:
		std::cout << "Error! Unable to allocate OpenCL resources." << std::endl;
		exit(1);
	case CL_SUCCESS:
		std::cout << "Output buffer allocated successfully." << std::endl;
		break;
	default:
		std::cout << "Error! But we could not detect what is wrong with the output buffer creation." << std::endl;
		exit(1);

	}

	/*
	 * Create buffer for outputDets.
	 */

	outputDetsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(mcomplex)*nMatrices, NULL, NULL);


	switch(erri){
	case CL_INVALID_CONTEXT:
		std::cout << "Error! Invalid context." << std::endl;
		exit(1);
	case CL_INVALID_VALUE:
		std::cout << "Error! Invalid cl_mem_flag value." << std::endl;
		exit(1);
	case CL_INVALID_BUFFER_SIZE:
		std::cout << "Error! Either the buffer size is 0 or beyond the limit of the computing device." << std::endl;
		exit(1);
	case CL_INVALID_HOST_PTR:
		std::cout << "Error! Invalid host pointer." << std::endl;
		exit(1);
	case CL_INVALID_MEM_OBJECT:
		std::cout << "Error! Unable to allocate memory object." << std::endl;
		exit(1);
	case CL_OUT_OF_HOST_MEMORY:
		std::cout << "Error! Unable to allocate OpenCL resources." << std::endl;
		exit(1);
	case CL_SUCCESS:
		std::cout << "Output buffer allocated successfully." << std::endl;
		break;
	default:
		std::cout << "Error! But we could not detect what is wrong with the output buffer creation." << std::endl;
		exit(1);

	}



	/*
	 * We have only two kernels, that is why we run over the for loop only for 2 values of i.
	 */

	for(unsigned int j = 0; j < replicates; j++){

		for(unsigned int i = 0; i < 2; i++){

			/*
			 * Specify the kernel to be used.
			 */

			kernel = clCreateKernel(program, kernel_strings[i].c_str(), &erri);

			switch(erri){
			case CL_INVALID_PROGRAM:
				std::cout << "Error! Invalid program." << std::endl;
				exit(1);
			case CL_INVALID_PROGRAM_EXECUTABLE:
				std::cout << "Error! Invalid program executable." << std::endl;
				exit(1);
			case CL_INVALID_KERNEL_NAME:
				std::cout << "Error! Invalid kernel name." << std::endl;
				exit(1);
			case CL_INVALID_KERNEL_DEFINITION:
				std::cout << "Error! Invalid kernel definition." << std::endl;
				exit(1);
			case CL_INVALID_VALUE:
				std::cout << "Error! Invalid value." << std::endl;
				exit(1);
			case CL_OUT_OF_RESOURCES:
				std::cout << "Error! Unable to allocate OpenCL resources." << std::endl;
				exit(1);
			case CL_OUT_OF_HOST_MEMORY:
				std::cout << "Error! Unable to allocate host resources." << std::endl;
				exit(1);
			case CL_SUCCESS:
				std::cout << "Input buffer allocated successfully." << std::endl;
				break;
			default:
				std::cout << "Error! But we could not detect what is wrong with the input buffer creation." << std::endl;
				exit(1);
			}


			/*
			 * Writing the input data into the buffer.
			 */

			erri = clEnqueueWriteBuffer(command_queue, inputMatricesBuffer, CL_TRUE, 0, sizeof(mcmatrix)*nMatrices, inputMatrices, 0, NULL, NULL);

			if(erri != CL_SUCCESS){
				std::cout << "Error! The writing of data to the buffer was not successful." << std::endl;
				exit(1);
			} else {
				std::cout << "The writing of data to the buffer was successful." << std::endl;
			}

			//Setting kernel arguments
			erri = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputMatricesBuffer);
			switch(erri){
			case CL_INVALID_KERNEL:
				std::cout << "Error! The kernel is invalid." << std::endl;
				exit(1);
			case CL_INVALID_ARG_INDEX:
				std::cout << "Error! Invalid kernel argument." << std::endl;
				exit(1);
			case CL_INVALID_MEM_OBJECT:
				std::cout << "Error! Invalid memory object." << std::endl;
				exit(1);
			case CL_INVALID_ARG_SIZE:
				std::cout << "Error! Invalid argument size." << std::endl;
				exit(1);
			case CL_SUCCESS:
				std::cout << "The first input argument was setup correctly." << std::endl;
				break;
			default:
				std::cout << "Error! But we could not detect what is wrong with setting the kernel arguments." << std::endl;
				exit(1);
			}

			//Setting kernel arguments
			erri = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputMatricesBuffer);
			switch(erri){
			case CL_INVALID_KERNEL:
				std::cout << "Error! The kernel is invalid." << std::endl;
				exit(1);
			case CL_INVALID_ARG_INDEX:
				std::cout << "Error! Invalid kernel argument." << std::endl;
				exit(1);
			case CL_INVALID_MEM_OBJECT:
				std::cout << "Error! Invalid memory object." << std::endl;
				exit(1);
			case CL_INVALID_ARG_SIZE:
				std::cout << "Error! Invalid argument size." << std::endl;
				exit(1);
			case CL_SUCCESS:
				std::cout << "The second input argument was setup correctly." << std::endl;
				break;
			default:
				std::cout << "Error! But we could not detect what is wrong with setting the kernel arguments." << std::endl;
				exit(1);
			}

			//Setting kernel arguments
			erri = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputDetsBuffer);
			switch(erri){
			case CL_INVALID_KERNEL:
				std::cout << "Error! The kernel is invalid." << std::endl;
				exit(1);
			case CL_INVALID_ARG_INDEX:
				std::cout << "Error! Invalid kernel argument." << std::endl;
				exit(1);
			case CL_INVALID_MEM_OBJECT:
				std::cout << "Error! Invalid memory object." << std::endl;
				exit(1);
			case CL_INVALID_ARG_SIZE:
				std::cout << "Error! Invalid argument size." << std::endl;
				exit(1);
			case CL_SUCCESS:
				std::cout << "The second input argument was setup correctly." << std::endl;
				break;
			default:
				std::cout << "Error! But we could not detect what is wrong with setting the kernel arguments." << std::endl;
				exit(1);
			}

			//Starting counting time
			clock_gettime(CLOCK_MONOTONIC, &time_init);

			//Enqueue kernel
			clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &clDATA_SIZE, NULL, 0, NULL, NULL);
			if(erri != CL_SUCCESS){
				std::cout << "Error! The kernel was not queued successfully." << std::endl;
				exit(1);
			}

			erri = clFinish(command_queue);
			if(erri != CL_SUCCESS){
				std::cout << "Error! The clFinish function was not successful." << std::endl;
				exit(1);
			}

			//End counting time
			clock_gettime(CLOCK_MONOTONIC, &time_end);
			if(i == 0){
				times_lu[j] = get_millisecs(diff_time(time_init, time_end));
				ftimes_lu << times_lu[j] << ";..." << std::endl;
			} else {
				times_fast[j] = get_millisecs(diff_time(time_init, time_end));
				ftimes_fast << times_fast[j] << ";..." << std::endl;
			}

			/*
			 * Reading the output buffer.
			 */

			if( i == 0){
				erri = clEnqueueReadBuffer(command_queue, outputMatricesBuffer, CL_TRUE, 0, sizeof(mcmatrix)*nMatrices, outputMatricesLU, 0, NULL, NULL);
			} else {
				erri = clEnqueueReadBuffer(command_queue, outputMatricesBuffer, CL_TRUE, 0, sizeof(mcmatrix)*nMatrices, outputMatricesFast, 0, NULL, NULL);
			}


			if(erri != CL_SUCCESS){
				std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
				exit(1);
			} else {
				std::cout << "The reading of data from the buffer was successful." << std::endl;
			}

			/*
			 * Reading the output buffer.
			 */

			if( i == 0){
				erri = clEnqueueReadBuffer(command_queue, outputDetsBuffer, CL_TRUE, 0, sizeof(mcomplex)*nMatrices, outputDetsLU, 0, NULL, NULL);
			} else {
				erri = clEnqueueReadBuffer(command_queue, outputDetsBuffer, CL_TRUE, 0, sizeof(mcomplex)*nMatrices, outputDetsFast, 0, NULL, NULL);
			}


			if(erri != CL_SUCCESS){
				std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
				exit(1);
			} else {
				std::cout << "The reading of data from the buffer was successful." << std::endl;
			}
		}

		for(unsigned int k = 0; k< nMatrices; k++) {
			double tdnorm = mcmatrix_norm(outputMatricesFast[k], outputMatricesLU[k]);
			double thresholdv = 1e-4;
			if(tdnorm >= thresholdv){
				std::cout << "The matrices in k = " << k << " in replicate j = " << j <<
						" show a norm of " << tdnorm << ", higher than the threshold of " << thresholdv << std::endl;
			}
		}
	}

	std::cout << "**********Statistics for the Cholesky factorization*********" << std::endl;
	double tmean = 0.0;
	for(unsigned int i = 0; i < replicates; i++) tmean += times_lu[i];
	tmean /= replicates;
	std::cout << "Mean time: " << tmean << std::endl;
	std::cout << "Max time: " << *std::max_element(times_lu.begin(), times_lu.end()) << std::endl;
	std::cout << "Min time: " << *std::min_element(times_lu.begin(), times_lu.end()) << std::endl;

	std::cout << "**********Statistics for the proposed  fast algorithm*********" << std::endl;
	tmean = 0.0;
	for(unsigned int i = 0; i < replicates; i++) tmean += times_fast[i];
	tmean /= replicates;
	std::cout << "Mean time: " << tmean << std::endl;
	std::cout << "Max time: " << *std::max_element(times_fast.begin(), times_fast.end()) << std::endl;
	std::cout << "Min time: " << *std::min_element(times_fast.begin(), times_fast.end()) << std::endl;
	tmean = 0.0;
	for(unsigned int i = 0; i < replicates; i++)  tmean += times_lu[i]/times_fast[i];
	tmean /= replicates;
	std::cout << "\n\nMean speedup: " << tmean << std::endl;

	//Closing files
	ftimes_lu.close();
	ftimes_fast.close();

	unsigned int zz = 25;

	std::cout << "The "<<zz<<"th input matrix is:" << std::endl;
	std::cout << "inputMatrices[" << zz <<"].a = " << inputMatrices[zz].a.a <<"+j" << inputMatrices[zz].a.b << std::endl;
	std::cout << "inputMatrices[" << zz <<"].b = " << inputMatrices[zz].b.a <<"+j" << inputMatrices[zz].b.b << std::endl;
	std::cout << "inputMatrices[" << zz <<"].c = " << inputMatrices[zz].c.a <<"+j" << inputMatrices[zz].c.b << std::endl;
	std::cout << "inputMatrices[" << zz <<"].d = " << inputMatrices[zz].d.a <<"+j" << inputMatrices[zz].d.b << std::endl;
	std::cout << "inputMatrices[" << zz <<"].e = " << inputMatrices[zz].e.a <<"+j" << inputMatrices[zz].e.b << std::endl;
	std::cout << "inputMatrices[" << zz <<"].f = " << inputMatrices[zz].f.a <<"+j" << inputMatrices[zz].f.b << std::endl;


	std::cout << "The "<<zz<<"th inverted matrices and determinants are:" << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].a = " << outputMatricesLU[zz].a.a <<"+j" << outputMatricesLU[zz].a.b << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].b = " << outputMatricesLU[zz].b.a <<"+j" << outputMatricesLU[zz].b.b << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].c = " << outputMatricesLU[zz].c.a <<"+j" << outputMatricesLU[zz].c.b << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].d = " << outputMatricesLU[zz].d.a <<"+j" << outputMatricesLU[zz].d.b << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].e = " << outputMatricesLU[zz].e.a <<"+j" << outputMatricesLU[zz].e.b << std::endl;
	std::cout << "outputMatricesLU[" << zz <<"].f = " << outputMatricesLU[zz].f.a <<"+j" << outputMatricesLU[zz].f.b << std::endl;
	std::cout << "outputDetsLU[" << zz <<"].a = " << outputDetsLU[zz].a<<"+j"<< outputDetsLU[zz].b << std::endl;

	std::cout << "outputMatricesFast[" << zz <<"].a = " << outputMatricesFast[zz].a.a <<"+j" << outputMatricesFast[zz].a.b << std::endl;
	std::cout << "outputMatricesFast[" << zz <<"].b = " << outputMatricesFast[zz].b.a <<"+j" << outputMatricesFast[zz].b.b << std::endl;
	std::cout << "outputMatricesFast[" << zz <<"].c = " << outputMatricesFast[zz].c.a <<"+j" << outputMatricesFast[zz].c.b << std::endl;
	std::cout << "outputMatricesFast[" << zz <<"].d = " << outputMatricesFast[zz].d.a <<"+j" << outputMatricesFast[zz].d.b << std::endl;
	std::cout << "outputMatricesFast[" << zz <<"].e = " << outputMatricesFast[zz].e.a <<"+j" << outputMatricesFast[zz].e.b << std::endl;
	std::cout << "outputMatricesFast[" << zz <<"].f = " << outputMatricesFast[zz].f.a <<"+j" << outputMatricesFast[zz].f.b << std::endl;
	std::cout << "outputDetsFast[" << zz <<"].a = " << outputDetsFast[zz].a <<"+j"<< outputDetsFast[zz].b << std::endl;

	if(SYNTHETIC){
		std::cout << "The simulation was performed with synthetic data." << std::endl;
	} else {
		std::cout << "The simulation was performed with real data." << std::endl;
	}

	//Cleaning up
	clReleaseMemObject(inputMatricesBuffer);
	clReleaseMemObject(outputMatricesBuffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete inputMatrices;
	delete outputMatricesLU;
	delete outputMatricesFast;
	delete outputDetsLU;
	delete outputDetsFast;


	return 0;
}
