/*
 * test_algs.cpp
 *
 *  Created on: Oct 03, 2017
 *      Author: Diego Coelho, PhD Candidate, University of Calgary
 *  Description:
 *  	This file describes test the methods described in the kernels for matrix inversion and determinant calculation.
 *
 *
 */

#include "test_algs.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cerrno>
#include <complex>
#include <CL/cl.h>

//#include "../OpenCL/numerical_t.h"

#include "../Sources/numerical_s.h"

#include "../../home/diego/softwares/eigen3.3/Eigen/Dense"

void test_inv_det_lu(){
	/*
	 * This function compares the result from the GPU output using the kernel det_in_lu
	 * with the result from the Eigen library. If the results match, a message is displayed confirming
	 * the correctness of the kernel implementation.
	 */

	srand((unsigned int) time(0));

	//Defining test matrix
	Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(3,3);

	//Forcing it to be conjugate symmetric
/*	A(1,0) = std::conj(static_cast<std::complex<double> >(A(0,1)));
	A(2,0) = std::conj(static_cast<std::complex<double> >(A(0,2)));
	A(2,1) = std::conj(static_cast<std::complex<double> >(A(1,2)));

	A(0,0) = std::abs(static_cast<std::complex<double> >(A(0,0)));
	A(1,1) = std::abs(static_cast<std::complex<double> >(A(1,1)));
	A(2,2) = std::abs(static_cast<std::complex<double> >(A(2,2)));*/

	A = A*A.transpose().conjugate();
	Eigen::LLT<Eigen::MatrixXcd> lltOfA(A);

	std::cout << "The input matrix is " << std::endl;
	std::cout << A << std::endl;
	std::cout << "The inverse of the input matrix, according Eigen, is \n\n" << std::endl;
	std::cout << A.inverse()  << std::endl;
	std::cout << " and det = " << A.determinant() << std::endl;

	//std::cout << "The inverse of Cholesky factorization is\n\n" << std::endl;
	//Eigen::MatrixXcd lA = lltOfA.matrixL();
	//std::cout << lA.inverse() << std::endl;
	//std::cout << "Reconstructing we get \n\n" << lA*lA.transpose().conjugate() << std::endl;

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
	const unsigned int nMatrices = 1;
	size_t clDATA_SIZE = static_cast<size_t>(nMatrices);
	char buildOptions[] = "-I /home/diego/cuda-workspace/SARImageOpenCL/OpenCL";


	mcmatrix inputMatrices[nMatrices];
	mcomplex mczero;
	mczero.a = 0.0; mczero.b = 0.0;
	mcmatrix mcmzero;
	setmcm(&mcmzero.a, &mczero);
	setmcm(&mcmzero.b, &mczero);
	setmcm(&mcmzero.c, &mczero);
	setmcm(&mcmzero.d, &mczero);
	setmcm(&mcmzero.e, &mczero);
	setmcm(&mcmzero.f, &mczero);
	mcmatrix outputMatrices[nMatrices] = {mcmzero};
	mcomplex outputDets[nMatrices] = {mczero};

	//Setting up the input matrix with random entries.
	for(unsigned int i = 0; i < nMatrices; i++){
		std::complex<double> temp = static_cast<std::complex<double> >(A(0,0));
		mcomplex mctemp; mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].a, &mctemp);

		temp = static_cast<std::complex<double> >(A(0,1));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].b, &mctemp);

		temp = static_cast<std::complex<double> >(A(0,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].c, &mctemp);

		temp = static_cast<std::complex<double> >(A(1,1));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].d, &mctemp);

		temp = static_cast<std::complex<double> >(A(1,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].e, &mctemp);

		temp = static_cast<std::complex<double> >(A(2,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].f, &mctemp);
	}

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
	 * Specify the kernel to be used (in case there are many).
	 */

	kernel = clCreateKernel(program, "inv_det_lu", &erri);

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

	//Enqueue kernel
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &clDATA_SIZE, NULL, 0, NULL, NULL);
	if(erri != CL_SUCCESS){
		std::cout << "Error! The kernel was not queued successfully." << std::endl;
		exit(1);
	} else {
		std::cout << "The kernel was queued successfully." << std::endl;
	}

	erri = clFinish(command_queue);
	if(erri != CL_SUCCESS){
		std::cout << "Error! The clFinish function was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The clFinish function was successful." << std::endl;
	}

	/*
	 * Reading the output buffer.
	 */

	erri = clEnqueueReadBuffer(command_queue, outputMatricesBuffer, CL_TRUE, 0, sizeof(mcmatrix)*nMatrices, outputMatrices, 0, NULL, NULL);

	if(erri != CL_SUCCESS){
		std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The reading of data from the buffer was successful." << std::endl;
	}

	/*
	 * Reading the output buffer.
	 */

	erri = clEnqueueReadBuffer(command_queue, outputDetsBuffer, CL_TRUE, 0, sizeof(mcomplex)*nMatrices, outputDets, 0, NULL, NULL);

	if(erri != CL_SUCCESS){
		std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The reading of data from the buffer was successful." << std::endl;
	}

	//Printing the results
	for(unsigned int i = 0; i< nMatrices; i++) {
		std::cout << "inputMatrix[i] = [";
		printmc(inputMatrices[i].a);
		std::cout<< " ";
		printmc(inputMatrices[i].b);
		std::cout << " ";
		printmc(inputMatrices[i].c);
		printmc(inputMatrices[i].d);
		std::cout << " ";
		printmc(inputMatrices[i].e);
		std::cout << " ";
		printmc(inputMatrices[i].f);
		std::cout << "]"<< std::endl;
	}

	for(unsigned int i = 0; i< nMatrices; i++) {
		std::cout << "outputMatrix[i] = [";
		printmc(outputMatrices[i].a);
		std::cout << " ";
		printmc(outputMatrices[i].b);
		std::cout << " ";
		printmc(outputMatrices[i].c);
		std::cout << " ";
		printmc(outputMatrices[i].d);
		std::cout << " ";
		printmc(outputMatrices[i].e);
		std::cout << " ";
		printmc(outputMatrices[i].f);
		std::cout << "]";
		std::cout << " and det = ";
		printmc(outputDets[i]);
		std::cout << std::endl;
	}


	//Cleaning up
	clReleaseMemObject(inputMatricesBuffer);
	clReleaseMemObject(outputMatricesBuffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void test_inv_det_fast(){
	/*
	 * This function compares the result from the GPU output using the kernel det_in_lu
	 * with the result from the Eigen library. If the results match, a message is displayed confirming
	 * the correctness of the kernel implementation.
	 */

	//Defining test matrix
	Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(3,3);
	//Forcing it to be conjugate symmetric
/*
	A(1,0) = std::conj(static_cast<std::complex<double> >(A(0,1)));
	A(2,0) = std::conj(static_cast<std::complex<double> >(A(0,2)));
	A(2,1) = std::conj(static_cast<std::complex<double> >(A(1,2)));

	A(0,0) = std::abs(static_cast<std::complex<double> >(A(0,0)));
	A(1,1) = std::abs(static_cast<std::complex<double> >(A(1,1)));
	A(2,2) = std::abs(static_cast<std::complex<double> >(A(2,2)));
*/

	A = A*A.transpose().conjugate();

	std::cout << "The input matrix is " << std::endl;
	std::cout << A << std::endl;
	std::cout << "The inverse of the input matrix, according Eigen, is " << std::endl;
	std::cout << A.inverse()  << std::endl;
	std::cout << " and det = " << A.determinant() << std::endl;

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
	const unsigned int nMatrices = 1;
	size_t clDATA_SIZE = static_cast<size_t>(nMatrices);
	char buildOptions[] = "-I /home/diego/cuda-workspace/SARImageOpenCL/OpenCL";


	mcmatrix inputMatrices[nMatrices];
	mcomplex mczero;
	mczero.a = 0.0; mczero.b = 0.0;
	mcmatrix mcmzero;
	setmcm(&mcmzero.a, &mczero);
	setmcm(&mcmzero.b, &mczero);
	setmcm(&mcmzero.c, &mczero);
	setmcm(&mcmzero.d, &mczero);
	setmcm(&mcmzero.e, &mczero);
	setmcm(&mcmzero.f, &mczero);
	mcmatrix outputMatrices[nMatrices] = {mcmzero};
	mcomplex outputDets[nMatrices] = {mczero};

	//Setting up the input matrix with random entries.
	for(unsigned int i = 0; i < nMatrices; i++){
		std::complex<double> temp = static_cast<std::complex<double> >(A(0,0));
		mcomplex mctemp; mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].a, &mctemp);

		temp = static_cast<std::complex<double> >(A(0,1));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].b, &mctemp);

		temp = static_cast<std::complex<double> >(A(0,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].c, &mctemp);

		temp = static_cast<std::complex<double> >(A(1,1));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].d, &mctemp);

		temp = static_cast<std::complex<double> >(A(1,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].e, &mctemp);

		temp = static_cast<std::complex<double> >(A(2,2));
		mctemp.a = static_cast<mreal>(temp.real()); mctemp.b = static_cast<mreal>(temp.imag());
		setmcm(&inputMatrices[i].f, &mctemp);
	}

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
	 * Specify the kernel to be used (in case there are many).
	 */

	kernel = clCreateKernel(program, "inv_det_fast", &erri);

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

	//Enqueue kernel
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &clDATA_SIZE, NULL, 0, NULL, NULL);
	if(erri != CL_SUCCESS){
		std::cout << "Error! The kernel was not queued successfully." << std::endl;
		exit(1);
	} else {
		std::cout << "The kernel was queued successfully." << std::endl;
	}

	erri = clFinish(command_queue);
	if(erri != CL_SUCCESS){
		std::cout << "Error! The clFinish function was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The clFinish function was successful." << std::endl;
	}

	/*
	 * Reading the output buffer.
	 */

	erri = clEnqueueReadBuffer(command_queue, outputMatricesBuffer, CL_TRUE, 0, sizeof(mcmatrix)*nMatrices, outputMatrices, 0, NULL, NULL);

	if(erri != CL_SUCCESS){
		std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The reading of data from the buffer was successful." << std::endl;
	}

	/*
	 * Reading the output buffer.
	 */

	erri = clEnqueueReadBuffer(command_queue, outputDetsBuffer, CL_TRUE, 0, sizeof(mcomplex)*nMatrices, outputDets, 0, NULL, NULL);

	if(erri != CL_SUCCESS){
		std::cout << "Error! The reading of data from the buffer was not successful." << std::endl;
		exit(1);
	} else {
		std::cout << "The reading of data from the buffer was successful." << std::endl;
	}

	//Printing the results
	for(unsigned int i = 0; i< nMatrices; i++) {
		std::cout << "inputMatrix[i] = [";
		printmc(inputMatrices[i].a);
		std::cout<< " ";
		printmc(inputMatrices[i].b);
		std::cout << " ";
		printmc(inputMatrices[i].c);
		printmc(inputMatrices[i].d);
		std::cout << " ";
		printmc(inputMatrices[i].e);
		std::cout << " ";
		printmc(inputMatrices[i].f);
		std::cout << "]"<< std::endl;
	}

	for(unsigned int i = 0; i< nMatrices; i++) {
		std::cout << "outputMatrix[i] = [";
		printmc(outputMatrices[i].a);
		std::cout << " ";
		printmc(outputMatrices[i].b);
		std::cout << " ";
		printmc(outputMatrices[i].c);
		printmc(outputMatrices[i].d);
		std::cout << " ";
		printmc(outputMatrices[i].e);
		std::cout << " ";
		printmc(outputMatrices[i].f);
		std::cout << "]";
		std::cout << " and det = ";
		printmc(outputDets[i]);
		std::cout << std::endl;
	}


	//Cleaning up
	clReleaseMemObject(inputMatricesBuffer);
	clReleaseMemObject(outputMatricesBuffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}
