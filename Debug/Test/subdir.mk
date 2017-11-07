################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Test/test_algs.cpp 

OBJS += \
./Test/test_algs.o 

CPP_DEPS += \
./Test/test_algs.d 


# Each subdirectory must supply rules for building sources it contributes
Test/%.o: ../Test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/share/R/include -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2 -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2/Rcpp/include -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2/RInside/include -G -g -O0 -std=c++11   -odir "Test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/share/R/include -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2 -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2/Rcpp/include -I/home/diego/R/x86_64-pc-linux-gnu-library/3.2/RInside/include -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


