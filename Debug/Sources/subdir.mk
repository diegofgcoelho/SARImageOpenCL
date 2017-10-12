################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Sources/SARImage.cpp \
../Sources/numerical_s.cpp \
../Sources/support.cpp 

OBJS += \
./Sources/SARImage.o \
./Sources/numerical_s.o \
./Sources/support.o 

CPP_DEPS += \
./Sources/SARImage.d \
./Sources/numerical_s.d \
./Sources/support.d 


# Each subdirectory must supply rules for building sources it contributes
Sources/%.o: ../Sources/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11   -odir "Sources" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


