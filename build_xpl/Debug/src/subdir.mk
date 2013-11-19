################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/basic_apert.c \
../src/basic_cmp.c \
../src/basic_common.c \
../src/basic_error.c \
../src/basic_img.c \
../src/basic_imgset.c \
../src/basic_itv.c \
../src/basic_mtm.c \
../src/basic_multi.c \
../src/basic_win.c \
../src/basic_xpl.c \
../src/basic_xplBB.c \
../src/basic_xplGG.c \
../src/build_xpl.c \
../src/collec_xpl.c \
../src/collec_xpl_bb.c \
../src/collec_xpl_gg.c \
../src/io_apert.c \
../src/io_header.c \
../src/io_img.c \
../src/io_imgset.c \
../src/io_win.c \
../src/io_xpl.c 

OBJS += \
./src/basic_apert.o \
./src/basic_cmp.o \
./src/basic_common.o \
./src/basic_error.o \
./src/basic_img.o \
./src/basic_imgset.o \
./src/basic_itv.o \
./src/basic_mtm.o \
./src/basic_multi.o \
./src/basic_win.o \
./src/basic_xpl.o \
./src/basic_xplBB.o \
./src/basic_xplGG.o \
./src/build_xpl.o \
./src/collec_xpl.o \
./src/collec_xpl_bb.o \
./src/collec_xpl_gg.o \
./src/io_apert.o \
./src/io_header.o \
./src/io_img.o \
./src/io_imgset.o \
./src/io_win.o \
./src/io_xpl.o 

C_DEPS += \
./src/basic_apert.d \
./src/basic_cmp.d \
./src/basic_common.d \
./src/basic_error.d \
./src/basic_img.d \
./src/basic_imgset.d \
./src/basic_itv.d \
./src/basic_mtm.d \
./src/basic_multi.d \
./src/basic_win.d \
./src/basic_xpl.d \
./src/basic_xplBB.d \
./src/basic_xplGG.d \
./src/build_xpl.d \
./src/collec_xpl.d \
./src/collec_xpl_bb.d \
./src/collec_xpl_gg.d \
./src/io_apert.d \
./src/io_header.d \
./src/io_img.d \
./src/io_imgset.d \
./src/io_win.d \
./src/io_xpl.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


