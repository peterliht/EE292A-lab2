#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
#include <fstream>
// #include <time.h>
// #include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../../shared/defines.h"
#include "../../shared/utils.h"
#include "assert.h"

// TODO: If you want to define constants, you can do it here
#define CONV1_WEIGHT_SIZE (5 * 5 * 1 * 32)   // filter size * channel * num_filters
#define CONV1_BIAS_COUNT 32  
#define CONV2_WEIGHT_SIZE (5 * 5 * 32 * 64) 
#define CONV2_BIAS_COUNT 64
#define DENSE1_WEIGHT_SIZE (7 * 7 * 64 * 256)
#define DENSE1_BIAS_COUNT 256
#define DENSE2_WEIGHT_SIZE (256 * 10)
#define DENSE2_BIAS_COUNT 10

using namespace aocl_utils;

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;

cl_uchar *input_images = NULL, *output_guesses = NULL, *reference_guesses = NULL;
cl_float *input_weights = NULL;
cl_mem input_images_buffer, output_guesses_buffer;

// TODO: add buffers for your weights
cl_float *conv1_w = NULL, *conv1_b = NULL, *conv2_w = NULL, *conv2_b = NULL;
cl_float *dense1_w = NULL, *dense1_b = NULL, *dense2_w = NULL, *dense2_b = NULL;

cl_mem conv1_w_buffer, conv1_b_buffer, conv2_w_buffer, conv2_b_buffer;
cl_mem dense1_w_buffer, dense1_b_buffer, dense2_w_buffer, dense2_b_buffer;

// Global variables.
std::string imagesFilename;
std::string labelsFilename;
std::string aocxFilename;
std::string deviceInfo;
unsigned int n_items;
bool use_fixed_point;
bool use_single_workitem;

// Function prototypes.
void classify();
void initCL();
void cleanup();
void teardown(int exit_status = 1);


int main(int argc, char **argv) {
	// Parsing command line arguments.
	Options options(argc, argv);

	if(options.has("images")) {
		imagesFilename = options.get<std::string>("images");
	} else {
		imagesFilename = "t10k-images.idx3-ubyte";
		printf("Defaulting to images file \"%s\"\n", imagesFilename.c_str());
	}
	
	if(options.has("labels")) {
		labelsFilename = options.get<std::string>("labels");  
	} else {
		labelsFilename = "t10k-labels.idx1-ubyte";
		printf("Defaulting to labels file \"%s\"\n", labelsFilename.c_str());
	}

	// Relative path to aocx filename option.
	if(options.has("aocx")) {
		aocxFilename = options.get<std::string>("aocx");  
	} else {
		aocxFilename = "linear_classifier_fp";
		printf("Defaulting to aocx file \"%s\"\n", aocxFilename.c_str());
	}
	
	// Read in the images and labels
	n_items = parse_MNIST_images(imagesFilename.c_str(), &input_images);
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		return -1;
	}
	if (n_items != parse_MNIST_labels(labelsFilename.c_str(), &reference_guesses)){
		printf("ERROR: Number of labels does not match number of images\n");
		return -1;
	}

	// TODO: Uncomment this to verify on a smaller set of examples
	n_items = 100;
	
	// Initializing OpenCL and the kernels.
	output_guesses = (cl_uchar*)alignedMalloc(sizeof(cl_uchar) * n_items);
	
	// TODO: Allocate space for weights if you so desire. To help you out, here's the declaration from last time:	
	// input_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * FEATURE_COUNT * NUM_DIGITS);
	conv1_w = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV1_WEIGHT_SIZE);
	conv1_b = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV1_BIAS_COUNT);
	conv2_w = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV2_WEIGHT_SIZE);
	conv2_b = (cl_float*)alignedMalloc(sizeof(cl_float) * CONV2_BIAS_COUNT);
	dense1_w = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE1_WEIGHT_SIZE);
	dense1_b = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE1_BIAS_COUNT);
	dense2_w = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE2_WEIGHT_SIZE);
	dense2_b = (cl_float*)alignedMalloc(sizeof(cl_float) * DENSE2_BIAS_COUNT);

	// TODO: Read in weights from weights files
	bool check = read_weights_file("weights/conv1_weights", conv1_w, CONV1_WEIGHT_SIZE);
    assert(check);
	read_weights_file("weights/conv1_bias", conv1_b, CONV1_BIAS_COUNT);
	read_weights_file("weights/conv2_weights", conv2_w, CONV2_WEIGHT_SIZE);
	read_weights_file("weights/conv2_bias", conv2_b, CONV2_BIAS_COUNT);
	read_weights_file("weights/dense1_weights", dense1_w, DENSE1_WEIGHT_SIZE);
	read_weights_file("weights/dense1_bias", dense1_b, DENSE1_BIAS_COUNT);
	read_weights_file("weights/dense2_weights", dense2_w, DENSE2_WEIGHT_SIZE);
	read_weights_file("weights/dense2_bias", dense2_b, DENSE2_BIAS_COUNT);


	initCL();

	// Start measuring time
	double start = get_wall_time();
	
	// Call the classifier.
	classify();
	
	// Stop measuring time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
   
	int correct = 0;
	for (unsigned i = 0; i < n_items; i++){
		if (output_guesses[i] == reference_guesses[i]) correct++;
	}
	printf("Classifier accuracy: %.2f%%\n", (float)correct*100/n_items);
	
	// Teardown OpenCL.
	teardown(0);
}

void classify() {
	size_t size = 1;
	cl_int status;
	cl_event event;
	const size_t global_work_size = n_items;
	
	// Create kernel input and output buffers.
	input_images_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * FEATURE_COUNT * n_items, NULL, &status);
	checkError(status, "Error: could not create input image buffer");
	output_guesses_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * n_items, NULL, &status);
	checkError(status, "Error: could not create output guesses buffer");
	
	// TODO: Add buffers for layer weights
    // bug?: CL_MEM_WRITE_ONLY vs. CL_MEM_READ_ONLY
	conv1_w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV1_WEIGHT_SIZE, NULL, &status);
	checkError(status, "Error: could not create conv1 weight buffer");

	conv1_b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV1_BIAS_COUNT, NULL, &status);
	checkError(status, "Error: could not create conv1 bias buffer");

	conv2_w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV2_WEIGHT_SIZE, NULL, &status);
	checkError(status, "Error: could not create conv2 weight buffer");	

	conv2_b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * CONV2_BIAS_COUNT, NULL, &status);
	checkError(status, "Error: could not create conv2 bias buffer");

	dense1_w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE1_WEIGHT_SIZE, NULL, &status);
	checkError(status, "Error: could not create dense1 weight buffer");

	dense1_b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE1_BIAS_COUNT, NULL, &status);
	checkError(status, "Error: could not create dense1 bias buffer");

	dense2_w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE2_WEIGHT_SIZE, NULL, &status);
	checkError(status, "Error: could not create dense2 weight buffer");

	dense2_b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE2_BIAS_COUNT, NULL, &status);
	checkError(status, "Error: could not create dense2 bias buffer");
	

	// Copy data to kernel input buffer.
	status = clEnqueueWriteBuffer(queue, input_images_buffer, CL_TRUE, 0, sizeof(unsigned char) * FEATURE_COUNT * n_items, input_images, 0, NULL, NULL);
	checkError(status, "Error: could not copy data into device");

	// TODO: Copy weights for your layers as well
	status = clEnqueueWriteBuffer(queue, conv1_w_buffer, CL_TRUE, 0, sizeof(float) * CONV1_WEIGHT_SIZE, conv1_w, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv1_w into device");

	status = clEnqueueWriteBuffer(queue, conv1_b_buffer, CL_TRUE, 0, sizeof(float) * CONV1_BIAS_COUNT, conv1_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv1_b into device");

	status = clEnqueueWriteBuffer(queue, conv2_w_buffer, CL_TRUE, 0, sizeof(float) * CONV2_WEIGHT_SIZE, conv2_w, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv2_w into device");

	status = clEnqueueWriteBuffer(queue, conv2_b_buffer, CL_TRUE, 0, sizeof(float) * CONV2_BIAS_COUNT, conv2_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv2_b into device");

	status = clEnqueueWriteBuffer(queue, dense1_w_buffer, CL_TRUE, 0, sizeof(float) * DENSE1_WEIGHT_SIZE, dense1_w, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense1_w into device");

	status = clEnqueueWriteBuffer(queue, dense1_b_buffer, CL_TRUE, 0, sizeof(float) * DENSE1_BIAS_COUNT, dense1_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense1_b into device");

	status = clEnqueueWriteBuffer(queue, dense2_w_buffer, CL_TRUE, 0, sizeof(float) * DENSE2_WEIGHT_SIZE, dense2_w, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense2_w into device");

	status = clEnqueueWriteBuffer(queue, dense2_b_buffer, CL_TRUE, 0, sizeof(float) * DENSE2_BIAS_COUNT, dense2_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense2_b into device");


	// Set the arguments for data_in, data_out and sobel kernels.
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_images_buffer);
	checkError(status, "Error: could not set argument 0");

	// TODO: Set arguments for your weights
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&conv1_w_buffer);
	checkError(status, "Error: could not set argument 1");
	
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&conv1_b_buffer);
	checkError(status, "Error: could not set argument 2");

	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&conv2_w_buffer);
	checkError(status, "Error: could not set argument 3");

	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&conv2_b_buffer);
	checkError(status, "Error: could not set argument 4");

	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&dense1_w_buffer);
	checkError(status, "Error: could not set argument 5");

	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&dense1_b_buffer);
	checkError(status, "Error: could not set argument 6");

	status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&dense2_w_buffer);
	checkError(status, "Error: could not set argument 7");

	status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&dense2_b_buffer);
	checkError(status, "Error: could not set argument 8");

	status = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&output_guesses_buffer);
	checkError(status, "Error: could not set argument 9");

	
	// Enqueue the kernel. //
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event);
	checkError(status, "Error: failed to launch data_in");
	
	// Wait for command queue to complete pending events.
	printf("Waiting for kernel to finish..?\n");
	status = clFinish(queue);
	printf("Kernel has finished\n");
	checkError(status, "Kernel failed to finish");

	clReleaseEvent(event);
	
	// Read output buffer from kernel.
	status = clEnqueueReadBuffer(queue, output_guesses_buffer, CL_TRUE, 0, sizeof(unsigned char) * n_items, output_guesses, 0, NULL, NULL);
	checkError(status, "Error: could not copy data from device");
}

void initCL() {
	cl_int status;

	// Start everything at NULL to help identify errors.
	kernel = NULL;
	queue = NULL;
	
	// Locate files via. relative paths.
	if(!setCwdToExeDir()) {
		teardown();
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA");
	if (platform == NULL) {
		teardown();
	}

	// Get the first device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError (status, "Error: could not query devices");

	char info[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
	deviceInfo = info;

	// Create the context.
	context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Error: could not create OpenCL context");

	// Create the command queues for the kernels.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
	std::cout << "Using AOCX: " << binary_file << "\n";
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 1, &device, "", NULL, NULL);
	checkError(status, "Error: could not build program");
	
	// Create the kernel - name passed in here must match kernel name in the original CL file.
	kernel = clCreateKernel(program, "linear_classifier", &status);
	checkError(status, "Failed to create kernel");
}

void cleanup() {
	// Called from aocl_utils::check_error, so there's an error.
	teardown(-1);
}

// seems to need to free conv/dense weight/bias here?  
void teardown(int exit_status) {
	if(kernel) clReleaseKernel(kernel);
	if(queue) clReleaseCommandQueue(queue);
	if(input_images) alignedFree(input_images);
	if(input_weights) alignedFree(input_weights);
	if(reference_guesses) alignedFree(reference_guesses);
	if(output_guesses) alignedFree(output_guesses);
	if(input_images_buffer) clReleaseMemObject(input_images_buffer);
	if(output_guesses_buffer) clReleaseMemObject(output_guesses_buffer);
	if(program) clReleaseProgram(program);
	if(context) clReleaseContext(context);

    // if(conv1_w) alignedFree(conv1_w);
    // if(conv1_b) alignedFree(conv1_b);
    // if(conv2_w) alignedFree(conv2_w);
    // if(conv2_b) alignedFree(conv2_b);
    // if(dense1_w) alignedFree(dense1_w);
    // if(dense1_b) alignedFree(dense1_b);
    // if(dense2_w) alignedFree(dense2_w);
    // if(dense2_b) alignedFree(dense2_b);
	
	exit(exit_status);
}