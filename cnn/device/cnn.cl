// TODO: Define any constants you'll need
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

// // IMG input
// #define PAD_SIZE 2
// #define IMG_DIM 28
// #define IMG_SIZE (IMG_DIM * IMG_DIM)
// #define IMG_PADDED_DIM 32
// #define IMG_PADDED_SIZE (IMG_PADDED_DIM * IMG_PADDED_DIM)

// // CONV1 
// #define CONV1_FILTER_DIM 5
// #define CONV1_NUM_FILTERS 32

// // MAXPOOL1
// #define WINDOW 2
// #define STRIDE 2
// #define MAXPOOL1_DIM 28
// #define MAXPOOL1_CHANNELS 32
// #define MAXPOOL1_SIZE (MAXPOOL1_DIM * MAXPOOL1_DIM * MAXPOOL1_CHANNELS)
// #define MAXPOOL1_OUT_SIZE (MAXPOOL1_SIZE / 4)

// // CONV2
// #define CONV2_IN_PADDED_DIM 18
// #define CONV2_IN_CHANNELS 32
// #define CONV2_IN_SIZE (CONV2_IN_PADDED_DIM * CONV2_IN_PADDED_DIM * CONV2_IN_CHANNELS)

// #define CONV2_FILTER_DIM 5
// #define CONV2_NUM_FILTERS 64

// // MAXPOOL2
// #define MAXPOOL2_DIM 14
// #define MAXPOOL2_CHANNELS 64
// #define MAXPOOL2_SIZE (MAXPOOL2_DIM * MAXPOOL2_DIM * MAXPOOL2_CHANNELS)

// // DENSE1 & DENSE2 & FINAL SOFTMAX
// #define DENSE1_IN_DIM 7
// #define DENSE1_IN_CHANNELS 64
// #define DENSE1_SIZE (DENSE1_IN_DIM * DENSE1_IN_DIM * DENSE1_IN_CHANNELS)
// #define DENSE2_IN_DIM 1
// #define DENSE2_IN_CHANNELS 256
// #define SOFTMAX_NODE_DIM 10

//test:
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_CHANNELS 1
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS)

#define IMAGE_INPUT (28 * 28 * 1)

#define CONV1_INPUT_WIDTH 32
#define CONV1_INPUT_HEIGHT 32
#define CONV1_INPUT_CHANNELS 1
#define CONV1_INPUT (32 * 32 * 1)

#define CONV1_FILTER_WIDTH 5
#define CONV1_FILTER_HEIGHT 5
#define CONV1_FILTER_CHANNELS 32

#define MAXPOOL1_INPUT_WIDTH 28
#define MAXPOOL1_INPUT_HEIGHT 28
#define MAXPOOL1_INPUT_CHANNELS 32
#define MAXPOOL1_INPUT (28 * 28 * 32)

#define CONV2_INPUT_WIDTH 18
#define CONV2_INPUT_HEIGHT 18
#define CONV2_INPUT_CHANNELS 32
#define CONV2_INPUT (18 * 18 * 32)

#define CONV2_FILTER_WIDTH 5
#define CONV2_FILTER_HEIGHT 5
#define CONV2_FILTER_CHANNELS 64

#define MAXPOOL2_INPUT_WIDTH 14
#define MAXPOOL2_INPUT_HEIGHT 14
#define MAXPOOL2_INPUT_CHANNELS 64
#define MAXPOOL2_INPUT (14 * 14 * 64)

#define DENSE1_INPUT_WIDTH 7
#define DENSE1_INPUT_HEIGHT 7
#define DENSE1_INPUT_CHANNELS 64
#define DENSE1_INPUT (7 * 7 * 64)

#define DENSE2_INPUT_WIDTH 1
#define DENSE2_INPUT_HEIGHT 1
#define DENSE2_INPUT_CHANNELS 256
#define DENSE2_INPUT (256 * 1 * 1)

#define SOFTMAX_INPUT 10

//



// TODO: If you decide you'd like to write helper functions, you can define them here

// float ReLU(float input)
// {
// 	if (input < 0.)
// 		return 0.;
// 	else
// 		return input;
// }


// float ReLU(float x) {
//   return (x < 0.) ? 0. : x;
// }

// void PaddingLayer(local float * restrict inputs, local float * restrict outputs, const int in_dim,
//                   const int in_channels, const int pad_dim)
// {
//     const int out_dim = in_dim + pad_dim * 2;
// 	const int out_channels = in_channels;
// 	for (int row = 0; row < out_dim; row++)
// 	{
// 		for (int col = 0; col < out_dim; col++)
// 		{
// 			for (int ch = 0; ch < out_channels; ch++)
// 			{
// 				const int out_index = row * out_channels * out_dim + col * out_channels + ch;
// 				if (row >= pad_dim && col >= pad_dim && (row < in_dim + pad_dim) && (col < in_dim + pad_dim))
// 				{
//                     const int ir = row - pad_dim;
//                     const int ic = col - pad_dim;
// 					const int in_index = ir * in_dim * in_channels
//                                        + ic * in_channels + ch; //2 bugs fixed here
// 					outputs[out_index] = inputs[in_index];
// 				}
// 				else
//                 {
//                     outputs[out_index] = 0;
//                 }
// 			}
// 		}
// 	}
// } 


// // image padding needs to be handled differently due to float vs. char data representations
// void PaddingImage(global const unsigned char * inputs, local float * restrict outputs,
// 				  const int in_dim, const int in_channels, const int pad_dim)
// {
//     const int out_dim = in_dim + pad_dim * 2;
// 	const int out_channels = in_channels;
// 	for (int row = 0; row < out_dim; row++)
// 	{
// 		for (int col = 0; col < out_dim; col++)
// 		{
// 			for (int ch = 0; ch < out_channels; ch++)
// 			{
// 				const int out_index = row * out_channels * out_dim + col * out_channels + ch;
// 				if (row >= pad_dim && col >= pad_dim && (row < in_dim + pad_dim) && (col < in_dim + pad_dim))
// 				{
//                     const int ir = row - pad_dim;
//                     const int ic = col - pad_dim;
// 					const int in_index = ir * in_dim * in_channels
//                                        + ic * in_channels + ch; //2 bugs fixed here
// 					outputs[out_index] = inputs[in_index];
// 				}
// 				else
//                 {
//                     outputs[out_index] = 0;
//                 }
// 			}
// 		}
// 	}
// } 

// void ConvLayer(constant float * restrict weights, constant float * restrict bias,
// 				local const float * restrict inputs, local float * restrict outputs,
// 				const int in_dim, const int in_channels, const int filter_dim, const int num_filters)
// {
// 	const int out_dim = in_dim - filter_dim + 1;
// 	const int out_channels = num_filters;
// 	// float dotprod = 0.0;
// 	// float receptive_inputs = 0;
// 	// float filter_weights = 0;
// 	for (int k = 0; k < num_filters; k++)
// 	{
// 		for (int row = 0; row < out_dim; row++)
// 		{
// 			for (int col = 0; col < out_dim; col++)
// 			{
// 				float dotprod = 0;
// 				for (int ch = 0; ch < in_channels; ch++)
// 				{
// 					for (int i_filter = 0; i_filter < filter_dim; i_filter++)
// 					{
// 						for (int j_filter = 0; j_filter < filter_dim; j_filter++)
// 						{
// 							// const int idx_receptive_inputs = (col + i_filter) * in_dim * in_channels 
//                             //                                + (row + j_filter) * in_channels + ch;
// 							// const int idx_filter_weights = i_filter * filter_dim * in_channels * num_filters
// 							// 					         + j_filter * in_channels * num_filters 
// 							// 					         + ch * num_filters + k;
// 							// dotprod += inputs[idx_receptive_inputs] * weights[idx_filter_weights];
// 							float receptive_inputs = inputs[(col + i_filter) * in_dim * in_channels + (row + j_filter) * in_channels + ch];
// 							float filter_weights = weights[i_filter * filter_dim * in_channels * num_filters
// 												         + j_filter * in_channels * num_filters 
// 												         + ch * num_filters + k];
// 							dotprod += receptive_inputs * filter_weights; // bug catched: should be "*"!!
// 						}
// 					}
// 				}
// 				outputs[k + row * out_channels + col * out_channels * out_dim] = ReLU(dotprod + bias[k]);
// 			}
// 		}
// 	}
// }


// void MaxPool(local const float * restrict inputs, local float * restrict outputs,
// 			 const int in_dim, const int in_channels, const int pool_dim, const int pool_stride)
// {
// 	const int out_dim = in_dim / pool_stride; 
// 	const int out_channels = in_channels;
// 	// float current_max = -INFINITY; // some random value....should be enough?
// 	// float pool_window = 0;

// 	for (int row = 0; row < in_dim; row += pool_stride)
// 	{
// 		for (int col = 0; col < in_dim; col += pool_stride)
// 		{
// 			for (int ch = 0; ch < in_channels; ch++)
// 			{
// 				float current_max = -INFINITY;
// 				for (int i = 0; i < pool_dim; i++)
// 				{
// 					for (int j = 0; j < pool_dim; j++)
// 					{
// 						const int idx = (row + i) * in_channels * in_dim + (col + j) * in_channels + ch;
// 						float pool_window = inputs[idx];
// 						if (pool_window > current_max)
// 						{
// 							current_max = pool_window;
// 						}
// 					}
// 				}
// 				const int out_index = (row / pool_stride) * out_channels * out_dim
// 							  + (col / pool_stride) * out_channels * ch;
// 				outputs[out_index] = current_max;
// 			}
// 		}
// 	}
// }


// void DenseLayer(constant float * restrict weights, constant float * restrict bias,
// 				local const float * restrict inputs, local float * restrict outputs,
// 				const int in_dim, const int in_channels, const int out_dim, bool isFinalLayer)
// {
// 	// float dotprod = 0.0;
// 	// float neuron_output = 0.0;
// 	for (int l = 0; l < out_dim; l++)
// 	{
// 		float dotprod = 0;
// 		for (int row = 0; row < in_dim; row++)
// 		{
// 			for (int col = 0; col < in_dim; col++)
// 			{
// 				for (int ch = 0; ch < in_channels; ch++)
// 				{
// 					// const int in_index = row * in_dim + in_channels + col * in_channels + ch;
// 					// const int weight_index = row * in_dim * in_channels * out_dim + col * in_channels
// 					// 				 * out_dim + ch * out_dim + l;
// 					// dotprod += inputs[in_index] * weights[weight_index];
// 					float neuron = inputs[row * in_dim * in_channels + col * in_channels + ch]; // previous bug: * instead of + before in_channels!!!
// 					float fc_weights = weights[row * in_dim * in_channels * out_dim + col * in_channels
// 									 * out_dim + ch * out_dim + l];
// 					dotprod += neuron * fc_weights;  // bug catched here: should be "*"!!!
// 				}
// 			}
// 		}
// 		float neuron_output = dotprod + bias[l];
// 		if (isFinalLayer == false)  // previous bug: missing "!" here! Only FinalLayer uses Softmax!
// 		{
// 			outputs[l] = ReLU(neuron_output);
// 		}
// 		else
// 		{
// 			outputs[l] = neuron_output;
// 		}
// 	}
// }



//TEST:

float relu(float x) {
  return (x < 0.) ? 0. : x;
}

void maxpool_layer(
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pool_width,
    const int pool_height,
    const int stride_height,
    const int stride_width
) {
  const int output_width = input_width / stride_width;
  const int output_height = input_height / stride_height;
  const int output_channels = input_channels;

  for (int w = 0; w < input_width; w += stride_width) {
    for (int h = 0; h < input_height; h += stride_height) {
      for (int c = 0; c < input_channels; c++) {
        float maxima = -INFINITY;
        for (int ww = 0; ww < pool_width; ww++) {
          for (int hh = 0; hh < pool_height; hh++) {
            const int idx = (w + ww) * input_channels * input_height +
              (h + hh) * input_channels + c;
            float pixel = inputs[idx];
            if(maxima < pixel) {
              maxima = pixel;
            }
          }
        }

        const int o_idx = (w / stride_width) * output_channels * output_height + 
          (h / stride_height) * output_channels + c;
        outputs[o_idx] = maxima;
      }
    }  
  }
}

void conv_layer(
    constant float * restrict weights,
    constant float * restrict bias,
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int filter_width,
    const int filter_height,
    const int filter_channels
) {
  const int OW = input_width - filter_width + 1;
  const int OH = input_height - filter_height + 1;
  const int OC = filter_channels;

  for (int f = 0; f < filter_channels; f++) {
    for (int h = 0; h < OH; h++) {
      for (int w = 0; w < OW; w++) {
        /* Compute the output of a single filter */
        float sum = 0;
        for (int c = 0; c < input_channels; c++) {
          for (int ww = 0; ww < filter_width; ww++) {
            for (int hh = 0; hh < filter_height; hh++) {
              float pix = inputs[
                (w + ww) * input_height * input_channels + 
                (h + hh) * input_channels + c];
              float wgt = weights[
                ww * filter_height * input_channels * filter_channels +
                hh * input_channels * filter_channels +
                c * filter_channels + f];
              sum += pix * wgt;
            }
          }
        }
        outputs[w * OC * OH + h * OC + f] = relu(sum + bias[f]);
      }
    }  
  }
}

void dense_layer(
    constant float * restrict weights,
    constant float * restrict bias,
    local const float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int output_size,
    bool use_relu
) {
  const int input_size = input_width * input_height;

  for (int x = 0; x < output_size; x++) {
    float sum = 0;
    for (int w = 0; w < input_width; w++) {
      for (int h = 0; h < input_height; h++) {
        for (int c = 0; c < input_channels; c++) {
          float pix = inputs[
            w * input_height * input_channels +
            h * input_channels + c];
          float wgt = weights[
            w * input_height * input_channels * output_size + 
            h * input_channels * output_size + 
            c * output_size + x];
          sum += pix * wgt;
        } 
      }
    }
    float activation = sum + bias[x];
    if (use_relu) {
      outputs[x] = relu(activation);
    } else {
      outputs[x] = activation;
    }
  }
}

void input_pad_layer(
    global const unsigned char * inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pad_width,
    const int pad_height
) {
  const int OW = input_width + pad_width * 2;
  const int OH = input_height + pad_height * 2;
  const int OC = input_channels;

  for (int w = 0; w < OW; w++) {
    for (int h = 0; h < OH; h++) {
      for (int c = 0; c < input_channels; c++) {
        const int o_idx = w * OC * OH + h * OC + c;
        if (w >= pad_width && h >= pad_height
            && w < input_width + pad_width 
            && h < input_height + pad_height) {
          const int ih = h - pad_height;
          const int iw = w - pad_width;
          const int i_idx = iw * input_height * input_channels + ih
            * input_channels + c;
          outputs[o_idx] =  inputs[i_idx];
        } else {
          outputs[o_idx] = 0;
        }
      }
    }
  }
}

void pad_layer(
    local float * restrict inputs,
    local float * restrict outputs,
    const int input_width,
    const int input_height,
    const int input_channels,
    const int pad_width,
    const int pad_height
) {
  const int OW = input_width + pad_width * 2;
  const int OH = input_height + pad_height * 2;
  const int OC = input_channels;

  for (int w = 0; w < OW; w++) {
    for (int h = 0; h < OH; h++) {
      for (int c = 0; c < input_channels; c++) {
        const int o_idx = w * OC * OH + h * OC + c;
        if (w >= pad_width && h >= pad_height
            && w < input_width + pad_width 
            && h < input_height + pad_height) {
          const int ih = h - pad_height;
          const int iw = w - pad_width;
          const int i_idx = iw * input_height * input_channels + ih
            * input_channels + c;
          outputs[o_idx] =  inputs[i_idx];
        } else {
          outputs[o_idx] = 0;
        }
      }
    }
  }
}

void print_buffer(const char* fmt, local const float * restrict buffer,
    const int width, const int height, const int channels) {
  for (int w = 0; w < width; w++) {
    for (int h = 0; h < height; h++) {
      for (int c = 0; c < 1; c++) {
        printf(fmt, buffer[w * height * channels + h * channels + c]);
      }
    }
    printf("\n");
  }
}

// TODO: Build a CNN!
__attribute__((reqd_work_group_size(100,1,1))) // change this to change workgroup size
__kernel void linear_classifier(global const unsigned char * restrict images, 
								constant float * restrict conv1_weights,
								constant float * restrict conv1_bias,
								constant float * restrict conv2_weights,
								constant float * restrict conv2_bias,
								constant float * restrict dense1_weights,
								constant float * restrict dense1_bias,
								constant float * restrict dense2_weights,
								constant float * restrict dense2_bias,
								global unsigned char * restrict guesses)
{
// 	global const unsigned char * image = &images[get_global_id(0) * IMG_SIZE];

//     // setting local variables for the inter-layer data forward pass
// 	// local float padded_img[IMG_PADDED_SIZE];
// 	// local float conv1_out[MAXPOOL1_SIZE];
// 	// local float maxpool1_out[MAXPOOL1_OUT_SIZE];
// 	// local float conv2_in[CONV2_IN_SIZE];
// 	// local float conv2_out[MAXPOOL2_SIZE];
// 	// local float dense1_in[DENSE1_SIZE];
// 	// local float dense2_in[DENSE2_IN_CHANNELS];
// 	// local float softmax_node[SOFTMAX_NODE_DIM];
// 	// // float neuron_max = -99920120210;
//     // float neuron_max = -INFINITY;
// 	// int predict = -1;  // for debugging purpose


// 	// /* CONV LAYER 1 */
//     // local float padded_img[IMG_PADDED_SIZE];
// 	// PaddingImage(image, padded_img, IMG_DIM, 1, 2);

//     // local float conv1_out[MAXPOOL1_SIZE];
// 	// ConvLayer(conv1_weights, conv1_bias, padded_img, conv1_out, IMG_PADDED_DIM, 1, CONV1_FILTER_DIM,
// 	// 		  CONV1_NUM_FILTERS);

// 	// /* MAXPOOL LAYER 1 PLUS PADDING */
//     // local float maxpool1_out[MAXPOOL1_OUT_SIZE];
// 	// MaxPool(conv1_out, maxpool1_out, MAXPOOL1_DIM, MAXPOOL1_CHANNELS, WINDOW, STRIDE);
//     // local float conv2_in[CONV2_IN_SIZE];
// 	// PaddingLayer(maxpool1_out, conv2_in, MAXPOOL1_DIM/2, MAXPOOL1_CHANNELS, PAD_SIZE);

// 	// /* CONV LAYER 2 */
//     // local float conv2_out[MAXPOOL2_SIZE];
// 	// ConvLayer(conv2_weights, conv2_bias, conv2_in, conv2_out, CONV2_IN_PADDED_DIM, CONV2_IN_CHANNELS,
// 	// 		  CONV2_FILTER_DIM, CONV2_NUM_FILTERS);

// 	// /* MAXPOOL LAYER 2 */
//     // local float dense1_in[DENSE1_SIZE];
// 	// MaxPool(conv2_out, dense1_in, MAXPOOL2_DIM, MAXPOOL2_CHANNELS, WINDOW, STRIDE);
	
// 	// /* DENSE LAYER */
//     // local float dense2_in[DENSE2_IN_CHANNELS];
// 	// DenseLayer(dense1_weights, dense1_bias, dense1_in, dense2_in, DENSE1_IN_DIM, DENSE1_IN_CHANNELS,
// 	// 		   DENSE2_IN_CHANNELS, false);

// 	// /* DENSE 2 */		
//     // local float softmax_node[SOFTMAX_NODE_DIM];
// 	// DenseLayer(dense2_weights, dense2_bias, dense2_in, softmax_node, DENSE2_IN_DIM, DENSE2_IN_CHANNELS,
// 	// 		   SOFTMAX_NODE_DIM, true);

// 	// /* CONV LAYER 1 */
//     // local float padded_img[1024];
// 	// PaddingImage(image, padded_img, 28, 1, 2);

//     // local float conv1_out[25088];
// 	// ConvLayer(conv1_weights, conv1_bias, padded_img, conv1_out, 32, 1, 5, 32);

// 	// /* MAXPOOL LAYER 1 PLUS PADDING */
//     // local float maxpool1_out[14 * 14 * 32];
// 	// MaxPool(conv1_out, maxpool1_out, 28, 32, 2, 2);
//     // local float conv2_in[18 * 18 * 32];
// 	// PaddingLayer(maxpool1_out, conv2_in, 14, 32, 2);

// 	// /* CONV LAYER 2 */
//     // local float conv2_out[14 * 14 * 64];
// 	// ConvLayer(conv2_weights, conv2_bias, conv2_in, conv2_out, 18, 32,
// 	// 		  5, 64);

// 	// /* MAXPOOL LAYER 2 */
//     // local float dense1_in[7 * 7 * 64];
// 	// MaxPool(conv2_out, dense1_in, 14, 64, 2, 2);
	
// 	// /* DENSE LAYER */
//     // local float dense2_in[256];
// 	// DenseLayer(dense1_weights, dense1_bias, dense1_in, dense2_in, 7, 64,
// 	// 		   256, false);

// 	// /* DENSE 2 */		
//     // local float softmax_node[10];
// 	// DenseLayer(dense2_weights, dense2_bias, dense2_in, softmax_node, 1, 256,
// 	// 		   10, true);

//     //test:

//     local float conv1_input[CONV1_INPUT];
//     PaddingImage(image, conv1_input, IMG_WIDTH, IMG_CHANNELS, 2);

//     /* CONV LAYER 1 */
//     local float maxpool1_input[MAXPOOL1_INPUT];
//     ConvLayer(conv1_weights, conv1_bias, conv1_input, maxpool1_input, 
//     CONV1_INPUT_WIDTH, CONV1_INPUT_CHANNELS,
//     CONV1_FILTER_WIDTH, CONV1_FILTER_CHANNELS);

//   /* MAXPOOL LAYER */
//     local float maxpool1_output[14 * 14 * 32];
//     MaxPool(
//     maxpool1_input, maxpool1_output,
//     MAXPOOL1_INPUT_WIDTH, MAXPOOL1_INPUT_CHANNELS, 2, 2);

//   /* pad */
//     local float conv2_input[CONV2_INPUT];
//     PaddingLayer(maxpool1_output, conv2_input, 14, 32, 2);
    
//     print_buffer("%09.6f, ", conv2_input, 18, 18, 32);
  
//   /* CONV LAYER 2 */
//     local float maxpool2_input[MAXPOOL2_INPUT];
//     ConvLayer(conv2_weights, conv2_bias, conv2_input, maxpool2_input, 18, 32, 5, 64);

//     print_buffer("%06.2f, ", maxpool2_input, 14, 14, 64);
  
//   /* MAXPOOL LAYER 2 */
//     local float dense1_input[DENSE1_INPUT];
//     MaxPool(maxpool2_input, dense1_input,
//       MAXPOOL2_INPUT_WIDTH, MAXPOOL2_INPUT_CHANNELS, 2, 2);
  
//     print_buffer("%06.2f, ", maxpool2_input, 7, 7, 64);

//   /* DENSE LAYER 1 */
//     local float dense2_input[DENSE2_INPUT];
//     DenseLayer(
//       dense1_weights, dense1_bias, dense1_input, dense2_input,
//       DENSE1_INPUT_WIDTH, DENSE1_INPUT_CHANNELS,
//       DENSE2_INPUT, false);
  
//   /* DENSE LAYER 2 */
//     local float softmax_input[SOFTMAX_INPUT];
//     DenseLayer(
//       dense2_weights, dense2_bias, dense2_input, softmax_input,
//       DENSE2_INPUT_WIDTH, DENSE2_INPUT_CHANNELS,
//       SOFTMAX_INPUT, true);

    

// 	/* FINAL GUESS */
//     // float neuron_max = -INFINITY;
// 	// int guess = -1;  // for debugging purpose
// 	// for (int i = 0; i < 10; i++)
// 	// {
// 	// 	float current_neuron = softmax_node[i];
// 	// 	if (neuron_max < current_neuron)
// 	// 	{
// 	// 		neuron_max = current_neuron;
// 	// 		guess = i;
// 	// 	}
// 	// }
// 	// guesses[get_global_id(0)] = guess;

//       /* FINAL GUESS */
//   float maximum = -INFINITY;
//   int guess = -1;
//   for (int i = 0; i < SOFTMAX_INPUT; i++) {
//     float pix = softmax_input[i];
//     if (maximum < pix) {
//       maximum = pix;
//       guess = i;
//     }
//   }

//   guesses[get_global_id(0)] = guess;

  global const unsigned char * image = &images[get_global_id(0) * IMG_SIZE];

  /* input image pad */
  local float conv1_input[CONV1_INPUT];
  input_pad_layer(image, conv1_input,
      IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
      2, 2);

  /* CONV LAYER 1 */
  local float maxpool1_input[MAXPOOL1_INPUT];
  conv_layer(
      conv1_weights, conv1_bias, conv1_input, maxpool1_input, 
      CONV1_INPUT_WIDTH, CONV1_INPUT_HEIGHT, CONV1_INPUT_CHANNELS,
      CONV1_FILTER_WIDTH, CONV1_FILTER_HEIGHT, CONV1_FILTER_CHANNELS);

  /* MAXPOOL LAYER */
  local float maxpool1_output[14 * 14 * 32];
  maxpool_layer(
      maxpool1_input, maxpool1_output,
      MAXPOOL1_INPUT_WIDTH, MAXPOOL1_INPUT_HEIGHT, MAXPOOL1_INPUT_CHANNELS,
      2, 2, 2, 2);

  /* pad */
  local float conv2_input[CONV2_INPUT];
  pad_layer(maxpool1_output, conv2_input,
      14, 14, 32, 2, 2);

//   print_buffer("%09.6f, ", conv2_input, 18, 18, 32);
  
  /* CONV LAYER 2 */
  local float maxpool2_input[MAXPOOL2_INPUT];
  conv_layer(
      conv2_weights, conv2_bias, conv2_input, maxpool2_input, 
      18, 18, 32,
      5, 5, 64);

  print_buffer("%06.2f, ", maxpool2_input, 14, 14, 64);
  
  /* MAXPOOL LAYER 2 */
  local float dense1_input[DENSE1_INPUT];
  maxpool_layer(
      maxpool2_input, dense1_input,
      MAXPOOL2_INPUT_WIDTH, MAXPOOL2_INPUT_HEIGHT, MAXPOOL2_INPUT_CHANNELS,
      2, 2, 2, 2);
  
  //print_buffer("%06.2f, ", maxpool2_input, 7, 7, 64);

  /* DENSE LAYER 1 */
  local float dense2_input[DENSE2_INPUT];
  dense_layer(
      dense1_weights, dense1_bias, dense1_input, dense2_input,
      DENSE1_INPUT_WIDTH, DENSE1_INPUT_HEIGHT, DENSE1_INPUT_CHANNELS,
      DENSE2_INPUT, true);
  
  /* DENSE LAYER 2 */
  local float softmax_input[SOFTMAX_INPUT];
  dense_layer(
      dense2_weights, dense2_bias, dense2_input, softmax_input,
      DENSE2_INPUT_WIDTH, DENSE2_INPUT_HEIGHT, DENSE2_INPUT_CHANNELS,
      SOFTMAX_INPUT, false);

  /* FINAL GUESS */
  float maximum = -INFINITY;
  int guess = -1;
  for (int i = 0; i < SOFTMAX_INPUT; i++) {
    float pix = softmax_input[i];
    if (maximum < pix) {
      maximum = pix;
      guess = i;
    }
  }

  guesses[get_global_id(0)] = guess;
}
