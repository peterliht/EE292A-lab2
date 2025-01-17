// TODO: Define any constants you'll need
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

// IMG input
#define PAD_SIZE 2
#define IMG_DIM 28
#define IMG_SIZE (IMG_DIM * IMG_DIM)
#define IMG_PADDED_DIM 32
#define IMG_PADDED_SIZE (IMG_PADDED_DIM * IMG_PADDED_DIM)

// CONV1 
#define CONV1_FILTER_DIM 5
#define CONV1_NUM_FILTERS 32

// MAXPOOL1
#define WINDOW 2
#define STRIDE 2
#define MAXPOOL1_DIM 28
#define MAXPOOL1_CHANNELS 32
#define MAXPOOL1_SIZE (MAXPOOL1_DIM * MAXPOOL1_DIM * MAXPOOL1_CHANNELS)
#define MAXPOOL1_OUT_SIZE (MAXPOOL1_SIZE / 4)

// CONV2
#define CONV2_IN_PADDED_DIM 18
#define CONV2_IN_CHANNELS 32
#define CONV2_IN_SIZE (CONV2_IN_PADDED_DIM * CONV2_IN_PADDED_DIM * CONV2_IN_CHANNELS)

#define CONV2_FILTER_DIM 5
#define CONV2_NUM_FILTERS 64

// MAXPOOL2
#define MAXPOOL2_DIM 14
#define MAXPOOL2_CHANNELS 64
#define MAXPOOL2_SIZE (MAXPOOL2_DIM * MAXPOOL2_DIM * MAXPOOL2_CHANNELS)

// DENSE1 & DENSE2 & FINAL SOFTMAX
#define DENSE1_IN_DIM 7
#define DENSE1_IN_CHANNELS 64
#define DENSE1_SIZE (DENSE1_IN_DIM * DENSE1_IN_DIM * DENSE1_IN_CHANNELS)
#define DENSE2_IN_DIM 1
#define DENSE2_IN_CHANNELS 256
#define SOFTMAX_NODE_DIM 10


// TODO: If you decide you'd like to write helper functions, you can define them here

float ReLU(float input)
{
	if (input < 0.)
		return 0.;
	else
		return input;
}


void PaddingLayer(local float * restrict inputs, local float * restrict outputs, const int in_dim,
                  const int in_channels, const int pad_dim)
{
    const int out_dim = in_dim + pad_dim * 2;
	const int out_channels = in_channels;
    #pragma unroll
	for (int row = 0; row < out_dim; row++)
	{
        #pragma unroll
		for (int col = 0; col < out_dim; col++)
		{
            #pragma unroll
			for (int ch = 0; ch < out_channels; ch++)
			{
				const int out_index = row * out_channels * out_dim + col * out_channels + ch;
				if (row >= pad_dim && col >= pad_dim && (row < in_dim + pad_dim) && (col < in_dim + pad_dim))
				{
                    const int ir = row - pad_dim;
                    const int ic = col - pad_dim;
					const int in_index = ir * in_dim * in_channels
                                       + ic * in_channels + ch; //2 bugs fixed here
					outputs[out_index] = inputs[in_index];
				}
				else
                {
                    outputs[out_index] = 0;
                }
			}
		}
	}
} 


// image padding needs to be handled differently due to float vs. char data representations
void PaddingImage(global const unsigned char * inputs, local float * restrict outputs,
				  const int in_dim, const int in_channels, const int pad_dim)
{
    const int out_dim = in_dim + pad_dim * 2;
	const int out_channels = in_channels;
    #pragma unroll
	for (int row = 0; row < out_dim; row++)
	{
        #pragma unroll
		for (int col = 0; col < out_dim; col++)
		{
            #pragma unroll
			for (int ch = 0; ch < out_channels; ch++)
			{
				const int out_index = row * out_channels * out_dim + col * out_channels + ch;
				if (row >= pad_dim && col >= pad_dim && (row < in_dim + pad_dim) && (col < in_dim + pad_dim))
				{
                    const int ir = row - pad_dim;
                    const int ic = col - pad_dim;
					const int in_index = ir * in_dim * in_channels
                                       + ic * in_channels + ch; //2 bugs fixed here
					outputs[out_index] = inputs[in_index];
				}
				else
                {
                    outputs[out_index] = 0;
                }
			}
		}
	}
} 

void ConvLayer(constant float * restrict weights, constant float * restrict bias,
				local const float * restrict inputs, local float * restrict outputs,
				const int in_dim, const int in_channels, const int filter_dim, const int num_filters)
{
	const int out_dim = in_dim - filter_dim + 1;
	const int out_channels = num_filters;
	float dotprod = 0;
	float receptive_inputs = 0;
	float filter_weights = 0;


    #pragma unroll

	for (int k = 0; k < num_filters; k++)
	{
        #pragma unroll
		for (int row = 0; row < out_dim; row++)
		{
            #pragma unroll
			for (int col = 0; col < out_dim; col++)
			{
				dotprod = 0;
                #pragma unroll
				for (int ch = 0; ch < in_channels; ch++)
				{
                    #pragma unroll
					for (int i_filter = 0; i_filter < filter_dim; i_filter++)
					{
                        #pragma unroll
						for (int j_filter = 0; j_filter < filter_dim; j_filter++)
						{
							// const int idx_receptive_inputs = (col + i_filter) * in_dim * in_channels 
                            //                                + (row + j_filter) * in_channels + ch;
							// const int idx_filter_weights = i_filter * filter_dim * in_channels * num_filters
							// 					         + j_filter * in_channels * num_filters 
							// 					         + ch * num_filters + k;
							// dotprod += inputs[idx_receptive_inputs] * weights[idx_filter_weights];
							receptive_inputs = inputs[(col + i_filter) * in_dim * in_channels + (row + j_filter) * in_channels + ch];
							filter_weights = weights[i_filter * filter_dim * in_channels * num_filters
												         + j_filter * in_channels * num_filters 
												         + ch * num_filters + k];
							dotprod += receptive_inputs * filter_weights; // bug catched: should be "*"!!
						}
					}
				}
				outputs[k + row * out_channels + col * out_channels * out_dim] = ReLU(dotprod + bias[k]);
			}
		}
	}
}


void MaxPool(local const float * restrict inputs, local float * restrict outputs,
			 const int in_dim, const int in_channels, const int pool_dim, const int pool_stride)
{
	const int out_dim = in_dim / pool_stride; 
	const int out_channels = in_channels;
	float current_max = -99920120210999;   //some random number...
	float pool_window = 0;

    #pragma unroll
	
    for (int row = 0; row < in_dim; row += pool_stride)
	{
        #pragma unroll
		for (int col = 0; col < in_dim; col += pool_stride)
		{
            #pragma unroll
			for (int ch = 0; ch < in_channels; ch++)
			{
				current_max = -99920120210999;
                #pragma unroll
				for (int i = 0; i < pool_dim; i++)
				{
                    #pragma unroll
					for (int j = 0; j < pool_dim; j++)
					{
						const int idx = (row + i) * in_channels * in_dim + (col + j) * in_channels + ch;
						pool_window = inputs[idx];
						if (pool_window > current_max)
						{
							current_max = pool_window;
						}
					}
				}
				const int out_index = (row / pool_stride) * out_channels * out_dim
							  + (col / pool_stride) * out_channels + ch; // bug here!! changed to "+ ch" 
				outputs[out_index] = current_max;
			}
		}
	}
}


void DenseLayer(constant float * restrict weights, constant float * restrict bias,
				local const float * restrict inputs, local float * restrict outputs,
				const int in_dim, const int in_channels, const int out_dim, bool isFinalLayer)
{
	// float dotprod = 0.0;
	// float neuron_output = 0.0;
    #pragma unroll

	for (int l = 0; l < out_dim; l++)
	{
		float dotprod = 0;
        #pragma unroll
		for (int row = 0; row < in_dim; row++)
		{
            #pragma unroll
			for (int col = 0; col < in_dim; col++)
			{
                #pragma unroll
				for (int ch = 0; ch < in_channels; ch++)
				{
					// const int in_index = row * in_dim + in_channels + col * in_channels + ch;
					// const int weight_index = row * in_dim * in_channels * out_dim + col * in_channels
					// 				 * out_dim + ch * out_dim + l;
					// dotprod += inputs[in_index] * weights[weight_index];
					float neuron = inputs[row * in_dim * in_channels + col * in_channels + ch]; // previous bug: * instead of + before in_channels!!!
					float fc_weights = weights[row * in_dim * in_channels * out_dim + col * in_channels
									 * out_dim + ch * out_dim + l];
					dotprod += neuron * fc_weights;  // bug catched here: should be "*"!!!
				}
			}
		}
		float neuron_output = dotprod + bias[l];
		if (isFinalLayer == false)  // previous bug: missing "!" here! Only FinalLayer uses Softmax!
		{
			outputs[l] = ReLU(neuron_output);
		}
		else
		{
			outputs[l] = neuron_output;
		}
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
	global const unsigned char * image = &images[get_global_id(0) * IMG_SIZE];

    // setting local variables for the inter-layer data forward pass
	local float padded_img[IMG_PADDED_SIZE];
	local float conv1_out[MAXPOOL1_SIZE];
	local float maxpool1_out[MAXPOOL1_OUT_SIZE];
	local float conv2_in[CONV2_IN_SIZE];
	local float conv2_out[MAXPOOL2_SIZE];
	local float dense1_in[DENSE1_SIZE];
	local float dense2_in[DENSE2_IN_CHANNELS];
	local float softmax_node[SOFTMAX_NODE_DIM];
	// float neuron_max = -99920120210;
    float neuron_max = -INFINITY;
	int predict = -1;  // for debugging purpose


	/* CONV LAYER 1 */
	PaddingImage(image, padded_img, IMG_DIM, 1, 2);

	ConvLayer(conv1_weights, conv1_bias, padded_img, conv1_out, IMG_PADDED_DIM, 1, CONV1_FILTER_DIM,
			  CONV1_NUM_FILTERS);

	/* MAXPOOL LAYER 1 PLUS PADDING */
	MaxPool(conv1_out, maxpool1_out, MAXPOOL1_DIM, MAXPOOL1_CHANNELS, WINDOW, STRIDE);

	PaddingLayer(maxpool1_out, conv2_in, MAXPOOL1_DIM/2, MAXPOOL1_CHANNELS, PAD_SIZE);

	/* CONV LAYER 2 */
	ConvLayer(conv2_weights, conv2_bias, conv2_in, conv2_out, CONV2_IN_PADDED_DIM, CONV2_IN_CHANNELS,
			  CONV2_FILTER_DIM, CONV2_NUM_FILTERS);

	/* MAXPOOL LAYER 2 */
	MaxPool(conv2_out, dense1_in, MAXPOOL2_DIM, MAXPOOL2_CHANNELS, WINDOW, STRIDE);
	
	/* DENSE LAYER */
	DenseLayer(dense1_weights, dense1_bias, dense1_in, dense2_in, DENSE1_IN_DIM, DENSE1_IN_CHANNELS,
			   DENSE2_IN_CHANNELS, false);

	/* DENSE 2 */		
	DenseLayer(dense2_weights, dense2_bias, dense2_in, softmax_node, DENSE2_IN_DIM, DENSE2_IN_CHANNELS,
			   SOFTMAX_NODE_DIM, true);


	/* FINAL GUESS */
	for (int i = 0; i < 10; i++)
	{
		float current_neuron = softmax_node[i];
		if (neuron_max < current_neuron)
		{
			neuron_max = current_neuron;
			predict = i;
		}
	}
	guesses[get_global_id(0)] = predict;
}
