#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <vector>
#include <map>

#define CUDA_FLOAT32_MIN -10000

// const int cur_len = 100;
// const int ngram_size = 3;
// const int num_hypos = 80;

// template <typename scalar_t>
// typedef struct
// {
//     scalar_t prev_ngram_tuple[ngram_size - 1];
//     scalar_t output_ngram;
// } custom_type;

template <typename scalar_t>
__global__ void _get_ngrams_kernel_one(int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> output, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // output[id].output_ngram = id;
    // printf("Thread id %d\n", id);
    int cur_len = prev_input_ids.size(1);
    output[id][0] = id / (cur_len - ngram_size + 1);
    for (int i = 1; i < ngram_size; i++)
    {
        output[id][i] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + i - 1];
    }
    output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
}

template <typename scalar_t>
__global__ void _get_ngrams_kernel_two(int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> generated_ngrams, torch::PackedTensorAccessor32<scalar_t, 2> scores, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // generated_ngrams[id].generated_ngrams_ngram = id;
    // printf("Thread id %d\n", id);
    int cur_len = prev_input_ids.size(1);
    // generated_ngrams[id][0] = id / (cur_len - ngram_size + 1);
    for (int i = 0; i < generated_ngrams.size(0); i++)
    {   
        int count = 0;
        for(int j = 0 ; j < generated_ngrams.size(1) - 1; j++) {
            if(j == 0) {
                if (id == generated_ngrams[i][0])++count;
            } else {
                if (generated_ngrams[i][j] == prev_input_ids[id][cur_len - ngram_size + j])++count;
            }
        }
        if ((count == generated_ngrams.size(1) - 1) and (id < scores.size(0)))
        {
            scores[id][(int)generated_ngrams[i][generated_ngrams.size(1) - 1]] = CUDA_FLOAT32_MIN;
        }
    }
    // output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
}

torch::Tensor _get_ngrams_cuda(  int ngram_size,
                        torch::Tensor prev_input_ids,
                        torch::Tensor scores,
                        int num_hypos) {

    torch::Device device(torch::kCUDA);
    int cur_len = prev_input_ids.size(1);
    dim3 one_grid_size(ceil((num_hypos * (cur_len - ngram_size + 1)) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // TORCH_WARN("Using grid size ");
    // TORCH_WARN(ceil((num_hypos * (cur_len - ngram_size + 1)) / 32.0));
    at::IntArrayRef generated_ngrams_size({(num_hypos * (cur_len - ngram_size + 1)), ngram_size + 1}); 
    torch::Tensor generated_ngrams = torch::zeros(generated_ngrams_size, device);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_one_cuda", ([&] {
        _get_ngrams_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), num_hypos);
    }));


    dim3 two_grid_size(ceil((num_hypos) / 32.0), 1, 1);
    dim3 two_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_two_cuda", ([&] { 
        _get_ngrams_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), scores.packed_accessor32<scalar_t, 2>(), num_hypos); 
    }));

    return generated_ngrams;
}
