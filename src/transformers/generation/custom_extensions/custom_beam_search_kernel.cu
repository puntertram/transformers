#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <vector>
#include <map>

#define CUDA_FLOAT32_MIN -10000
#define CUDA_FLOAT32_MAX 10000

// const int cur_len = 100;
// const int ngram_size = 3;
// const int num_hypos = 80;

// template <typename scalar_t>
// typedef struct
// {
//     scalar_t prev_ngram_tuple[ngram_size - 1];
//     scalar_t output_ngram;
// } custom_type;

using namespace torch::indexing;

template <typename scalar_t>
__global__ void _get_ngrams_kernel_one(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> output, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // output[id].output_ngram = id;
    // printf("Thread id %d\n", id);
    if (id < kernel_size) {
        int cur_len = prev_input_ids.size(1);
        output[id][0] = id / (cur_len - ngram_size + 1);
        for (int i = 1; i < ngram_size; i++)
        {
            output[id][i] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + i - 1];
        }
        output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
    }
}

template <typename scalar_t>
__global__ void _get_ngrams_kernel_two(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> generated_ngrams, torch::PackedTensorAccessor32<scalar_t, 2> scores, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < kernel_size) {
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
    }
    // output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
}

template <typename scalar_t>
__global__ void _beam_search_kernel_one(
    float batch_size,
    float group_size,
    float max_token_length,
    float pad_token_id,
    float eos_token_id,
    float early_stopping,
    float length_penalty,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 2> next_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_indices,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_indices,
    torch::PackedTensorAccessor32<scalar_t, 1> _done,
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hyps,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hyps_sizes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        if (_done[batch_idx] == 1) {
            for(int j = 0 ; j < group_size; j++) {
                next_beam_scores[batch_idx][j] = 0;
                next_beam_tokens[batch_idx][j] = pad_token_id;
                next_beam_indices[batch_idx][j] = 0;
            }
        } else {
            int num_tokens = 2 * group_size;
            int beam_idx = 0;
            for(int beam_token_rank = 0; beam_token_rank < num_tokens; beam_token_rank++) {
                float next_token = next_tokens[batch_idx][beam_token_rank];
                float next_index = next_indices[batch_idx][beam_token_rank];
                float next_score = next_scores[batch_idx][beam_token_rank];
                float batch_beam_idx = batch_idx * group_size + next_index;
                if((next_token == eos_token_id)) {
                    
                    if (beam_token_rank < group_size) {

                        float index;
                        int scenario = -1;
                        if (beam_hyps_sizes[batch_idx][0] >= group_size) {
                            index = beam_hyps_sizes[batch_idx][2];
                            scenario = 1;
                        } else {
                            index = beam_hyps_sizes[batch_idx][0];
                            scenario = 2;
                        }
                        // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hyps_sizes[batch_idx][0], group_size);
                        if(scenario == 2) {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score / pow(input_ids.size(1), length_penalty);
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++) {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            beam_hyps_sizes[batch_idx][0]++;
                            if (next_score < beam_hyps_sizes[batch_idx][1]) {
                                beam_hyps_sizes[batch_idx][1] = next_score;
                                beam_hyps_sizes[batch_idx][2] = index;
                            }
                        } else if (scenario == 1 && (next_score > beam_hyps_sizes[batch_idx][1])) {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score;
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++) {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            float worst_score = CUDA_FLOAT32_MAX;
                            int worst_index = -1;
                            for(int j = 0 ; j < group_size; j++) {
                                if (worst_score > beam_hyps[batch_idx][j][1]) {
                                    worst_score = beam_hyps[batch_idx][j][1];
                                    worst_index = j;
                                }
                            }
                            beam_hyps_sizes[batch_idx][1] = worst_score;
                            beam_hyps_sizes[batch_idx][2] = worst_index;
                        }
                    } 
                } else {
                    // printf("[%d][%d] Adding %f %f %f\n", batch_idx, beam_idx, next_score, next_token, batch_beam_idx);
                    next_beam_scores[batch_idx][beam_idx] = next_score;
                    next_beam_tokens[batch_idx][beam_idx] = next_token;
                    next_beam_indices[batch_idx][beam_idx] = batch_beam_idx;
                    beam_idx++;
                }
                if (beam_idx == group_size) {
                    break;
                }
            }
        }
        float is_done = 0;
        if(beam_hyps_sizes[batch_idx][0] <  group_size) {
            is_done = 0;
        } else {
            if(early_stopping) {
                is_done = 1;
            } 
            else {
                // float best_sum_logprobs = torch::max(next_scores.index({batch_idx})).index({0}) / pow(cur_len, length_penalty);
                // is_done = beam_hyps[batch_idx][]
            }
        }
        float result = _done[batch_idx] + is_done;
        _done[batch_idx] = (result > 0);
    }
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
        _get_ngrams_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>((num_hypos * (cur_len - ngram_size + 1)), ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), num_hypos);
    }));


    dim3 two_grid_size(ceil((num_hypos) / 32.0), 1, 1);
    dim3 two_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_two_cuda", ([&] { 
        _get_ngrams_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(num_hypos, ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), scores.packed_accessor32<scalar_t, 2>(), num_hypos); 
    }));

    return generated_ngrams;
}

std::vector<torch::Tensor> _beam_search_process_cuda(
    int batch_size,
    int group_size,
    int max_token_length,
    float pad_token_id,
    float eos_token_id,

    int early_stopping,
    float length_penalty,
    torch::Tensor input_ids,
    torch::Tensor next_scores,
    torch::Tensor next_tokens,
    torch::Tensor next_indices,
    torch::Tensor next_beam_scores,
    torch::Tensor next_beam_tokens,
    torch::Tensor next_beam_indices,
    torch::Tensor _done,
    torch::Tensor beam_hyps,
    torch::Tensor beam_hyps_sizes
    )
{
    // torch::Device device(torch::kCUDA);
    // at::IntArrayRef beam_hyp_size({batch_size, 2 * group_size, max_token_length});
    // torch::Tensor beam_hyps = torch::zeros(beam_hyp_size, device=device);
    // at::IntArrayRef beam_hyps_sizes_size({batch_size, 3});
    // // std::cout << "max token length : " << max_token_length << std::endl;
    // // std::cout << "Next tokens size" << next_tokens.sizes() << "\n";
    // // std::cout << "Next tokens " << next_tokens << "\n";
    // // printf("Next tokens[0] = %f\n", next_tokens.data()[0][2]);
    // torch::Tensor beam_hyps_sizes = torch::zeros(beam_hyps_sizes_size, device = device);
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "[TOKEN]" << std::endl;
    // std::cout << "next_beam_scores scalar type : " << next_beam_scores.scalar_type() << std::endl;
    AT_DISPATCH_FLOATING_TYPES(next_beam_scores.scalar_type(), "_beam_search_process_cuda", ([&] {
        _beam_search_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            batch_size, 
            group_size,
            max_token_length, 
            pad_token_id,
            eos_token_id, 
            early_stopping,
            length_penalty,
            input_ids.packed_accessor32<scalar_t, 2>(),
            next_scores.packed_accessor32<scalar_t, 2>(),
            next_tokens.packed_accessor32<scalar_t, 2>(),
            next_indices.packed_accessor32<scalar_t, 2>(),
            next_beam_scores.packed_accessor32<scalar_t, 2>(),
            next_beam_tokens.packed_accessor32<scalar_t, 2>(),
            next_beam_indices.packed_accessor32<scalar_t, 2>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hyps.packed_accessor32<scalar_t, 3>(),
            beam_hyps_sizes.packed_accessor32<scalar_t, 2>()
        );
    }));
    std::vector<torch::Tensor> ret;
    ret.push_back(beam_hyps);
    ret.push_back(beam_hyps_sizes);
    return ret;
}

template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_one(
    int batch_size,
    int num_beams,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done, 
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta
    
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        if (_done[batch_idx] != 1) {
            for(int beam_id = 0; beam_id < num_beams; beam_id++) {
                int batch_beam_idx = batch_idx * num_beams + beam_id;
                float final_score = final_beam_scores[batch_beam_idx];
                float index;
                int scenario = -1;
                if (beam_hypotheses_meta[batch_idx][0] >= num_beams) {
                    index = beam_hypotheses_meta[batch_idx][2];
                    scenario = 1;
                } else {
                    index = beam_hypotheses_meta[batch_idx][0];
                    scenario = 2;
                }
                // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hypotheses_meta[batch_idx][0], group_size);
                if(scenario == 2) {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++) {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    beam_hypotheses_meta[batch_idx][0]++;
                    if (final_score < beam_hypotheses_meta[batch_idx][1]) {
                        beam_hypotheses_meta[batch_idx][1] = final_score;
                        beam_hypotheses_meta[batch_idx][2] = index;
                    }
                } else if (scenario == 1 && (final_score > beam_hypotheses_meta[batch_idx][1])) {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++) {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    float worst_score = CUDA_FLOAT32_MAX;
                    int worst_index = -1;
                    for(int j = 0 ; j < num_beams; j++) {
                        if (worst_score > beam_hypotheses[batch_idx][j][1]) {
                            worst_score = beam_hypotheses[batch_idx][j][1];
                            worst_index = j;
                        }
                    }
                    beam_hypotheses_meta[batch_idx][1] = worst_score;
                    beam_hypotheses_meta[batch_idx][2] = worst_index;
                }
            }           
        } 
    }
}


template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_two(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done, 
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta
    
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        int bitset = 0;
        for(int k = 0 ; k < (num_beam_hyps_to_keep); k++) {
            float best_score = CUDA_FLOAT32_MIN;
            int best_index = -1;
            for (int j = 0; j < num_beams; j++)
            {
                if (bitset & (1 << j)) continue;
                if (best_score < beam_hypotheses[batch_idx][j][1])
                {
                    best_score = beam_hypotheses[batch_idx][j][1];
                    best_index = j;
                }
            }
            beam_hypotheses[batch_idx][best_index][3] = k;
            bitset |= (1 << best_index);
        }
    }
}

void _beam_search_finalize_cuda(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::Tensor input_ids,
    torch::Tensor final_beam_scores,
    torch::Tensor beam_hypotheses,
    torch::Tensor beam_hypotheses_meta,
    torch::Tensor _done
) {
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "Came here\n";
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_one_cuda", ([&] {
        _beam_search_finalize_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            batch_size, 
            num_beams,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()
        );
    }));
    // std::cout << "Came here\n";

    dim3 two_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 two_block_size(32); 
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_two_cuda", ([&] {
        _beam_search_finalize_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(
            batch_size, 
            num_beams,
            num_beam_hyps_to_keep,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()
        );
    }));
}



template <typename scalar_t>
__global__ void temp_one_kernel_one(
    torch::PackedTensorAccessor32<scalar_t, 2> A, 
    torch::PackedTensorAccessor32<scalar_t, 3> B
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < A.size(0)) {
        A[id][0] = 10;
        B[id][0][0] = 20;
    }
}

torch::Tensor temp_one_cuda(torch::Tensor A, torch::Tensor B) {
    dim3 one_grid_size(ceil((A.size(0)) / 32.0), 1, 1);
    dim3 one_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "temp_one_cuda", ([&] { 
        temp_one_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            A.packed_accessor32<scalar_t, 2>(), 
            B.packed_accessor32<scalar_t, 3>()
        ); 
    }));
    return A;
}
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <vector>
#include <map>

#define CUDA_FLOAT32_MIN -10000
#define CUDA_FLOAT32_MAX 10000

// const int cur_len = 100;
// const int ngram_size = 3;
// const int num_hypos = 80;

// template <typename scalar_t>
// typedef struct
// {
//     scalar_t prev_ngram_tuple[ngram_size - 1];
//     scalar_t output_ngram;
// } custom_type;

using namespace torch::indexing;

template <typename scalar_t>
__global__ void _get_ngrams_kernel_one(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> output, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // output[id].output_ngram = id;
    // printf("Thread id %d\n", id);
    if (id < kernel_size) {
        int cur_len = prev_input_ids.size(1);
        output[id][0] = id / (cur_len - ngram_size + 1);
        for (int i = 1; i < ngram_size; i++)
        {
            output[id][i] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + i - 1];
        }
        output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
    }
}

template <typename scalar_t>
__global__ void _get_ngrams_kernel_two(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> generated_ngrams, torch::PackedTensorAccessor32<scalar_t, 2> scores, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < kernel_size) {
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
    }
    // output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
}

template <typename scalar_t>
__global__ void _beam_search_kernel_one(
    float batch_size,
    float group_size,
    float max_token_length,
    float pad_token_id,
    float eos_token_id,
    float early_stopping,
    float length_penalty,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 2> next_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_indices,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_indices,
    torch::PackedTensorAccessor32<scalar_t, 1> _done,
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hyps,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hyps_sizes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        if (_done[batch_idx] == 1) {
            for(int j = 0 ; j < group_size; j++) {
                next_beam_scores[batch_idx][j] = 0;
                next_beam_tokens[batch_idx][j] = pad_token_id;
                next_beam_indices[batch_idx][j] = 0;
            }
        } else {
            int num_tokens = 2 * group_size;
            int beam_idx = 0;
            for(int beam_token_rank = 0; beam_token_rank < num_tokens; beam_token_rank++) {
                float next_token = next_tokens[batch_idx][beam_token_rank];
                float next_index = next_indices[batch_idx][beam_token_rank];
                float next_score = next_scores[batch_idx][beam_token_rank];
                float batch_beam_idx = batch_idx * group_size + next_index;
                if((next_token == eos_token_id)) {
                    
                    if (beam_token_rank < group_size) {

                        float index;
                        int scenario = -1;
                        if (beam_hyps_sizes[batch_idx][0] >= group_size) {
                            index = beam_hyps_sizes[batch_idx][2];
                            scenario = 1;
                        } else {
                            index = beam_hyps_sizes[batch_idx][0];
                            scenario = 2;
                        }
                        // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hyps_sizes[batch_idx][0], group_size);
                        if(scenario == 2) {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score / pow(input_ids.size(1), length_penalty);
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++) {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            beam_hyps_sizes[batch_idx][0]++;
                            if (next_score < beam_hyps_sizes[batch_idx][1]) {
                                beam_hyps_sizes[batch_idx][1] = next_score;
                                beam_hyps_sizes[batch_idx][2] = index;
                            }
                        } else if (scenario == 1 && (next_score > beam_hyps_sizes[batch_idx][1])) {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score;
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++) {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            float worst_score = CUDA_FLOAT32_MAX;
                            int worst_index = -1;
                            for(int j = 0 ; j < group_size; j++) {
                                if (worst_score > beam_hyps[batch_idx][j][1]) {
                                    worst_score = beam_hyps[batch_idx][j][1];
                                    worst_index = j;
                                }
                            }
                            beam_hyps_sizes[batch_idx][1] = worst_score;
                            beam_hyps_sizes[batch_idx][2] = worst_index;
                        }
                    } 
                } else {
                    // printf("[%d][%d] Adding %f %f %f\n", batch_idx, beam_idx, next_score, next_token, batch_beam_idx);
                    next_beam_scores[batch_idx][beam_idx] = next_score;
                    next_beam_tokens[batch_idx][beam_idx] = next_token;
                    next_beam_indices[batch_idx][beam_idx] = batch_beam_idx;
                    beam_idx++;
                }
                if (beam_idx == group_size) {
                    break;
                }
            }
        }
        float is_done = 0;
        if(beam_hyps_sizes[batch_idx][0] <  group_size) {
            is_done = 0;
        } else {
            if(early_stopping) {
                is_done = 1;
            } 
            else {
                // float best_sum_logprobs = torch::max(next_scores.index({batch_idx})).index({0}) / pow(cur_len, length_penalty);
                // is_done = beam_hyps[batch_idx][]
            }
        }
        float result = _done[batch_idx] + is_done;
        _done[batch_idx] = (result > 0);
    }
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
        _get_ngrams_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>((num_hypos * (cur_len - ngram_size + 1)), ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), num_hypos);
    }));


    dim3 two_grid_size(ceil((num_hypos) / 32.0), 1, 1);
    dim3 two_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_two_cuda", ([&] { 
        _get_ngrams_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(num_hypos, ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), scores.packed_accessor32<scalar_t, 2>(), num_hypos); 
    }));

    return generated_ngrams;
}

std::vector<torch::Tensor> _beam_search_process_cuda(
    int batch_size,
    int group_size,
    int max_token_length,
    float pad_token_id,
    float eos_token_id,

    int early_stopping,
    float length_penalty,
    torch::Tensor input_ids,
    torch::Tensor next_scores,
    torch::Tensor next_tokens,
    torch::Tensor next_indices,
    torch::Tensor next_beam_scores,
    torch::Tensor next_beam_tokens,
    torch::Tensor next_beam_indices,
    torch::Tensor _done,
    torch::Tensor beam_hyps,
    torch::Tensor beam_hyps_sizes
    )
{
    // torch::Device device(torch::kCUDA);
    // at::IntArrayRef beam_hyp_size({batch_size, 2 * group_size, max_token_length});
    // torch::Tensor beam_hyps = torch::zeros(beam_hyp_size, device=device);
    // at::IntArrayRef beam_hyps_sizes_size({batch_size, 3});
    // // std::cout << "max token length : " << max_token_length << std::endl;
    // // std::cout << "Next tokens size" << next_tokens.sizes() << "\n";
    // // std::cout << "Next tokens " << next_tokens << "\n";
    // // printf("Next tokens[0] = %f\n", next_tokens.data()[0][2]);
    // torch::Tensor beam_hyps_sizes = torch::zeros(beam_hyps_sizes_size, device = device);
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "[TOKEN]" << std::endl;
    // std::cout << "next_beam_scores scalar type : " << next_beam_scores.scalar_type() << std::endl;
    AT_DISPATCH_FLOATING_TYPES(next_beam_scores.scalar_type(), "_beam_search_process_cuda", ([&] {
        _beam_search_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            batch_size, 
            group_size,
            max_token_length, 
            pad_token_id,
            eos_token_id, 
            early_stopping,
            length_penalty,
            input_ids.packed_accessor32<scalar_t, 2>(),
            next_scores.packed_accessor32<scalar_t, 2>(),
            next_tokens.packed_accessor32<scalar_t, 2>(),
            next_indices.packed_accessor32<scalar_t, 2>(),
            next_beam_scores.packed_accessor32<scalar_t, 2>(),
            next_beam_tokens.packed_accessor32<scalar_t, 2>(),
            next_beam_indices.packed_accessor32<scalar_t, 2>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hyps.packed_accessor32<scalar_t, 3>(),
            beam_hyps_sizes.packed_accessor32<scalar_t, 2>()
        );
    }));
    std::vector<torch::Tensor> ret;
    ret.push_back(beam_hyps);
    ret.push_back(beam_hyps_sizes);
    return ret;
}

template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_one(
    int batch_size,
    int num_beams,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done, 
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta
    
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        if (_done[batch_idx] != 1) {
            for(int beam_id = 0; beam_id < num_beams; beam_id++) {
                int batch_beam_idx = batch_idx * num_beams + beam_id;
                float final_score = final_beam_scores[batch_beam_idx];
                float index;
                int scenario = -1;
                if (beam_hypotheses_meta[batch_idx][0] >= num_beams) {
                    index = beam_hypotheses_meta[batch_idx][2];
                    scenario = 1;
                } else {
                    index = beam_hypotheses_meta[batch_idx][0];
                    scenario = 2;
                }
                // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hypotheses_meta[batch_idx][0], group_size);
                if(scenario == 2) {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++) {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    beam_hypotheses_meta[batch_idx][0]++;
                    if (final_score < beam_hypotheses_meta[batch_idx][1]) {
                        beam_hypotheses_meta[batch_idx][1] = final_score;
                        beam_hypotheses_meta[batch_idx][2] = index;
                    }
                } else if (scenario == 1 && (final_score > beam_hypotheses_meta[batch_idx][1])) {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++) {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    float worst_score = CUDA_FLOAT32_MAX;
                    int worst_index = -1;
                    for(int j = 0 ; j < num_beams; j++) {
                        if (worst_score > beam_hypotheses[batch_idx][j][1]) {
                            worst_score = beam_hypotheses[batch_idx][j][1];
                            worst_index = j;
                        }
                    }
                    beam_hypotheses_meta[batch_idx][1] = worst_score;
                    beam_hypotheses_meta[batch_idx][2] = worst_index;
                }
            }           
        } 
    }
}


template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_two(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done, 
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta
    
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        int bitset = 0;
        for(int k = 0 ; k < (num_beam_hyps_to_keep); k++) {
            float best_score = CUDA_FLOAT32_MIN;
            int best_index = -1;
            for (int j = 0; j < num_beams; j++)
            {
                if (bitset & (1 << j)) continue;
                if (best_score < beam_hypotheses[batch_idx][j][1])
                {
                    best_score = beam_hypotheses[batch_idx][j][1];
                    best_index = j;
                }
            }
            beam_hypotheses[batch_idx][best_index][3] = k;
            bitset |= (1 << best_index);
        }
    }
}

void _beam_search_finalize_cuda(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::Tensor input_ids,
    torch::Tensor final_beam_scores,
    torch::Tensor beam_hypotheses,
    torch::Tensor beam_hypotheses_meta,
    torch::Tensor _done
) {
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "Came here\n";
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_one_cuda", ([&] {
        _beam_search_finalize_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            batch_size, 
            num_beams,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()
        );
    }));
    // std::cout << "Came here\n";

    dim3 two_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 two_block_size(32); 
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_two_cuda", ([&] {
        _beam_search_finalize_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(
            batch_size, 
            num_beams,
            num_beam_hyps_to_keep,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()
        );
    }));
}



template <typename scalar_t>
__global__ void temp_one_kernel_one(
    torch::PackedTensorAccessor32<scalar_t, 2> A, 
    torch::PackedTensorAccessor32<scalar_t, 3> B
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < A.size(0)) {
        A[id][0] = 10;
        B[id][0][0] = 20;
    }
}

torch::Tensor temp_one_cuda(torch::Tensor A, torch::Tensor B) {
    dim3 one_grid_size(ceil((A.size(0)) / 32.0), 1, 1);
    dim3 one_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "temp_one_cuda", ([&] { 
        temp_one_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            A.packed_accessor32<scalar_t, 2>(), 
            B.packed_accessor32<scalar_t, 3>()
        ); 
    }));
    return A;
}
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <vector>
#include <map>

#define CUDA_FLOAT32_MIN -10000
#define CUDA_FLOAT32_MAX 10000

// const int cur_len = 100;
// const int ngram_size = 3;
// const int num_hypos = 80;

// template <typename scalar_t>
// typedef struct
// {
//     scalar_t prev_ngram_tuple[ngram_size - 1];
//     scalar_t output_ngram;
// } custom_type;

using namespace torch::indexing;

template <typename scalar_t>
__global__ void _get_ngrams_kernel_one(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> output, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // output[id].output_ngram = id;
    // printf("Thread id %d\n", id);
    if (id < kernel_size)
    {
        int cur_len = prev_input_ids.size(1);
        output[id][0] = id / (cur_len - ngram_size + 1);
        for (int i = 1; i < ngram_size; i++)
        {
            output[id][i] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + i - 1];
        }
        output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
    }
}

template <typename scalar_t>
__global__ void _get_ngrams_kernel_two(int kernel_size, int ngram_size, torch::PackedTensorAccessor32<scalar_t, 2> prev_input_ids, torch::PackedTensorAccessor32<scalar_t, 2> generated_ngrams, torch::PackedTensorAccessor32<scalar_t, 2> scores, int num_hypos)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < kernel_size)
    {
        // generated_ngrams[id].generated_ngrams_ngram = id;
        // printf("Thread id %d\n", id);
        int cur_len = prev_input_ids.size(1);
        // generated_ngrams[id][0] = id / (cur_len - ngram_size + 1);
        for (int i = 0; i < generated_ngrams.size(0); i++)
        {
            int count = 0;
            for (int j = 0; j < generated_ngrams.size(1) - 1; j++)
            {
                if (j == 0)
                {
                    if (id == generated_ngrams[i][0])
                        ++count;
                }
                else
                {
                    if (generated_ngrams[i][j] == prev_input_ids[id][cur_len - ngram_size + j])
                        ++count;
                }
            }
            if ((count == generated_ngrams.size(1) - 1) and (id < scores.size(0)))
            {
                scores[id][(int)generated_ngrams[i][generated_ngrams.size(1) - 1]] = CUDA_FLOAT32_MIN;
            }
        }
    }
    // output[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
}

template <typename scalar_t>
__global__ void _beam_search_kernel_one(
    float batch_size,
    float group_size,
    float max_token_length,
    float pad_token_id,
    float eos_token_id,
    float early_stopping,
    float length_penalty,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 2> next_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_indices,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_tokens,
    torch::PackedTensorAccessor32<scalar_t, 2> next_beam_indices,
    torch::PackedTensorAccessor32<scalar_t, 1> _done,
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hyps,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hyps_sizes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size)
    {
        if (_done[batch_idx] == 1)
        {
            for (int j = 0; j < group_size; j++)
            {
                next_beam_scores[batch_idx][j] = 0;
                next_beam_tokens[batch_idx][j] = pad_token_id;
                next_beam_indices[batch_idx][j] = 0;
            }
        }
        else
        {
            int num_tokens = 2 * group_size;
            int beam_idx = 0;
            for (int beam_token_rank = 0; beam_token_rank < num_tokens; beam_token_rank++)
            {
                float next_token = next_tokens[batch_idx][beam_token_rank];
                float next_index = next_indices[batch_idx][beam_token_rank];
                float next_score = next_scores[batch_idx][beam_token_rank];
                float batch_beam_idx = batch_idx * group_size + next_index;
                if ((next_token == eos_token_id))
                {

                    if (beam_token_rank < group_size)
                    {

                        float index;
                        int scenario = -1;
                        if (beam_hyps_sizes[batch_idx][0] >= group_size)
                        {
                            index = beam_hyps_sizes[batch_idx][2];
                            scenario = 1;
                        }
                        else
                        {
                            index = beam_hyps_sizes[batch_idx][0];
                            scenario = 2;
                        }
                        // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hyps_sizes[batch_idx][0], group_size);
                        if (scenario == 2)
                        {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score / pow(input_ids.size(1), length_penalty);
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++)
                            {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            beam_hyps_sizes[batch_idx][0]++;
                            if (next_score < beam_hyps_sizes[batch_idx][1])
                            {
                                beam_hyps_sizes[batch_idx][1] = next_score;
                                beam_hyps_sizes[batch_idx][2] = index;
                            }
                        }
                        else if (scenario == 1 && (next_score > beam_hyps_sizes[batch_idx][1]))
                        {
                            // beam_hyps_sizes[batch_idx][0] = beam_hyps_sizes[batch_idx][0] + 1;
                            beam_hyps[batch_idx][index][0] = input_ids.size(1);
                            beam_hyps[batch_idx][index][1] = next_score;
                            beam_hyps[batch_idx][index][2] = next_index;
                            beam_hyps[batch_idx][index][3] = -1;
                            for (int j = 0; j < input_ids.size(1); j++)
                            {
                                beam_hyps[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                            }
                            float worst_score = CUDA_FLOAT32_MAX;
                            int worst_index = -1;
                            for (int j = 0; j < group_size; j++)
                            {
                                if (worst_score > beam_hyps[batch_idx][j][1])
                                {
                                    worst_score = beam_hyps[batch_idx][j][1];
                                    worst_index = j;
                                }
                            }
                            beam_hyps_sizes[batch_idx][1] = worst_score;
                            beam_hyps_sizes[batch_idx][2] = worst_index;
                        }
                    }
                }
                else
                {
                    // printf("[%d][%d] Adding %f %f %f\n", batch_idx, beam_idx, next_score, next_token, batch_beam_idx);
                    next_beam_scores[batch_idx][beam_idx] = next_score;
                    next_beam_tokens[batch_idx][beam_idx] = next_token;
                    next_beam_indices[batch_idx][beam_idx] = batch_beam_idx;
                    beam_idx++;
                }
                if (beam_idx == group_size)
                {
                    break;
                }
            }
        }
        float is_done = 0;
        if (beam_hyps_sizes[batch_idx][0] < group_size)
        {
            is_done = 0;
        }
        else
        {
            if (early_stopping)
            {
                is_done = 1;
            }
            else
            {
                // float best_sum_logprobs = torch::max(next_scores.index({batch_idx})).index({0}) / pow(cur_len, length_penalty);
                // is_done = beam_hyps[batch_idx][]
            }
        }
        float result = _done[batch_idx] + is_done;
        _done[batch_idx] = (result > 0);
    }
}

torch::Tensor _get_ngrams_cuda(int ngram_size,
                               torch::Tensor prev_input_ids,
                               torch::Tensor scores,
                               int num_hypos)
{

    torch::Device device(torch::kCUDA);
    int cur_len = prev_input_ids.size(1);
    dim3 one_grid_size(ceil((num_hypos * (cur_len - ngram_size + 1)) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // TORCH_WARN("Using grid size ");
    // TORCH_WARN(ceil((num_hypos * (cur_len - ngram_size + 1)) / 32.0));
    at::IntArrayRef generated_ngrams_size({(num_hypos * (cur_len - ngram_size + 1)), ngram_size + 1});
    torch::Tensor generated_ngrams = torch::zeros(generated_ngrams_size, device);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_one_cuda", ([&] { 
        _get_ngrams_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>((num_hypos * (cur_len - ngram_size + 1)), ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), num_hypos); 
    }));

    dim3 two_grid_size(ceil((num_hypos) / 32.0), 1, 1);
    dim3 two_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "_get_ngrams_two_cuda", ([&] { 
        _get_ngrams_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(num_hypos, ngram_size, prev_input_ids.packed_accessor32<scalar_t, 2>(), generated_ngrams.packed_accessor32<scalar_t, 2>(), scores.packed_accessor32<scalar_t, 2>(), num_hypos); 
    }));

    return generated_ngrams;
}

std::vector<torch::Tensor> _beam_search_process_cuda(
    int batch_size,
    int group_size,
    int max_token_length,
    float pad_token_id,
    float eos_token_id,

    int early_stopping,
    float length_penalty,
    torch::Tensor input_ids,
    torch::Tensor next_scores,
    torch::Tensor next_tokens,
    torch::Tensor next_indices,
    torch::Tensor next_beam_scores,
    torch::Tensor next_beam_tokens,
    torch::Tensor next_beam_indices,
    torch::Tensor _done,
    torch::Tensor beam_hyps,
    torch::Tensor beam_hyps_sizes)
{
    // torch::Device device(torch::kCUDA);
    // at::IntArrayRef beam_hyp_size({batch_size, 2 * group_size, max_token_length});
    // torch::Tensor beam_hyps = torch::zeros(beam_hyp_size, device=device);
    // at::IntArrayRef beam_hyps_sizes_size({batch_size, 3});
    // // std::cout << "max token length : " << max_token_length << std::endl;
    // // std::cout << "Next tokens size" << next_tokens.sizes() << "\n";
    // // std::cout << "Next tokens " << next_tokens << "\n";
    // // printf("Next tokens[0] = %f\n", next_tokens.data()[0][2]);
    // torch::Tensor beam_hyps_sizes = torch::zeros(beam_hyps_sizes_size, device = device);
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "[TOKEN]" << std::endl;
    // std::cout << "next_beam_scores scalar type : " << next_beam_scores.scalar_type() << std::endl;
    AT_DISPATCH_FLOATING_TYPES(next_beam_scores.scalar_type(), "_beam_search_process_cuda", ([&] { 
        _beam_search_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
        batch_size,
        group_size,
        max_token_length,
        pad_token_id,
        eos_token_id,
        early_stopping,
        length_penalty,
        input_ids.packed_accessor32<scalar_t, 2>(),
        next_scores.packed_accessor32<scalar_t, 2>(),
        next_tokens.packed_accessor32<scalar_t, 2>(),
        next_indices.packed_accessor32<scalar_t, 2>(),
        next_beam_scores.packed_accessor32<scalar_t, 2>(),
        next_beam_tokens.packed_accessor32<scalar_t, 2>(),
        next_beam_indices.packed_accessor32<scalar_t, 2>(),
        _done.packed_accessor32<scalar_t, 1>(),
        beam_hyps.packed_accessor32<scalar_t, 3>(),
        beam_hyps_sizes.packed_accessor32<scalar_t, 2>()); 
    }));
    std::vector<torch::Tensor> ret;
    ret.push_back(beam_hyps);
    ret.push_back(beam_hyps_sizes);
    return ret;
}

template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_one(
    int batch_size,
    int num_beams,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done,
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta

)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size)
    {
        if (_done[batch_idx] != 1)
        {
            for (int beam_id = 0; beam_id < num_beams; beam_id++)
            {
                int batch_beam_idx = batch_idx * num_beams + beam_id;
                float final_score = final_beam_scores[batch_beam_idx];
                float index;
                int scenario = -1;
                if (beam_hypotheses_meta[batch_idx][0] >= num_beams)
                {
                    index = beam_hypotheses_meta[batch_idx][2];
                    scenario = 1;
                }
                else
                {
                    index = beam_hypotheses_meta[batch_idx][0];
                    scenario = 2;
                }
                // printf("[%d] Entered here: scenario %d, %f, %f\n", batch_idx, scenario, beam_hypotheses_meta[batch_idx][0], group_size);
                if (scenario == 2)
                {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++)
                    {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    beam_hypotheses_meta[batch_idx][0]++;
                    if (final_score < beam_hypotheses_meta[batch_idx][1])
                    {
                        beam_hypotheses_meta[batch_idx][1] = final_score;
                        beam_hypotheses_meta[batch_idx][2] = index;
                    }
                }
                else if (scenario == 1 && (final_score > beam_hypotheses_meta[batch_idx][1]))
                {
                    // beam_hypotheses_meta[batch_idx][0] = beam_hypotheses_meta[batch_idx][0] + 1;
                    beam_hypotheses[batch_idx][index][0] = input_ids.size(1);
                    beam_hypotheses[batch_idx][index][1] = final_score;
                    beam_hypotheses[batch_idx][index][2] = -1;
                    beam_hypotheses[batch_idx][index][3] = -1;
                    for (int j = 0; j < input_ids.size(1); j++)
                    {
                        beam_hypotheses[batch_idx][index][j + 4] = input_ids[batch_beam_idx][j];
                    }
                    float worst_score = CUDA_FLOAT32_MAX;
                    int worst_index = -1;
                    for (int j = 0; j < num_beams; j++)
                    {
                        if (worst_score > beam_hypotheses[batch_idx][j][1])
                        {
                            worst_score = beam_hypotheses[batch_idx][j][1];
                            worst_index = j;
                        }
                    }
                    beam_hypotheses_meta[batch_idx][1] = worst_score;
                    beam_hypotheses_meta[batch_idx][2] = worst_index;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void _beam_search_finalize_kernel_two(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::PackedTensorAccessor32<scalar_t, 2> input_ids,
    torch::PackedTensorAccessor32<scalar_t, 1> final_beam_scores,
    torch::PackedTensorAccessor32<scalar_t, 1> _done,
    torch::PackedTensorAccessor32<scalar_t, 3> beam_hypotheses,
    torch::PackedTensorAccessor32<scalar_t, 2> beam_hypotheses_meta

)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size)
    {
        int bitset = 0;
        for (int k = 0; k < (num_beam_hyps_to_keep); k++)
        {
            float best_score = CUDA_FLOAT32_MIN;
            int best_index = -1;
            for (int j = 0; j < num_beams; j++)
            {
                if (bitset & (1 << j))
                    continue;
                if (best_score < beam_hypotheses[batch_idx][j][1])
                {
                    best_score = beam_hypotheses[batch_idx][j][1];
                    best_index = j;
                }
            }
            beam_hypotheses[batch_idx][best_index][3] = k;
            bitset |= (1 << best_index);
        }
    }
}

void _beam_search_finalize_cuda(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::Tensor input_ids,
    torch::Tensor final_beam_scores,
    torch::Tensor beam_hypotheses,
    torch::Tensor beam_hypotheses_meta,
    torch::Tensor _done)
{
    dim3 one_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 one_block_size(32);
    // std::cout << "Came here\n";
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_one_cuda", ([&] { 
        _beam_search_finalize_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
            batch_size,
            num_beams,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()); 
        })
    );
    // std::cout << "Came here\n";

    dim3 two_grid_size(ceil((batch_size) / 32.0), 1, 1);
    dim3 two_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(final_beam_scores.scalar_type(), "_beam_search_finalize_two_cuda", ([&] { 
        _beam_search_finalize_kernel_two<scalar_t><<<two_grid_size, two_block_size>>>(
            batch_size,
            num_beams,
            num_beam_hyps_to_keep,
            input_ids.packed_accessor32<scalar_t, 2>(),
            final_beam_scores.packed_accessor32<scalar_t, 1>(),
            _done.packed_accessor32<scalar_t, 1>(),
            beam_hypotheses.packed_accessor32<scalar_t, 3>(),
            beam_hypotheses_meta.packed_accessor32<scalar_t, 2>()); 
        })
    );
}

template <typename scalar_t>
__global__ void temp_one_kernel_one(
    torch::PackedTensorAccessor32<scalar_t, 2> A,
    torch::PackedTensorAccessor32<scalar_t, 3> B)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < A.size(0))
    {
        A[id][0] = 10;
        B[id][0][0] = 20;
    }
}

torch::Tensor temp_one_cuda(torch::Tensor A, torch::Tensor B)
{
    dim3 one_grid_size(ceil((A.size(0)) / 32.0), 1, 1);
    dim3 one_block_size(32);
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "temp_one_cuda", ([&] { 
        temp_one_kernel_one<scalar_t><<<one_grid_size, one_block_size>>>(
        A.packed_accessor32<scalar_t, 2>(),
        B.packed_accessor32<scalar_t, 3>()); 
    }));
    return A;
}
