#include <torch/extension.h>
#include <vector>
#include <map>
#include <cuda_runtime.h>

// CUDA declarations

torch::Tensor _get_ngrams_cuda(int ngram_size,
                               torch::Tensor prev_input_ids,
                               torch::Tensor scores,
                               int num_hypos);
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
    torch::Tensor beam_hyps_sizes);

void _beam_search_finalize_cuda(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::Tensor input_ids,
    torch::Tensor final_beam_scores,
    torch::Tensor beam_hypotheses,
    torch::Tensor beam_hypotheses_meta,
    torch::Tensor _done
);

torch::Tensor temp_one_cuda(torch::Tensor A, torch::Tensor B);
// C++ Interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

    torch::Tensor _get_ngrams(int ngram_size,
                              torch::Tensor prev_input_ids,
                              torch::Tensor scores,
                              int num_hypos) {
    CHECK_INPUT(prev_input_ids);
    return _get_ngrams_cuda(ngram_size, prev_input_ids, scores, num_hypos);
}

std::vector<torch::Tensor> _beam_search_process(
    int batch_size,
    int group_size,
    float pad_token_id,
    float eos_token_id,
    int max_token_length,
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
    CHECK_INPUT(next_beam_scores);
    CHECK_INPUT(next_beam_tokens);
    CHECK_INPUT(next_beam_indices);
    CHECK_INPUT(_done);
    return _beam_search_process_cuda(
        batch_size,
        group_size,
        max_token_length,
        pad_token_id,
        eos_token_id,
        early_stopping,
        length_penalty,
        input_ids,
        next_scores,
        next_tokens,
        next_indices,
        next_beam_scores,
        next_beam_tokens,
        next_beam_indices,
        _done,
        beam_hyps,
        beam_hyps_sizes);
}

void _beam_search_finalize(
    int batch_size,
    int num_beams,
    int num_beam_hyps_to_keep,
    torch::Tensor input_ids,
    torch::Tensor final_beam_scores,
    torch::Tensor beam_hypotheses,
    torch::Tensor beam_hypotheses_meta,
    torch::Tensor _done)
{
    CHECK_INPUT(input_ids);
    CHECK_INPUT(final_beam_scores);
    CHECK_INPUT(beam_hypotheses);
    CHECK_INPUT(beam_hypotheses_meta);
    _beam_search_finalize_cuda(
        batch_size,
        num_beams,
        num_beam_hyps_to_keep,
        input_ids,
        final_beam_scores,
        beam_hypotheses,
        beam_hypotheses_meta,
        _done
    );
}

torch::Tensor temp_one(torch::Tensor A, torch::Tensor B) {
    return temp_one_cuda(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_get_ngrams", &_get_ngrams, "Get ngrams (CUDA)");
    m.def("_beam_search_process", &_beam_search_process, "Beam Search Process(CUDA)");
    m.def("temp_one", &temp_one, "Temp for testing(CUDA)");
    m.def("_beam_search_finalize", &_beam_search_finalize, "Beam Search Finalize(CUDA)");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
