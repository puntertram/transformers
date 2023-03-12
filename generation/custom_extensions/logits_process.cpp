#include <torch/extension.h>
#include <vector>
#include <map>
#include <cuda_runtime.h>

// CUDA declarations

torch::Tensor _get_ngrams_cuda(int ngram_size,
                               torch::Tensor prev_input_ids,
                               torch::Tensor scores,
                               int num_hypos);
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



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_get_ngrams", &_get_ngrams, "Get ngrams (CUDA)");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
