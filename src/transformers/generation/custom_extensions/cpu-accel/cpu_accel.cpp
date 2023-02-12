#include <torch/extension.h>
#include <vector>
#include <map>
#include <omp.h>

// CUDA declarations
// C++ Interface

#define CUDA_FLOAT32_MIN -10000

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor NoRepeatNGramLogitsProcessor(int ngram_size,
                                           torch::Tensor prev_input_ids,
                                           torch::Tensor scores,
                                           int num_hypos)
{
    CHECK_INPUT(prev_input_ids);
    CHECK_INPUT(scores);
    // Bring both prev_input_ids and scores to the CPU
    torch::Device cpuDevice(torch::kCPU);
    prev_input_ids = prev_input_ids.to(cpuDevice);
    scores = scores.to(cpuDevice);

    at::IntArrayRef generated_ngrams_size({(num_hypos * (cur_len - ngram_size + 1)), ngram_size + 1});
    torch::Tensor generated_ngrams = torch::zeros(generated_ngrams_size, cpuDevice);

    omp_set_num_threads(8);
    #pragma omp parallel shared(prev_input_ids, scores, generated_ngrams, ngram_size) {
        int cur_len = prev_input_ids.size(1);
        #pragma omp for
        for (int id = 0; id < (num_hypos * (cur_len - ngram_size + 1)); id++) {
            generated_ngrams[id][0] = id / (cur_len - ngram_size + 1);
            for (int i = 1; i < ngram_size; i++) {
                generated_ngrams[id][i] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + i - 1];
            }
            generated_ngrams[id][ngram_size] = prev_input_ids[id / (cur_len - ngram_size + 1)][(id % (cur_len - ngram_size + 1)) + ngram_size - 1];
        }

        #pragma omp for
        for (int id = 0; id < num_hypos; id++) {
            for (int i = 0; i < 0; i < generated_ngrams.size(0); i++)
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
    }
    omp_set_num_threads(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("NoRepeatNGramLogitsProcessor", &NoRepeatNGramLogitsProcessor, "cpu accelerated NoRepeatNGramLogitsProcessor");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
