from enum import Enum
import torch
import random
from custom_time_profile_gpu import measure_times

from workload_partitioner import PARTITION_RESIDENT_DEVICES, PARTITION_TYPES
from .generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from .generation.configuration_utils import GenerationConfig
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class PARTITION_TYPES(Enum):
        BASELINE = 1
        CPU_GPU = 2
        GPU = 3
        CPU = 4

        
class PARTITION_RESIDENT_DEVICES(Enum):
    '''
        1. CPU: This means that the partition bundle sits on the CPU
        2. GPU: This means that the partition bundle sits on the GPU
        3. CPU_GPU: This means that the parition bundle initially resided on the CPU then got transferred to the GPU
    '''
    CPU = 1
    GPU = 2
    CPU_GPU = 3
    BASELINE = 4

class WorkloadBundle:
    def __init__(
        self, 
        resident_device: PARTITION_RESIDENT_DEVICES
    ) -> None:
        self.input_ids = None
        self.resident_device = resident_device


    def transfer(
        self,
        to: PARTITION_RESIDENT_DEVICES
    ):
        assert self.resident_device == PARTITION_RESIDENT_DEVICES.CPU and to == PARTITION_RESIDENT_DEVICES.GPU, f"Expected  \
            cpu to gpu transfer, got {self.resident_device} to {to} transfer..."
        if self.resident_device == PARTITION_RESIDENT_DEVICES.CPU and to == PARTITION_RESIDENT_DEVICES.GPU:
            # Transfer all the tensors to the GPU
            pass
        

    def add_encoder_inputs_tensor(
        self,
        inputs_tensor:torch.LongTensor,
        generation_config,
        input_ids_seq_length,
        prefix_allowed_tokens_fn,
        logits_processor
    ):
        self.encoder_inputs_tensor = inputs_tensor 
        # Also initialize the logits_processor as it needs encoder_inputs_tensor as the input
        self.logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
    @measure_times
    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
                )
            )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            if self.resident_device == PARTITION_RESIDENT_DEVICES.CPU:
                processors.append(NoRepeatNGramLogitsProcessorCPU(generation_config.no_repeat_ngram_size))
            elif self.resident_device == PARTITION_RESIDENT_DEVICES.GPU:
                processors.append(NoRepeatNGramLogitsProcessorGPU(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            if self.config.is_encoder_decoder:
                processors.append(
                    EncoderNoRepeatNGramLogitsProcessor(
                        generation_config.encoder_no_repeat_ngram_size, encoder_input_ids
                    )
                )
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config.eos_token_id,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors  

    def add_decoder_input_ids(self, input_ids):
        self.decoder_input_ids = input_ids


class WorkloadPartitioner:
    def __init__(self, partition_type: PARTITION_TYPES) -> None:
        self.joined = False
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        



    def get_sequence_outputs(self):
        assert self.joined, f"Did not join both the cpu and gpu versions, self.joined must be True, but found False instead..."
        return self.sequence_outputs
    
    def join(self):
        self.joined = True
    
    def partition_workload_pre_encoder(self, inputs_tensor: torch.LongTensor, partition_type: PARTITION_TYPES, split_ratio: int):
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        if partition_type == PARTITION_TYPES.CPU_GPU:
            indices = [i for i in range(inputs_tensor.shape[0])]
            cpu_indices = random.sample(indices, k=int(split_ratio*inputs_tensor.shape[0]))
            gpu_indices = list(filter(lambda x: x not in cpu_indices, indices))
            self.mapping_function = {"cpu": cpu_indices, "gpu": gpu_indices}
            inputs_tensor_cpu = inputs_tensor[cpu_indices].to("cpu")
            inputs_tensor_gpu = inputs_tensor[gpu_indices]
            self.gpu_bundle.add_encoder_inputs_tensor(inputs_tensor_gpu)
            self.cpu_bundle.add_encoder_inputs_tensor(inputs_tensor_cpu)
        elif partition_type == PARTITION_TYPES.GPU:
            self.gpu_bundle.add_encoder_inputs_tensor(inputs_tensor)
        elif partition_type == PARTITION_TYPES.BASELINE:
            self.bundle.add_encoder_inputs_tensor(inputs_tensor)
    
    def partition_workload_pre_decoder(self, input_ids: torch.LongTensor, partition_type: PARTITION_TYPES, split_ratio: int):
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        if partition_type == PARTITION_TYPES.CPU_GPU:
            input_ids_cpu = input_ids[self.mapping_function["cpu"]].to("cpu")
            input_ids_gpu = input_ids[self.mapping_function["gpu"]]
            self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
            self.cpu_bundle.add_decoder_input_ids(input_ids_cpu)
        elif partition_type == PARTITION_TYPES.GPU:
            self.gpu_bundle.add_decoder_input_ids(input_ids)
        elif partition_type == PARTITION_TYPES.BASELINE:
            self.bundle.add_decoder_input_ids(input_ids)
        

    def transfer_partition(self, **kwargs):
        assert kwargs is not None, f"{kwargs} passed to transfer partition method should not be None"
        assert kwargs["from"] == PARTITION_RESIDENT_DEVICES.CPU, f"Dont know how to handle transfer from \
            {kwargs['from']} to {kwargs['to']}, only cpu to gpu transfer supported for now..."

        if kwargs["from"] == PARTITION_RESIDENT_DEVICES.CPU:
            #  Transfer the cpu bundle to the gpu
            self.cpu_bundle.transfer(kwargs["to"])
            self.cpu_bundle.resident_device = PARTITION_RESIDENT_DEVICES.CPU_GPU


    def add_sequence_outputs(self, sequence_outputs, resident_device:PARTITION_RESIDENT_DEVICES):
        if resident_device == PARTITION_RESIDENT_DEVICES.GPU:
            self.gpu_bundle.sequence_outputs = sequence_outputs
        elif resident_device == PARTITION_RESIDENT_DEVICES.CPU or resident_device == PARTITION_RESIDENT_DEVICES.CPU_GPU:
             self.cpu_bundle.sequence_outputs = sequence_outputs
    
