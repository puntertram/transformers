from enum import Enum
import torch
import random
from custom_time_profile_gpu import measure_times

from .generation.beam_search import BeamSearchScorer, BeamSearchScorerCPU, BeamSearchScorerGPU

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
    NoRepeatNGramLogitsProcessorCPU,
    NoRepeatNGramLogitsProcessorGPU,
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
        
    @measure_times
    def _merge_criteria_processor_list(
        self,
        default_list: LogitsProcessorList,
        custom_list: LogitsProcessorList,
    ) -> LogitsProcessorList:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list


    @measure_times
    def init_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        **kwargs
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
                processors.append(NoRepeatNGramLogitsProcessorCPU(generation_config.no_repeat_ngram_size, kwargs["number_of_threads"]))
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

        self.logits_processor = processors

    def init_beam_search_scorer(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        if self.resident_device == PARTITION_RESIDENT_DEVICES.BASELINE:
            self.beam_scorer = BeamSearchScorer(
                batch_size,
                num_beams,
                device,
                length_penalty,
                do_early_stopping,
                num_beam_hyps_to_keep,
                num_beam_groups,
                max_length
            )
        elif self.resident_device == PARTITION_RESIDENT_DEVICES.GPU:
            self.beam_scorer = BeamSearchScorerGPU(
                batch_size,
                num_beams,
                device,
                length_penalty,
                do_early_stopping,
                num_beam_hyps_to_keep,
                num_beam_groups,
                max_length
            )
        elif self.resident_device == PARTITION_RESIDENT_DEVICES.CPU:
            self.beam_scorer = BeamSearchScorerCPU(
                batch_size,
                num_beams,
                device,
                length_penalty,
                do_early_stopping,
                num_beam_hyps_to_keep,
                num_beam_groups,
                max_length
            )


    def add_decoder_input_ids(self, input_ids):
        self.decoder_input_ids = input_ids


class WorkloadPartitioner:
    def __init__(self, partition_type: PARTITION_TYPES) -> None:
        self.joined = False
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        

    def get_sequence_outputs(self):
        assert self.joined, f"Did not join both the cpu and gpu versions, self.joined must be True, but found False instead..."
        return self.sequence_outputs
    
    @measure_times
    def join(self):
        self.joined = True
        # Construct sequences
        sequences = torch.zeros(self.cpu_bundle.batch_size + self.gpu_bundle.batch_size, max(self.cpu_bundle.sequence_outputs["sequences"].shape[1], self.gpu_bundle.sequence_outputs["sequences"].shape[1]), dtype=self.cpu_bundle.sequence_outputs["sequences"].dtype)
        sequence_scores = torch.zeros(self.cpu_bundle.batch_size + self.gpu_bundle.batch_size)
        # self.gpu_bundle.sequence_outputs["sequences"] = self.gpu_bundle.sequence_outputs["sequences"].to("cpu")
        for idx, gpu_idx in enumerate(self.mapping_function["gpu"]):
            sequences[gpu_idx, :self.gpu_bundle.sequence_outputs["sequences"].shape[1]] = self.gpu_bundle.sequence_outputs["sequences"][idx]
            sequence_scores[gpu_idx] = self.gpu_bundle.sequence_outputs["sequence_scores"][idx]
        for idx, cpu_idx in enumerate(self.mapping_function["cpu"]):
            sequences[cpu_idx, :self.cpu_bundle.sequence_outputs["sequences"].shape[1]] = self.cpu_bundle.sequence_outputs["sequences"][idx]
            sequence_scores[cpu_idx] = self.cpu_bundle.sequence_outputs["sequence_scores"][idx]
        # sequence_scores[self.mapping_function["gpu"]] = self.gpu_bundle.sequence_outputs["sequence_scores"].to("cpu")
        # sequence_scores[self.mapping_function["cpu"]] = self.cpu_bundle.sequence_outputs["sequence_scores"]
        
        self.sequence_outputs = {
            "sequences": sequences,
            "sequence_scores": sequence_scores
        }

    @measure_times
    def partition_workload_pre_encoder(self, inputs_tensor: torch.LongTensor, attention_mask: torch.tensor, partition_type: PARTITION_TYPES, cpu_size: int):
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        if partition_type == PARTITION_TYPES.CPU_GPU:
            indices = [i for i in range(inputs_tensor.shape[0])]
            cpu_indices = random.sample(indices, k=cpu_size)
            gpu_indices = list(filter(lambda x: x not in cpu_indices, indices))
            self.mapping_function = {"cpu": cpu_indices, "gpu": gpu_indices}
            inputs_tensor_cpu = inputs_tensor[cpu_indices].to("cpu")
            inputs_tensor_gpu = inputs_tensor[gpu_indices]
            attention_mask_cpu = attention_mask[cpu_indices].to("cpu")
            attention_mask_gpu = attention_mask[gpu_indices]
            self.gpu_bundle.encoder_inputs_tensor = inputs_tensor_gpu
            self.gpu_bundle.encoder_attention_mask = attention_mask_gpu
            self.cpu_bundle.encoder_inputs_tensor = inputs_tensor_cpu
            self.cpu_bundle.encoder_attention_mask = attention_mask_cpu
            self.gpu_bundle.batch_size = inputs_tensor_gpu.shape[0]
            self.cpu_bundle.batch_size = inputs_tensor_cpu.shape[0]    

        elif partition_type == PARTITION_TYPES.GPU:
            self.gpu_bundle.encoder_inputs_tensor = inputs_tensor
            self.gpu_bundle.encoder_attention_mask = attention_mask
            self.gpu_bundle.batch_size = inputs_tensor.shape[0]
        elif partition_type == PARTITION_TYPES.BASELINE:
            self.bundle.encoder_inputs_tensor = inputs_tensor
            self.bundle.encoder_attention_mask = attention_mask
            self.bundle.batch_size = inputs_tensor.shape[0]

    # @measure_times
    # def partition_workload_pre_decoder(self, input_ids: torch.LongTensor, partition_type: PARTITION_TYPES, cpu_size: int, cur_len: int):
    #     assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
    #     if cur_len == 1:
    #         if partition_type == PARTITION_TYPES.CPU_GPU:
    #             t_indices = [i for i in range(self.gpu_bundle.batch_size + self.cpu_bundle.batch_size)]
    #             t_cpu_indices = random.sample(t_indices, k=cpu_size)
    #             t_gpu_indices = list(filter(lambda x: x not in t_cpu_indices, t_indices))
    #             # t_cpu_indices.sort()
    #             tensor_staging_area = {}
    #             if len(t_cpu_indices) > len(self.mapping_function["cpu"]):
    #                 zeros_tensor_last_hidden_state = torch.zeros([len(t_cpu_indices) - len(self.mapping_function["cpu"]), *list(self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"].shape[1:])], device="cpu")
    #                 self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.cat([self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], zeros_tensor_last_hidden_state], dim=0)
    #                 zeros_tensor_attention_mask = torch.zeros([len(t_cpu_indices) - len(self.mapping_function["cpu"]), *list(self.cpu_bundle.model_kwargs["attention_mask"].shape[1:])], device="cpu")
    #                 self.cpu_bundle.model_kwargs["attention_mask"] = torch.cat([self.cpu_bundle.model_kwargs["attention_mask"], zeros_tensor_attention_mask], dim=0)
                    
    #             if len(t_gpu_indices) > len(self.mapping_function["gpu"]):
    #                 zeros_tensor_last_hidden_state = torch.zeros([len(t_gpu_indices) - len(self.mapping_function["gpu"]), *list(self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"].shape[1:])], device="cuda")
    #                 self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.cat([self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], zeros_tensor_last_hidden_state], dim=0)
    #                 zeros_tensor_attention_mask = torch.zeros([len(t_gpu_indices) - len(self.mapping_function["gpu"]), *list(self.gpu_bundle.model_kwargs["attention_mask"].shape[1:])], device="cuda")
    #                 self.gpu_bundle.model_kwargs["attention_mask"] = torch.cat([self.gpu_bundle.model_kwargs["attention_mask"], zeros_tensor_attention_mask], dim=0)
                    
    #             for cpu_insert_ind, cpu_ind in enumerate(t_cpu_indices):
    #                 if cpu_ind in self.mapping_function["gpu"]:
    #                     # Perform a D2H transfer
    #                     gpu_ind = self.mapping_function["gpu"].index(cpu_ind)
    #                     # encoder_inputs_tensor_slice = self.gpu_bundle.encoder_inputs_tensor[gpu_ind].clone().detach().to("cpu")
    #                     # encoder_attention_mask_slice = self.gpu_bundle.encoder_attention_mask[gpu_ind].clone().detach().to("cpu")
    #                     encoder_last_hidden_state = self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][gpu_ind].clone().detach().to("cpu")
    #                     encoder_attention_mask = self.gpu_bundle.model_kwargs["attention_mask"][gpu_ind].clone().detach().to("cpu")
    #                     tensor_staging_area[cpu_ind] = {
    #                         "cpu_insert_ind": cpu_insert_ind,
    #                         "encoder_last_hidden_state": encoder_last_hidden_state,
    #                         "encoder_attention_mask": encoder_attention_mask,
    #                         "type": 0
    #                     }
    #             for gpu_insert_ind, gpu_ind in enumerate(t_gpu_indices):
    #                 if gpu_ind in self.mapping_function["cpu"]:
    #                     # Perform a H2D transfer
    #                     cpu_ind = self.mapping_function["cpu"].index(gpu_ind)
    #                     # encoder_inputs_tensor_slice = self.cpu_bundle.encoder_inputs_tensor[cpu_ind].clone().detach().to("cuda")
    #                     # encoder_attention_mask_slice = self.cpu_bundle.encoder_attention_mask[cpu_ind].clone().detach().to("cuda")
    #                     encoder_last_hidden_state = self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][cpu_ind].clone().detach().to("cuda")
    #                     encoder_attention_mask = self.cpu_bundle.model_kwargs["attention_mask"][cpu_ind].clone().detach().to("cuda")
    #                     tensor_staging_area[gpu_ind] = {
    #                         "gpu_insert_ind": gpu_insert_ind,
    #                         "encoder_last_hidden_state": encoder_last_hidden_state,
    #                         "encoder_attention_mask": encoder_attention_mask,
    #                         "type": 1
    #                     }
                
    #             for key, value in tensor_staging_area.items():
    #                 if value["type"] == 1:
    #                     # self.gpu_bundle.encoder_inputs_tensor[value["gpu_insert_ind"]] = value["encoder_inputs_tensor_slice"]
    #                     # self.gpu_bundle.encoder_attention_mask[value["gpu_insert_ind"]] = value["encoder_attention_mask_slice"]
    #                     self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][value["gpu_insert_ind"]] = value["encoder_last_hidden_state"]
    #                     self.gpu_bundle.model_kwargs["attention_mask"][value["gpu_insert_ind"]] = value["encoder_attention_mask"]
    #                 elif value["type"] == 0:
    #                     # self.cpu_bundle.encoder_inputs_tensor[value["cpu_insert_ind"]] = value["encoder_inputs_tensor_slice"]
    #                     # self.cpu_bundle.encoder_attention_mask[value["cpu_insert_ind"]] = value["encoder_attention_mask_slice"]
    #                     self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][value["cpu_insert_ind"]] = value["encoder_last_hidden_state"]
    #                     self.cpu_bundle.model_kwargs["attention_mask"][value["cpu_insert_ind"]] = value["encoder_attention_mask"]
                
    #             if len(t_cpu_indices) < len(self.mapping_function["cpu"]):
    #                 self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.index_select(self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], 0, torch.arange( self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"].size(0) - (len(self.mapping_function["cpu"]) - len(t_cpu_indices))))
    #                 self.cpu_bundle.model_kwargs["attention_mask"] = torch.index_select(self.cpu_bundle.model_kwargs["attention_mask"], 0, torch.arange( self.cpu_bundle.model_kwargs["attention_mask"].size(0) - (len(self.mapping_function["cpu"]) - len(t_cpu_indices))))
    #             if len(t_gpu_indices) < len(self.mapping_function["gpu"]):
    #                 self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.index_select(self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], 0, torch.arange( self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"].size(0) - (len(self.mapping_function["gpu"]) - len(t_gpu_indices))))
    #                 self.gpu_bundle.model_kwargs["attention_mask"] = torch.index_select(self.gpu_bundle.model_kwargs["attention_mask"], 0, torch.arange( self.gpu_bundle.model_kwargs["attention_mask"].size(0) - (len(self.mapping_function["gpu"]) - len(t_gpu_indices))))


    #             self.mapping_function = {"cpu": t_cpu_indices, "gpu": t_gpu_indices}
    #             input_ids_cpu = input_ids[self.mapping_function["cpu"]].to("cpu")
    #             input_ids_gpu = input_ids[self.mapping_function["gpu"]]
    #             self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
    #             self.cpu_bundle.add_decoder_input_ids(input_ids_cpu)
    #             self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
    #             self.cpu_bundle.batch_size = input_ids_cpu.shape[0]

    #         elif partition_type == PARTITION_TYPES.GPU:
    #             self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
    #             self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
    #         elif partition_type == PARTITION_TYPES.BASELINE:
    #             self.bundle.add_decoder_input_ids(input_ids)
    #             self.bundle.batch_size = input_ids.shape[0]
    #     elif cur_len == 2:
    #         if partition_type == PARTITION_TYPES.CPU_GPU:
    #             indices = [i for i in range(input_ids.shape[0])]
    #             cpu_indices = random.sample(indices, k=cpu_size)
    #             gpu_indices = list(filter(lambda x: x not in cpu_indices, indices))
    #             self.mapping_function = {"cpu": cpu_indices, "gpu": gpu_indices}
    #             input_ids_cpu = input_ids[self.mapping_function["cpu"]].to("cpu")
    #             input_ids_gpu = input_ids[self.mapping_function["gpu"]]
    #             self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
    #             self.cpu_bundle.add_decoder_input_ids(input_ids_cpu)
    #             self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
    #             self.cpu_bundle.batch_size = input_ids_cpu.shape[0]
    #         elif partition_type == PARTITION_TYPES.GPU:
    #             self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
    #             self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
    #         elif partition_type == PARTITION_TYPES.BASELINE:
    #             self.bundle.add_decoder_input_ids(input_ids)
    #             self.bundle.batch_size = input_ids.shape[0]
    #     else:
    #         # Nothing to do
    #         pass 
    
    @measure_times
    def partition_workload_pre_decoder_second(self, partition_type: PARTITION_TYPES, cpu_size: int):
        t_cpu_indices = random.sample(self.mapping_function["cpu"], k=cpu_size)
        c_cpu_indices_one = []
        c_cpu_indices_two = []
        t_cpu_indices_one = []
        for cpu_ind_sel, cpu_ind in enumerate(self.mapping_function["cpu"]):
            if cpu_ind not in t_cpu_indices:
                c_cpu_indices_one.append(cpu_ind_sel)
                c_cpu_indices_two.append(cpu_ind)
            else:
                t_cpu_indices_one.append(cpu_ind_sel)
        for cpu_ind_sel, cpu_ind in zip(c_cpu_indices_one, c_cpu_indices_two):
            # make a H2D transfer
            # encoder_last_hidden_state = self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][cpu_ind_sel].clone().detach().to("cuda")
            num_beams = self.gpu_bundle.decoder_input_ids.shape[0] // self.gpu_bundle.batch_size
            cpu_indices_sel = [cpu_ind_sel * num_beams + id for id in range(num_beams)]
            encoder_attention_mask = self.cpu_bundle.model_kwargs["attention_mask"][cpu_indices_sel].clone().detach().to("cuda")
            # self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.cat([self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], torch.unsqueeze(encoder_last_hidden_state, 0)], dim=0)
            self.gpu_bundle.model_kwargs["attention_mask"] = torch.cat([self.gpu_bundle.model_kwargs["attention_mask"], encoder_attention_mask], dim=0)
            for pk_idx in range(len(self.gpu_bundle.model_kwargs["past_key_values"])):
                self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][0] = torch.cat([self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][0], self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][0][cpu_indices_sel, :, :].clone().detach().to("cuda")], dim=0)
                self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][1] = torch.cat([self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][1], self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][1][cpu_indices_sel, :, :].clone().detach().to("cuda")], dim=0)
                self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][2] = torch.cat([self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][2], self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][2][cpu_indices_sel, :, :].clone().detach().to("cuda")], dim=0)
                self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][3] = torch.cat([self.gpu_bundle.model_kwargs["past_key_values"][pk_idx][3], self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][3][cpu_indices_sel, :, :].clone().detach().to("cuda")], dim=0)
            self.gpu_bundle.beam_scores = torch.cat([self.gpu_bundle.beam_scores, self.cpu_bundle.beam_scores[cpu_indices_sel].clone().detach().to("cuda")], dim=0)
            self.gpu_bundle.decoder_input_ids = torch.cat([self.gpu_bundle.decoder_input_ids, self.cpu_bundle.decoder_input_ids[cpu_indices_sel].clone().detach().to("cuda")], dim=0)
            # move beam_scorer state
            beam_hypotheses_tensor = torch.zeros((1, self.gpu_bundle.beam_hypotheses.shape[1], self.gpu_bundle.beam_hypotheses.shape[2]), device="cuda")
            beam_hypotheses_meta_tensor = torch.zeros((1, 3), device="cuda")
            for beam_idx in range(num_beams):
                if beam_idx >= len(self.cpu_bundle.beam_scorer._beam_hyps[cpu_ind_sel].beams):
                    break
                else:
                    beam_hypotheses_tensor[0][beam_idx][0] = self.cpu_bundle.beam_scorer._beam_hyps[cpu_ind_sel].beams[beam_idx][1].shape[0]
                    beam_hypotheses_tensor[0][beam_idx][1] = self.cpu_bundle.beam_scorer._beam_hyps[cpu_ind_sel].beams[beam_idx][0]
                    beam_hypotheses_tensor[0][beam_idx][2] = -1
                    beam_hypotheses_tensor[0][beam_idx][3] = -1        
            self.gpu_bundle.beam_hypotheses = torch.cat([self.gpu_bundle.beam_hypotheses, beam_hypotheses_tensor], dim=0)
            self.gpu_bundle.beam_hypotheses_meta = torch.cat([self.gpu_bundle.beam_hypotheses_meta, beam_hypotheses_meta_tensor], dim=0)
        t_c_cpu_indices_one = c_cpu_indices_one.copy()
        t_c_cpu_indices_one.sort()
        t_c_cpu_indices_one.reverse()
        for cpu_ind_sel in t_c_cpu_indices_one:
            del self.cpu_bundle.beam_scorer._beam_hyps[cpu_ind_sel]
            
             
        t_t_cpu_indices_one = torch.tensor(t_cpu_indices_one)
        # self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.index_select(self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], 0, t_t_cpu_indices_one)
        
        
        batched_t_cpu_indices_one = []
        for t_cpu_idx in t_cpu_indices_one:
            batched_t_cpu_indices_one.extend([t_cpu_idx * num_beams + idx for idx in range(num_beams)])
        t_batched_t_cpu_indices_one = torch.tensor(batched_t_cpu_indices_one)
        for pk_idx in range(len(self.gpu_bundle.model_kwargs["past_key_values"])):
            self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][0] = torch.index_select(self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][0], 0, t_batched_t_cpu_indices_one)
            self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][1] = torch.index_select(self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][1], 0, t_batched_t_cpu_indices_one)
            self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][2] = torch.index_select(self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][2], 0, t_batched_t_cpu_indices_one)
            self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][3] = torch.index_select(self.cpu_bundle.model_kwargs["past_key_values"][pk_idx][3], 0, t_batched_t_cpu_indices_one) 
        
        self.cpu_bundle.decoder_input_ids = torch.index_select(self.cpu_bundle.decoder_input_ids, 0, t_batched_t_cpu_indices_one)
        self.cpu_bundle.beam_scores = torch.index_select(self.cpu_bundle.beam_scores, 0, t_batched_t_cpu_indices_one)
        self.cpu_bundle.model_kwargs["attention_mask"] = torch.index_select(self.cpu_bundle.model_kwargs["attention_mask"], 0, t_batched_t_cpu_indices_one) 
        self.mapping_function = {"cpu": t_cpu_indices, "gpu": self.mapping_function["gpu"] + c_cpu_indices_two}
        self.gpu_bundle.batch_size = self.gpu_bundle.decoder_input_ids.shape[0] // num_beams
        self.cpu_bundle.batch_size = self.cpu_bundle.decoder_input_ids.shape[0] // num_beams
        

        
        # input_ids_cpu = input_ids[self.mapping_function["cpu"]].to("cpu")
        # input_ids_gpu = input_ids[self.mapping_function["gpu"]]
        # self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
        # self.cpu_bundle.add_decoder_input_ids(input_ids_cpu)
        # self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
        # self.cpu_bundle.batch_size = input_ids_cpu.shape[0]
    
    @measure_times
    def partition_workload_pre_decoder_first(self, input_ids: torch.LongTensor, partition_type: PARTITION_TYPES, cpu_size: int):
        assert partition_type in [PARTITION_TYPES.GPU, PARTITION_TYPES.CPU_GPU, PARTITION_TYPES.BASELINE], f"Only CPU_GPU, GPU and BASELINE partition types supported, but found {partition_type}"
        if partition_type == PARTITION_TYPES.CPU_GPU:
                t_cpu_indices = random.sample(self.mapping_function["cpu"], k=cpu_size)
                c_cpu_indices_one = []
                c_cpu_indices_two = []
                t_cpu_indices_one = []
                for cpu_ind_sel, cpu_ind in enumerate(self.mapping_function["cpu"]):
                    if cpu_ind not in t_cpu_indices:
                        c_cpu_indices_one.append(cpu_ind_sel)
                        c_cpu_indices_two.append(cpu_ind)
                    else:
                        t_cpu_indices_one.append(cpu_ind_sel)
                for cpu_ind_sel, cpu_ind in zip(c_cpu_indices_one, c_cpu_indices_two):
                    # make a H2D transfer
                    encoder_last_hidden_state = self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"][cpu_ind_sel].clone().detach().to("cuda")
                    encoder_attention_mask = self.cpu_bundle.model_kwargs["attention_mask"][cpu_ind_sel].clone().detach().to("cuda")
                    self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.cat([self.gpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], torch.unsqueeze(encoder_last_hidden_state, 0)], dim=0)
                    self.gpu_bundle.model_kwargs["attention_mask"] = torch.cat([self.gpu_bundle.model_kwargs["attention_mask"], torch.unsqueeze(encoder_attention_mask, 0)], dim=0)
                t_t_cpu_indices_one = torch.tensor(t_cpu_indices_one)
                self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"] = torch.index_select(self.cpu_bundle.model_kwargs["encoder_outputs"]["last_hidden_state"], 0, t_t_cpu_indices_one)
                self.cpu_bundle.model_kwargs["attention_mask"] = torch.index_select(self.cpu_bundle.model_kwargs["attention_mask"], 0, t_t_cpu_indices_one) 
                self.mapping_function = {"cpu": t_cpu_indices, "gpu": self.mapping_function["gpu"] + c_cpu_indices_two}
                input_ids_cpu = input_ids[self.mapping_function["cpu"]].to("cpu")
                input_ids_gpu = input_ids[self.mapping_function["gpu"]]
                self.gpu_bundle.add_decoder_input_ids(input_ids_gpu)
                self.cpu_bundle.add_decoder_input_ids(input_ids_cpu)
                self.gpu_bundle.batch_size = input_ids_gpu.shape[0]
                self.cpu_bundle.batch_size = input_ids_cpu.shape[0]



    @measure_times
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
    
