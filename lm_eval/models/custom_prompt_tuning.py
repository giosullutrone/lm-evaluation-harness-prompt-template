import os
from typing import Optional, Union, Literal
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
import transformers

from peft.peft_model import PeftModelForCausalLM
from peft.utils import _get_batch_size, PeftType
import torch
import warnings
from typing import Union, List
from functools import partial
from transformers import PreTrainedTokenizer
import tqdm
from lm_eval.utils import (
    eval_logger
)

def get_template_token_position(x: torch.Tensor, token_ids: torch.Tensor) -> Union[int, None]:
    token_ids_start_idx = None
    for idx in torch.where(x == token_ids[0])[0]:
        # Here we are just making sure that the token IDs match
        if (idx + len(token_ids) < len(x)) and (all(token_ids == x[idx : idx + len(token_ids)])):
            token_ids_start_idx = idx
    if token_ids_start_idx is None: return
    return int(token_ids_start_idx + len(token_ids))


def fit_template_tokens(sample, system_token_ids):
    token_ids_start_idx = None
    idx = 0
    while idx < len(system_token_ids):
        token_ids_start_idx = get_template_token_position(sample, system_token_ids[idx:])
        if token_ids_start_idx is not None: break
        idx += 1
    else: raise Exception("Could not find fitting template tokens")
    eval_logger.info(f"System template in tokens: {idx}")
    return system_token_ids[idx:]


class ProxySystemPromptTuning:
    @staticmethod
    def add_proxy(model: PeftModelForCausalLM, system_template: Union[str, List[int]], tokenizer: PreTrainedTokenizer=None):
        # --------------------------------------------------------------------------
        if isinstance(system_template, str):
            assert tokenizer is not None, "If the template is a string, a tokenizer must be provided"
            # The user provides a string, must tokenize
            system_token_ids = tokenizer.encode(system_template, add_special_tokens=False, return_tensors="pt")[0].to(model.base_model.device)
        else:
            # The user already provides the token ids
            system_token_ids = system_template.to(model.base_model.device)
        # --------------------------------------------------------------------------
        model.system_token_ids = None
        model.system_token_ids_start = system_token_ids
        model.system_token_ids_start_idx = None
        model.forward = partial(ProxySystemPromptTuning.forward, model)

    @staticmethod
    def forward(self,
                input_ids=None, 
                attention_mask=None, 
                inputs_embeds=None, 
                labels=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None, 
                task_ids=None, 
                **kwargs):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning or peft_config.peft_type == PeftType.PREFIX_TUNING:
            return super(PeftModelForCausalLM, self).forward(input_ids, attention_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict, task_ids, **kwargs)

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        # We want to append the learned tokens to the input prompt at a specific location defined
        # by a sequence of tokens (for example in llama2 it may be after '[INST]')
        # Note: We keep it fixed for all prompts.
        if self.system_token_ids_start_idx is None:
            if self.system_token_ids is None: self.system_token_ids = fit_template_tokens(input_ids[0], self.system_token_ids_start)
            self.system_token_ids_start_idx = get_template_token_position(input_ids[0], self.system_token_ids)
            assert self.system_token_ids_start_idx is not None, "Could not find the system token"
        concatenated_embeds = torch.cat((inputs_embeds[:, :self.system_token_ids_start_idx], prompts, inputs_embeds[:, self.system_token_ids_start_idx:]), dim=1)
        inputs_embeds = concatenated_embeds # torch.cat((prompts, inputs_embeds), dim=1)
        # concat prompt labels
        if labels is not None:
            prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            # Here we subtract 1 since the labels are for example "labels = input[1:]" so everyting is
            # shifted by one
            concatenated_labels = torch.cat((labels[:, :self.system_token_ids_start_idx-1], prefix_labels, labels[:, self.system_token_ids_start_idx-1:]), dim=1)
            kwargs["labels"] = concatenated_labels # torch.cat((prefix_labels, labels), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


@register_model("custom", "custom_prompt_tuning")
class CustomPromptTuning(HFLM):
    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        backend: Optional[
            Literal["default", "causal", "seq2seq"]
        ] = "default",  # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT and quantization options
        peft: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        system_template: str=" <<SYS>>\n ",
        **kwargs
    ) -> None:
        super().__init__(pretrained, 
                         backend, 
                         revision, 
                         subfolder, 
                         tokenizer, 
                         truncation, 
                         max_length, 
                         device, dtype, 
                         batch_size, 
                         max_batch_size, 
                         trust_remote_code, 
                         use_fast_tokenizer, 
                         parallelize, 
                         device_map_option, 
                         max_memory_per_gpu, 
                         max_cpu_memory, 
                         offload_folder, 
                         peft, 
                         autogptq, 
                         **kwargs)
        ProxySystemPromptTuning.add_proxy(self._model, system_template, self.tokenizer)