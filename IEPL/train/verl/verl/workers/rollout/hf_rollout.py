import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from .base import BaseRollout

from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


__all__ = ['HFRollout']


def _unwrap(m: nn.Module) -> nn.Module:  # >>> NEW: Compatible with DDP/FSDP wrapping
    return m.module if hasattr(m, "module") else m


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config,
                 action_head: nn.Module, proprio_projector: nn.Module,
                 noisy_action_projector: nn.Module, sigma_net: nn.Module,
                 context_projector: nn.Module, tokenizer=None):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer  # Passed separately; OmegaConf does not support non-primitive types
        # Token Generation Mode: all these are None
        self.action_head = _unwrap(action_head) if action_head is not None else None
        self.proprio_projector = _unwrap(proprio_projector) if proprio_projector is not None else None
        self.noisy_action_projector = _unwrap(noisy_action_projector) if noisy_action_projector is not None else None
        self.sigma_net = _unwrap(sigma_net) if sigma_net is not None else None
        self.context_projector = _unwrap(context_projector) if context_projector is not None else None


    def generate_actions(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def generate_sequences(self, prompts):
        raise NotImplementedError(
            "HFRollout does not support generate_sequences. Use generate_actions instead."
        )
    
    def set_to_eval(self):
        self.module.eval()
        if self.action_head is not None:
            self.action_head.eval()
        if self.proprio_projector is not None:
            self.proprio_projector.eval()
        if self.noisy_action_projector is not None:
            self.noisy_action_projector.eval()
        if self.sigma_net is not None:
            self.sigma_net.eval()
        if self.context_projector is not None:
            self.context_projector.eval()

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        """
        Token Generation Mode: Autoregressive action token sampling.
        Uses model forward with pixel_values, samples action tokens, computes log_probs.
        """
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from verl.utils.torch_functional import logprobs_from_logits

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        pixels = prompts.batch['pixels']
        proprio = prompts.batch.get('proprio')

        B = idx.size(0)
        device = idx.device
        prompt_len = idx.size(1)
        max_new_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM

        self.set_to_eval()

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        logits_lst = []
        with param_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                curr_idx = idx
                curr_attn = attention_mask
                for _ in range(max_new_tokens):
                    out = self.module(
                        input_ids=curr_idx,
                        attention_mask=curr_attn,
                        pixel_values=pixels,
                        labels=None,
                        output_hidden_states=False,
                        proprio=proprio,
                        proprio_projector=None,
                        use_film=False,
                    )
                    logits = out.logits[:, -1, :].float()
                    probs = torch.nn.functional.softmax(logits / 1.0, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    logits_lst.append(logits)
                    curr_idx = torch.cat([curr_idx, idx_next], dim=1)
                    curr_attn = torch.cat([
                        curr_attn,
                        torch.ones((B, 1), dtype=curr_attn.dtype, device=device)
                    ], dim=1)

        response_tokens = curr_idx[:, prompt_len:]
        logits_stack = torch.stack(logits_lst, dim=1)
        log_probs = logprobs_from_logits(logits=logits_stack, labels=response_tokens)
        traj_log_prob = log_probs.sum(dim=1)

        bins = 256
        min_action = -1.0
        max_action = 1.0
        if hasattr(self.config, 'action_bins'):
            bins = int(self.config.action_bins)
        if hasattr(self.config, 'action_min'):
            min_action = float(self.config.action_min)
        if hasattr(self.config, 'action_max'):
            max_action = float(self.config.action_max)
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = getattr(self.config, 'tokenizer', None)
        if tokenizer is None:
            tokenizer = getattr(self.module, 'tokenizer', None) or (
                self.module.llm_backbone.tokenizer if hasattr(self.module, 'llm_backbone') else None
            )
        if tokenizer is None and hasattr(self.module, 'processor'):
            tokenizer = getattr(self.module.processor, 'tokenizer', None)
        if tokenizer is None:
            from transformers import AutoTokenizer
            ckpt = getattr(self.module.config, '_name_or_path', None) or getattr(self.module.config, 'name_or_path', '')
            if not ckpt and hasattr(self.module, 'config'):
                ckpt = getattr(self.module.config, '_name_or_path', '')
            tokenizer = AutoTokenizer.from_pretrained(ckpt or 'meta-llama/Llama-2-7b-hf', trust_remote_code=True)
        act_tok = ActionTokenizer(tokenizer, bins=bins, min_action=min_action, max_action=max_action)
        decoded = act_tok.decode_token_ids_to_actions(response_tokens.cpu().numpy())
        predicted_actions = torch.tensor(decoded, dtype=torch.float32, device=device).reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM)

        labels = prompts.batch.get('labels', idx)
        proprio_out = proprio if proprio is not None else torch.zeros(B, 8, device=device, dtype=pixels.dtype)

        batch = TensorDict(
            {
                "predicted_actions": predicted_actions,
                "input_ids": idx,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixels": pixels,
                "proprio": proprio_out,
                "old_log_probs": traj_log_prob,
                "response_tokens": response_tokens,
            },
            batch_size=B
        )

        torch.cuda.empty_cache()
        return DataProto(batch=batch)
