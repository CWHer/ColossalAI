from typing import List

import torch
import torch.nn.functional as F
from coati.models.generation import generate

from .base import Experience, ExperienceMaker


class MultiStepExperienceMaker(ExperienceMaker):
    """
    Multi Step experience maker.
    NOTE: Multi Step experience maker treats each token as a single step in MDP.
    """
    @torch.no_grad()
    def make_experience(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        **generate_kwargs
                        ) -> List[Experience]:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        sequences = generate(self.actor,
                             input_ids,
                             attention_mask=attention_mask,
                             **generate_kwargs)

        input_len = input_ids.size(1)
        eos_token_id = generate_kwargs.get("eos_token_id", None)
        action_mask = torch.ones_like(sequences[:, input_len:], dtype=torch.bool)
        if eos_token_id is not None:
            # action is |action|eos|pad|, and action_mask is |1|1|0|
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1, -1), value=True)  # shift right by 1, include eos token
        attention_mask = torch.cat([attention_mask, action_mask], dim=-1)

        # compute action log probs
        action_logits = self.actor(sequences, attention_mask)["logits"]
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        base_action_logits = self.initial_model(sequences, attention_mask)["logits"]
        base_log_probs = F.log_softmax(base_action_logits, dim=-1)

        # TODO
        # compute kl_div
        kl_list = compute_approx_kl(action_log_probs, base_log_probs)
        # clip kl
        kl_list = torch.clamp(kl_list, max=10, min=1e-4)

        # add eos token to the end of sequence and compute reward
        eos_tensor = torch.tensor([self.eos_token_id], device=input_ids.device).repeat(input_ids.size(0), 1)
        sequence_with_eos = torch.cat([sequences, eos_tensor], dim=-1)
        rewards = self.reward_model(sequence_with_eos)

        # reward clip
        rewards = torch.clamp(rewards, max=10, min=-10)

        # running mean reward
        for i in range(rewards.size(0)):
            value = rewards[i]
            self.reward_count += 1
            delta = value - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = value - self.reward_mean
            self.reward_M2 += delta * delta2

        std = self.reward_M2 / (self.reward_count - 1)
        rewards = (rewards - self.reward_mean) / std

        print("rewards: ", rewards)
        rewards = rewards * (1 - self.kl_coef)

        # get action mask
        action_mask = action_mask[:, in_len:]
        kl_list = kl_list[:, in_len:]

        # compute the advantages
        advantages, returns = self.compute_gae(kl_list, rewards, values, action_mask)

        for i in range(in_len, sequences.size(1) - 1):
            for j in range(sequences.size(0)):
                if sequences[j, i] != self.eos_token_id:
                    _state = sequences[j, :i]
                    _action_log_prob = action_log_probs[j, i]
                    _value = values[j, i - in_len]
                    _return = returns[j, i - in_len]
                    _adv = advantages[j, i - in_len]
                    _attention_mask = attention_mask[j, :i]
                    _action_mask = action_mask[j, :i - in_len]
                    exp = Experience(_state, _action_log_prob, _value, _return, _adv, _attention_mask, _action_mask)
                    self.buffer.append(exp)
        buffer = self.buffer
        return buffer

    @torch.no_grad()
    def compute_gae(self, kl_list: torch.Tensor,
                    reward: torch.Tensor,
                    values: torch.Tensor,
                    action_mask: torch.Tensor) -> torch.Tensor:
        kl = -kl_list * action_mask * self.kl_coef
        values = values * action_mask
        T = torch.sum(values.ne(0), dim=1)
        self.total_len = sum(T)
        max_len = max(T)
        gae_values = torch.zeros_like(values)
        delta_list = torch.zeros_like(values)

        # add reward to kl[:, -1]
        for i in range(len(T)):
            kl[i, T[i] - 1] += reward[i]

        # compute delta
        for t in range(max_len - 1):
            next_v = values[:, t + 1] if t + 1 < max_len else 0
            delta_list[:, t] = kl[:, t] + self.gamma * next_v - values[:, t]

        # compute gae
        gae_values[:, max_len - 1] = delta_list[:, max_len - 1]
        for t in range(max_len - 2, -1, -1):
            gae_values[:, t] = delta_list[:, t] + self.gamma * self.lamda * gae_values[:, t + 1]

        # compute return
        returns = gae_values + values
        return gae_values, returns


@torch.no_grad()
def generate_with_value(actor: nn.Module,
                        critic: nn.Module,
                        input_ids: torch.Tensor,
                        max_length: int,
                        early_stopping: bool = True,
                        eos_token_id: Optional[int] = None,
                        pad_token_id: Optional[int] = None,
                        top_k: Optional[int] = None,
                        top_p: Optional[float] = None,
                        temperature: Optional[float] = None,
                        prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
                        update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
                        **model_kwargs) -> torch.Tensor:
    if input_ids.size(1) >= max_length:
        return input_ids

    temperature = 1.0
    logits_processor = prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    values = []

    for _ in range(input_ids.size(1), max_length):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
            "input_ids": input_ids
        }
        outputs = actor(**model_inputs)

        next_token_logits = outputs["logits"][:, -1, :]
        # pre-process distribution
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        # if "nan" in str(probs):
        #     for name, param in actor.named_parameters():
        #         print(name, param)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # compute value on the last hidden_state
        eos_tensor = torch.tensor([eos_token_id], device=input_ids.device).repeat(input_ids.size(0), 1)
        value_input = torch.cat([input_ids, eos_tensor], dim=-1)
        value = critic(value_input)
        values.append(value)

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished if early_stopping=True
        if early_stopping and _is_sequence_finished(unfinished_sequences):
            break
    # transform values to tensor
    values = torch.cat(values, dim=0)
    # reshape to (x,4)
    values = values.view(4, -1)
    return input_ids, values


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    # if action_mask is not None:
    #     approx_kl = masked_mean(approx_kl, action_mask, dim=1)
    #     return approx_kl
    approx_kl = approx_kl.sum(dim=-1)
    return approx_kl
