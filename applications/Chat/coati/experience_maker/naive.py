import torch
import torch.nn.functional as F
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    NOTE: Naive experience maker treats the whole sequence as a single step in MDP.
    """

    @torch.no_grad()
    def make_experience(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        **generate_kwargs
                        ) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate sequences
        sequences = generate(self.actor,
                             input_ids,
                             attention_mask=attention_mask,
                             **generate_kwargs)

        # calculate auxiliary tensors
        input_len = input_ids.size(1)
        num_actions = sequences.size(1) - input_len

        eos_token_id = generate_kwargs.get("eos_token_id", None)
        action_mask = torch.ones_like(sequences[:, input_len:], dtype=torch.bool)
        if eos_token_id is not None:
            # action is |action|eos|pad|, and action_mask is |1|1|0|
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1, -1), value=True)  # shift right by 1, include eos token
        attention_mask = torch.cat([attention_mask, action_mask], dim=-1)

        actor_logits = self.actor(sequences, attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_logits, sequences, num_actions)
        base_model_logits = self.initial_model(sequences, attention_mask)["logits"]
        base_action_log_probs = calc_action_log_probs(base_model_logits, sequences, num_actions)
        value = self.critic(sequences, attention_mask)
        r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
