class MPPOTrainer(Trainer):
    def training_step(self, experience: Experience) -> Dict[str, float]:

        num_actions = experience.action_mask.size(1)
        action_logits = self.actor.model(experience.sequences)['logits'][:, -1]
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages)
        # ptx loss
        if self.ptx_coef != 0:
            batch = next(self.pretrain_dataloader)
            ptx = batch['input_ids'].to(torch.cuda.current_device())
            label = batch['labels'].to(torch.cuda.current_device())[:, 1:]
            attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
            ptx_log_probs = self.actor.get_base_model()(ptx, attention_mask=attention_mask)['logits'][..., :-1, :]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs.view(-1, ptx_log_probs.size(-1)), label.view(-1))
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (1 - self.ptx_coef)

        return {'returns': experience.reward.mean().item()}
