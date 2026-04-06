import torch
from torch.nn.functional import softplus
from trl.experimental.kto import KTOTrainer


class RDROTrainer(KTOTrainer):
    _tag_names = ["trl", "rdro"]
    _name = "RDRO"

    def __init__(self, alpha: float = 0.5, **kwargs):
        assert 0.0 < alpha < 1.0, "alpha must be in (0,1)."

        self.alpha = alpha
        self.kwargs = kwargs
        self.calculate_KL = False
        super().__init__(**kwargs)

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self._rdro_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            policy_KL_logps=policy_KL_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            reference_KL_logps=reference_KL_logps,
        )

    def _rdro_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the RDRO loss for a batch of policy and reference model log probabilities.
        """

        # Compute KL divergence
        kl = self._compute_kl_divergence(policy_KL_logps=policy_KL_logps, reference_KL_logps=reference_KL_logps)

        # Compute chosen losses and rewards
        if policy_chosen_logps.shape[0] == 0 and reference_chosen_logps.shape[0] == 0:
            chosen_losses = self._empty_tensor()
            chosen_rewards = self._empty_tensor()
        else:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = (1 + self.alpha) * softplus(chosen_logratios) - chosen_logratios
            chosen_rewards = self.beta * chosen_logratios.detach()

        # Compute rejected losses and rewards
        if policy_rejected_logps.shape[0] == 0 and reference_rejected_logps.shape[0] == 0:
            rejected_losses = self._empty_tensor()
            rejected_rewards = self._empty_tensor()
        else:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = (1 - self.alpha) * softplus(rejected_logratios)
            rejected_rewards = self.beta * rejected_logratios.detach()

        losses = torch.cat([chosen_losses, rejected_losses], dim=0)

        return losses, chosen_rewards, rejected_rewards, kl  # type: ignore

    def _compute_kl_divergence(
        self, policy_KL_logps: torch.FloatTensor, reference_KL_logps: torch.FloatTensor
    ) -> torch.Tensor:
        if not self.calculate_KL:
            return torch.zeros(1).to(self.accelerator.device)

        kl = (policy_KL_logps - reference_KL_logps).mean().detach()
        kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        return kl  # type: ignore

    def _empty_tensor(self) -> torch.Tensor:
        return torch.Tensor([]).to(self.accelerator.device)
