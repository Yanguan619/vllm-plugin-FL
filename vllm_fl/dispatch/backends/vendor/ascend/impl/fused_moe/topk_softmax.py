import torch


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    # Use pure PyTorch implementation to avoid FlagGems Triton kernel
    # issues on Ascend NPU.
    topk = topk_weights.size(1)

    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights_out, topk_indices_out = torch.topk(
        scores, k=topk, dim=-1, sorted=False
    )
    if renormalize:
        topk_weights_out = topk_weights_out / topk_weights_out.sum(
            dim=-1, keepdim=True
        )
    topk_weights.copy_(topk_weights_out)
    topk_indices.copy_(topk_indices_out.to(topk_indices.dtype))
    return topk_weights, topk_indices
