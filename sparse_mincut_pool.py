import torch
import math
from torch import Tensor

def sparse_mincut_pool_batch(
    x: Tensor,
    edge_index: Tensor,
    s: Tensor,
    batch: Tensor,
    temp: float = 1.0,
    graph_mask: Tensor = None,   #[B] bool, True=real参与MinCut/Ortho
) -> (Tensor, Tensor, Tensor, Tensor):
    """
    Batch-wise sparse MinCut pooling, refined to avoid inplace ops and reduce complexity.

    Returns:
        pooled_x: [B, C, F]
        pooled_adj: [B, C, C]
        mincut_loss: scalar
        ortho_loss: scalar
    """
    device = x.device
    N, F_in = x.size()
    C = s.size(-1)
    B = int(batch.max().item()) + 1

    # soft assignment
    s = torch.softmax((s / temp) if temp != 1.0 else s, dim=-1)

    if graph_mask is None:
        graph_mask = torch.ones(B, device=device, dtype=torch.bool)
    else:
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        if graph_mask.numel() != B:
            raise ValueError(f"graph_mask must have length {B}, got {graph_mask.numel()}")


    # init outputs
    pooled_x = torch.zeros(B, C, F_in, device=device, dtype=x.dtype)
    pooled_adj = torch.zeros(B, C, C, device=device, dtype=x.dtype)

    mincut_losses = []
    ortho_losses = []

    # process each graph in batch
    for b in range(B):
        mask_b = (batch == b)
        x_b = x[mask_b]
        s_b = s[mask_b]

        # extract edges for this graph
        ei = edge_index 
        mask_e = mask_b[ei[0]] & mask_b[ei[1]] 
        ei_b = ei[:, mask_e]

        # re-index nodes
        idx = mask_b.nonzero(as_tuple=False).view(-1)
        idx_map = -torch.ones(N, device=device, dtype=torch.long)
        idx_map[idx] = torch.arange(idx.size(0), device=device)
        ei_b = idx_map[ei_b]

        row, col = ei_b
        # compute A * s
        s_col = s_b[col]               # [E_b, C]

        A_s = torch.zeros(x_b.size(0), C, device=device, dtype=x.dtype)
        A_s.index_add_(0, row, s_col)
        adj_b = torch.mm(s_b.t(), A_s) # [C, C]

        pooled_adj[b] = adj_b
        pooled_x[b]   = torch.mm(s_b.t(), x_b)

        # Only real graphs contribute to MinCut/Ortho
        if graph_mask[b]:
            # MinCut loss
            num = torch.trace(adj_b)
            deg = torch.zeros(x_b.size(0), device=device, dtype=x.dtype)
            deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
            den = torch.trace(torch.mm(s_b.t(), deg.view(-1,1) * s_b)) + 1e-10
            mincut_losses.append(-num / den)

            # Orthogonality loss
            ss = torch.mm(s_b.t(), s_b)
            norm_ss = torch.norm(ss, p='fro') + 1e-10
            ss_norm = ss / norm_ss
            I = torch.eye(C, device=device, dtype=x.dtype) / math.sqrt(C)
            ortho_losses.append(torch.norm(ss_norm - I, p='fro'))

    # normalize pooled_adj across batch
    I = torch.eye(C, device=device, dtype=torch.bool).unsqueeze(0)  # [1,C,C]
    pooled_adj = pooled_adj.masked_fill(I, 0.0)
    # degree
    EPS = 1e-15
    d = pooled_adj.sum(dim=2)           # [B, C]
    d_sqrt = torch.sqrt(d) + EPS
    pooled_adj = pooled_adj / d_sqrt.unsqueeze(2) / d_sqrt.unsqueeze(1)

    if len(mincut_losses) == 0:
        mincut_loss = torch.zeros((), device=device, dtype=x.dtype)
        ortho_loss  = torch.zeros((), device=device, dtype=x.dtype)
    else:
        mincut_loss = torch.stack(mincut_losses).mean()
        ortho_loss  = torch.stack(ortho_losses).mean()

    return pooled_x, pooled_adj, mincut_loss, ortho_loss




