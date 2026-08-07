import torch
import torch.nn as nn

def random_orthonormal(N: int, k: int = None,
                       device=None, dtype=torch.float32, seed: int | None = None) -> torch.Tensor:
    """
    Generate k random orthonormal vectors in R^N (columns of Q), i.e., Q^T Q = I_k.

    Args:
        N: ambient dimension (rows)
        k: number of vectors (cols). If None, k=N.
        device, dtype: torch tensor specs
        seed: optional RNG seed for reproducibility

    Returns:
        Q: [N, k] with orthonormal columns
    """
    if k is None:
        k = N
    if k > N:
        raise ValueError("k must be <= N for column-orthonormal Q.")

    if seed is not None:
        torch.manual_seed(seed)

    X = torch.randn(N, k, device=device, dtype=dtype)
    # QR with reduced mode gives Q:[N,k], R:[k,k]
    Q, R = torch.linalg.qr(X, mode='reduced')

    # Make the result deterministic w.r.t. sign by enforcing positive diag(R)
    s = torch.sign(torch.diag(R))
    s[s == 0] = 1
    Q = Q * s  # column-wise sign fix via broadcasting

    return Q

# Example:
# U = random_orthonormal(N=2048, k=128, device='cuda', seed=42)  # U.T @ U ≈ I_128


class GraphFourier(nn.Module):
    """
    Forward (spatial→spectral)   :  x_hat = Uᵀ · x
    Inverse (spectral→spatial)   :  x_rec = U    · x_hat

    Pick the direction with the `inverse` flag.
    """

    def __init__(self, U: torch.Tensor, k: int, inverse: bool = False):
        super().__init__()
        assert U.ndim == 2, "U must be [N, k]"
        U_k = U[:, :k] if U.shape[1] >= k else U
        self.register_buffer("U", U_k.double(), persistent=False)
        self.k = k
        self.inverse = inverse                        # False = forward GFT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x :  [N, F]  or  [B, N, F]

        Returns
        -------
        Tensor
            [k, F]  or  [B, k, F]   if forward   (inverse=False)
            [N, F]  or  [B, N, F]   if inverse   (inverse=True)
        """
        U = self.U.to(x.device)           # ensure same device

        if x.dim() == 2:                  # --- single sample ---
            return (U @ x) if self.inverse else (U.T @ x)

        if x.dim() == 3:                  # --- batch ---
            # forward:  n k , b n f  ->  b k f
            # inverse:  n k , b k f  ->  b n f
            if self.inverse:
                return torch.einsum("nk,bkf->bnf", U, x)
            else:
                return torch.einsum("nk,bnf->bkf", U, x)

        if x.dim() == 4:                  # --- batch ---
            # forward:  n k , b n f  ->  b k f
            # inverse:  n k , b k f  ->  b n f
            if self.inverse:
                return torch.einsum("nk,tbkf->tbnf", U, x)
            else:
                return torch.einsum("nk,tbnf->tbkf", U, x)

        raise ValueError("x must have shape [N,F] or [B,N,F]")

    def __repr__(self):
        N, k = self.U.shape
        direction = "inverse" if self.inverse else "forward"
        return f"{self.__class__.__name__}(N={N}, k={k}, {direction})"

