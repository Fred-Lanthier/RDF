# rdf_barrier.py
import torch
import torch.nn as nn

class RDF_Barrier(nn.Module):
    """
    Module PyTorch agissant comme une Control Barrier Function (CBF).
    Il encapsule le moteur RDF pour fournir la distance ET le gradient exact.
    """
    def __init__(self, rdf_core, d_safe=0.05):
        super().__init__()
        self.rdf_core = rdf_core
        self.d_safe = d_safe  # Marge de sécurité (ex: 0.05 = 5 centimètres)

    def forward(self, q, x_obs, pose=None):
        B = q.size(0)
        
        if pose is None:
            pose = torch.eye(4, device=q.device).unsqueeze(0).expand(B, 4, 4)

        if not q.requires_grad:
            q.requires_grad_(True)

        # 1. Distances pour tous les points : shape [Batch, N_points]
        sdf_distances = self.rdf_core.get_whole_body_sdf_batch(x_obs, pose, q)

        # 2. LE CORRECTIF : On extrait le pire cas (le point le plus proche)
        # min_distance shape : [Batch]
        min_distance, min_idx = torch.min(sdf_distances, dim=1)

        # 3. Calcul de la Barrière h(q) sur le point critique
        h = min_distance - self.d_safe

        # 4. Le Gradient (PyTorch saura exactement quel lien du robot est menacé)
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=q,
            grad_outputs=torch.ones_like(h),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        return h, grad_h, min_idx