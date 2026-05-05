# rdf_core.py
import torch

class RDFCore():
    def __init__(self, n_func, domain_min, domain_max, robot, device, model):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device    
        self.robot = robot
        
        # ⚡ ON PREND TOUS LES LIENS DU MODÈLE DYNAMIQUEMENT
        self.used_links = list(model.keys())
        self.K = len(self.used_links)
        
        # On charge tous les tenseurs une seule fois...
        self.offsets = torch.cat([model[i]['offset'].unsqueeze(0) for i in self.used_links], dim=0).to(device).contiguous()
        self.scales = torch.tensor([model[i]['scale'] for i in self.used_links], device=device).contiguous()
        self.weights = torch.cat([model[i]['weights'].unsqueeze(0) for i in self.used_links], dim=0).to(device).contiguous()

        # Pré-calcul des coefficients binomiaux (ils ne changent jamais)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        self.comb = torch.exp(torch.lgamma(torch.tensor(n + 1.0, device=device)) - 
                              torch.lgamma(i + 1.0) - 
                              torch.lgamma(torch.tensor(n, device=device) - i + 1.0)).contiguous()

        self.used_links_tensor = torch.tensor(self.used_links, dtype=torch.long, device=device)
        self.i_tensor = torch.arange(self.n_func, device=device)

    def build_bernstein_t(self, t):
        t = torch.clamp(t, min=1e-4, max=1-1e-4)
        n = self.n_func - 1
        
        # Utilisation du tenseur statique
        phi = self.comb * (1 - t).unsqueeze(-1) ** (n - self.i_tensor) * t.unsqueeze(-1) ** self.i_tensor
        return phi.float()

    def build_basis_function_from_points(self, p):
        N = len(p)
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
        phi = self.build_bernstein_t(p) 
        phi = phi.reshape(N, 3, self.n_func)
        
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        
        # Optimisation des multiplications de base
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func**3)
        
        return phi_xyz

    def get_whole_body_sdf_batch(self, x, pose, theta):
        """
        Calcule la distance SDF ultra-rapidement (sans création de dictionnaires, sans inverse matriciel).
        """
        B = theta.size(0)
        N = x.size(0)
        K = self.K
        
        # 1. Forward Kinematics (FK)
        trans_list = self.robot.get_transformations_each_link(pose, theta)
        # torch.stack est beaucoup plus propre et "Graph-Safe"
        trans_stacked = torch.stack(trans_list, dim=1) 
        # On utilise le tenseur GPU pour indexer, évitant le transfert CPU->GPU !
        fk_trans = trans_stacked[:, self.used_links_tensor, :, :].reshape(B*K, 4, 4)

        # OPTIMISATION 2 : Remplacement de l'inversion de matrice par une simple transposition
        # Pour une matrice de transformation T = [R | t], T_inv(x) = R^T * (x - t)
        R = fk_trans[:, :3, :3]
        R_inv = R.transpose(1, 2).contiguous() # Transposée de la rotation (B*K, 3, 3)
        t_vec = fk_trans[:, :3, 3].contiguous() # Translation (B*K, 3)
        
        # Différence (x - t)
        # x est (N, 3). On le broadcast en (B*K, N, 3)
        diff = x.unsqueeze(0) - t_vec.unsqueeze(1) 
        
        # OPTIMISATION 3 : Batch Matrix Multiplication (Ultra rapide sur GPU)
        x_robot_frame_batch = torch.bmm(diff, R_inv) # Résultat: (B*K, N, 3)

        # 2. Mise à l'échelle (Scaling)
        offsets_expanded = self.offsets.unsqueeze(0).expand(B, K, 3).reshape(B*K, 1, 3)
        scales_expanded = self.scales.unsqueeze(0).expand(B, K).reshape(B*K, 1, 1)

        x_scaled = (x_robot_frame_batch - offsets_expanded) / scales_expanded

        # 3. Bornage aux limites du volume de Bernstein
        x_bounded = torch.clamp(x_scaled, min=-1.0+1e-2, max=1.0-1e-2)
        res_x = x_scaled - x_bounded

        # 4. Évaluation du polynôme de Bernstein
        phi = self.build_basis_function_from_points(x_bounded.reshape(B*K*N, 3))
        phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B*N, -1) 
        
        # 5. Calcul des distances finales
        sdf = torch.einsum('kni,ki->kn', phi, self.weights).reshape(K, B, N).transpose(0, 1).reshape(B*K, N)
        sdf = sdf + res_x.norm(dim=-1)
        sdf = sdf.reshape(B, K, N)
        sdf = sdf * self.scales.unsqueeze(0).unsqueeze(2)
        
        # On extrait la distance minimale parmi tous les liens du robot
        sdf_value, _ = sdf.min(dim=1)
        
        return sdf_value