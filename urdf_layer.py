import torch
import torch.nn.functional as F
import trimesh
import os
import pytorch_kinematics as pk
import warnings

class URDFLayer(torch.nn.Module):
    def __init__(self, urdf_path, device='cpu', package_dir=None, voxel_dir=None):
        super().__init__()
        self.device = device
        self.urdf_path = urdf_path
        self.package_dir = package_dir if package_dir else os.path.dirname(os.path.abspath(urdf_path))
        self.voxel_dir = voxel_dir
        
        # --- CONSTANTES PRÉ-CALCULÉES POUR LE GRAPHE CUDA ---
        self.register_buffer('I33', torch.eye(3, device=device))
        self.register_buffer('corner_diag', (torch.arange(4, device=device) == 3).float())
        self.register_buffer('bottom_row', (torch.arange(4, device=device) == 3).float().view(1, 1, 4))
        
        # 1. On utilise pk UNIQUEMENT comme lecteur XML pour s'épargner du code
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.urdf_path.endswith('.xacro'):
                import xacro
                doc = xacro.process_file(self.urdf_path)
                urdf_text = doc.toxml()
            else:
                with open(self.urdf_path, 'r') as f:
                    urdf_text = f.read()
            self.chain = pk.build_chain_from_urdf(urdf_text.encode()).to(dtype=torch.float32, device=self.device)
            
        self.joint_names = self.chain.get_joint_parameter_names()
        self.dof = len(self.joint_names)
        
        # 2. Construction de l'Arbre Natif (Extraction de la moelle du URDF)
        self.kinematic_tree = []
        self._extract_tree(self.chain._root)
        
        self.meshes_info = self._extract_meshes_from_chain()
        
        # Discover Joint Limits 
        limits = self.chain.get_joint_limits()
        if len(limits) == 2 and len(limits[0]) == self.dof: 
            self.theta_min = torch.tensor(limits[0], device=self.device)
            self.theta_max = torch.tensor(limits[1], device=self.device)
            self.theta_mid = (self.theta_min + self.theta_max) / 2.0
            self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
            self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        elif len(limits) > 0:
            joint_lim = torch.tensor(limits)
            self.theta_min = joint_lim[:, 0].to(self.device)
            self.theta_max = joint_lim[:, 1].to(self.device)
            self.theta_mid = (self.theta_min + self.theta_max) / 2.0
            self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
            self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        else:
            self.theta_min = self.theta_max = self.theta_mid = self.theta_min_soft = self.theta_max_soft = torch.empty(0).to(self.device)

    def _extract_tree(self, node):
        """Parcourt le URDF 1 seule fois et pré-calcule TOUTES les mathématiques statiques."""
        for child in node.children:
            joint = child.joint
            offset_mat = joint.offset.get_matrix().squeeze(0).to(self.device)
            
            axis = K = K_sq = None
            if joint.axis is not None:
                axis = joint.axis.to(self.device)
                if joint.joint_type in ['revolute', 'continuous']:
                    # Matrice Anti-Symétrique précalculée pour la formule de Rodrigues
                    K_mat = torch.tensor([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ], device=self.device)
                    K = K_mat
                    K_sq = torch.matmul(K_mat, K_mat)
                    
            idx = self.joint_names.index(joint.name) if joint.name in self.joint_names else -1
            
            self.kinematic_tree.append({
                'child_link': child.link.name,
                'parent_link': node.link.name,
                'type': joint.joint_type,
                'offset': offset_mat,
                'axis': axis,
                'K': K,
                'K_sq': K_sq,
                'idx': idx
            })
            self._extract_tree(child)

    def _resolve_mesh_path(self, raw_path):
        import rospkg
        rospack = rospkg.RosPack()
        
        # 1. Expand ROS package:// URIs
        if raw_path.startswith('package://'):
            parts = raw_path[10:].split('/', 1)
            pkg_name = parts[0]
            rel_path = parts[1] if len(parts) > 1 else ''
            try:
                pkg_dir = rospack.get_path(pkg_name)
                raw_path = os.path.join(pkg_dir, rel_path)
            except rospkg.ResourceNotFound:
                pass # Fallback to manual resolution below

        # 2. Force voxel directory override if requested
        if self.voxel_dir is not None:
            basename = os.path.basename(raw_path)
            name_no_ext = os.path.splitext(basename)[0]
            voxel_candidates = [
                os.path.join(self.voxel_dir, f"{name_no_ext}.stl"),
                os.path.join(self.voxel_dir, f"{name_no_ext}.STL"),
                os.path.join(self.voxel_dir, f"{name_no_ext}.obj")
            ]
            for candidate in voxel_candidates:
                if os.path.isfile(candidate):
                    return candidate

        if os.path.isfile(raw_path):
            return raw_path

        # 3. Last ditch fallback (manual path stripping)
        if 'package://' in raw_path:
            relative = raw_path.replace('package://', '')
            p1 = os.path.normpath(os.path.join(self.package_dir, relative))
            if os.path.exists(p1): return p1
            stripped_relative = relative.split('/', 1)[-1] if '/' in relative else relative
            p2 = os.path.normpath(os.path.join(self.package_dir, stripped_relative))
            if os.path.exists(p2): return p2
            p3 = os.path.normpath(os.path.join(os.path.dirname(self.package_dir), stripped_relative))
            if os.path.exists(p3): return p3
            return p1
        elif not os.path.isabs(raw_path):
            return os.path.join(os.path.dirname(self.urdf_path), raw_path)
        return raw_path

    def _extract_meshes_from_chain(self):
        meshes_info = []
        for link in self.chain.get_links():
            if hasattr(link, 'visuals') and link.visuals:
                for visual in link.visuals:
                    if visual.geom_type == 'mesh':
                        filename, scale_tuple = visual.geom_param
                        scale = list(scale_tuple) if scale_tuple else [1.0, 1.0, 1.0]
                        offset = getattr(visual, 'offset', None)
                        if offset is None:
                            offset_mat = torch.eye(4, dtype=torch.float32, device=self.device)
                        else:
                            offset_mat = offset.get_matrix().squeeze(0).to(self.device)
                        
                        abs_mesh_path = self._resolve_mesh_path(filename)
                        if not os.path.isfile(abs_mesh_path): continue
                            
                        meshes_info.append({
                            'link_name': link.name,
                            'mesh_path': abs_mesh_path,
                            'scale': scale,
                            'visual_offset': offset_mat
                        })
        return meshes_info

    def get_mesh_paths(self): return [info['mesh_path'] for info in self.meshes_info]
    def get_mesh_names(self): return [os.path.basename(info['mesh_path']).split('.')[0] for info in self.meshes_info]

    # =====================================================================
    # ⚡ LE MOTEUR CINÉMATIQUE PUR PYTORCH (100% GRAPH & VMAP SAFE)
    # =====================================================================
    def _build_revolute(self, K, K_sq, q):
        # L'utilisation de [..., None, None] permet à vmap de gérer ses 
        # dimensions secrètes sans qu'on ne force un "Batch Size" manuellement.
        sin_q = torch.sin(q)[..., None, None]
        cos_q = torch.cos(q)[..., None, None]
        
        R = self.I33 + sin_q * K + (1 - cos_q) * K_sq 
        R_44 = F.pad(R, (0, 1, 0, 1))
        return R_44 + torch.diag(self.corner_diag)

    def _build_prismatic(self, axis, q):
        translation = (axis * q[..., None])[..., None]
        batch_shape = translation.shape[:-2]
        
        I33_exp = self.I33.expand(*batch_shape, 3, 3)
        top_3x4 = torch.cat([I33_exp, translation], dim=-1)
        bottom = self.bottom_row.expand(*batch_shape, 1, 4)
        return torch.cat([top_3x4, bottom], dim=-2)

    def _native_forward_kinematics(self, theta):
        batch_shape = theta.shape[:-1]
        poses = {}
        
        poses[self.chain._root.link.name] = torch.eye(4, device=self.device, dtype=theta.dtype).expand(*batch_shape, 4, 4)
        
        for joint in self.kinematic_tree:
            T_offset = joint['offset'].expand(*batch_shape, 4, 4).to(dtype=theta.dtype)
            
            if joint['type'] in ['revolute', 'continuous']:
                q = theta[..., joint['idx']]
                T_joint = self._build_revolute(joint['K'], joint['K_sq'], q)
                # ⚡ CRITIQUE : L'opérateur @ (matmul) s'adapte à vmap, contrairement à torch.bmm
                T_local = T_offset @ T_joint
            elif joint['type'] == 'prismatic':
                q = theta[..., joint['idx']]
                T_joint = self._build_prismatic(joint['axis'], q)
                T_local = T_offset @ T_joint
            else: # fixed
                T_local = T_offset
                
            T_parent = poses[joint['parent_link']]
            poses[joint['child_link']] = T_parent @ T_local
            
        return poses

    def get_transformations_each_link(self, pose, theta):
        # Padding Dynamique sans forcer la dimension B
        if theta.shape[-1] < self.dof:
            batch_shape = theta.shape[:-1]
            padding = theta.new_zeros((*batch_shape, self.dof - theta.shape[-1]))
            theta = torch.cat([theta, padding], dim=-1)
            
        link_poses = self._native_forward_kinematics(theta)
        
        synced_mesh_transforms = []
        for info in self.meshes_info:
            target_link = info['link_name']
            if target_link not in link_poses:
                raise ValueError(f"Link {target_link} not mapped.")
            
            T_joint = link_poses[target_link]
            T_world_joint = pose @ T_joint
            
            batch_shape = T_world_joint.shape[:-2]
            T_visual = info['visual_offset'].expand(*batch_shape, 4, 4)
            T_final = T_world_joint @ T_visual
            
            synced_mesh_transforms.append(T_final)
            
        return synced_mesh_transforms

    def get_forward_robot_mesh(self, pose, theta):
        batch_size = theta.shape[0] if len(theta.shape)>1 else 1
        synced_transforms = self.get_transformations_each_link(pose, theta)
        
        batch_robots = []
        for b in range(batch_size):
            robot_meshes = []
            for i, info in enumerate(self.meshes_info):
                mesh = trimesh.load(info['mesh_path'], force='mesh')
                mesh.apply_scale(info['scale'])
                mesh.apply_transform(synced_transforms[i][b].detach().cpu().numpy())
                robot_meshes.append(mesh)
            batch_robots.append(robot_meshes)
        return batch_robots