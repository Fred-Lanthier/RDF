import torch
from panda_layer.panda_layer import PandaLayer
from rdf_core import RDFCore
from rdf_barrier import RDF_Barrier

def update_obstacles_for_graph(raw_camera_points, static_x_obs_tensor, K_max=100):
    """
    Adapte le nuage de points de la caméra à la taille fixe du CUDA Graph.
    raw_camera_points: Tenseur [N, 3] provenant de TAPNEXT++ / Caméra
    static_x_obs_tensor: Le tenseur de taille fixe [100, 3] lié au graphe
    """
    N = raw_camera_points.size(0)
    
    if N == 0:
        # Aucun obstacle vu : on met tous les points "à l'infini"
        static_x_obs_tensor[:] = torch.tensor([0.0, 0.0, 100.0], device=static_x_obs_tensor.device)
        
    elif N < K_max:
        # PADDING : On copie les N vrais points
        static_x_obs_tensor[:N] = raw_camera_points
        # On remplit le reste (de N à 100) avec le point à l'infini
        static_x_obs_tensor[N:] = torch.tensor([0.0, 0.0, 100.0], device=static_x_obs_tensor.device)
        
    else:
        # TRONCATURE / DOWNSAMPLING : On a trop de points
        # Option simple : prendre les K_max points aléatoirement
        # Option avancée (Jour 12) : prendre les K_max points les plus proches du robot
        indices = torch.randperm(N)[:K_max]
        static_x_obs_tensor[:] = raw_camera_points[indices]

def run_barrier_cuda_graph():
    device = 'cuda'
    
    robot = PandaLayer(device)
    model = torch.load('models/BP_8.pt', map_location=device, weights_only=False)
    
    core = RDFCore(8, -1.0, 1.0, robot, device, model)
    barrier = RDF_Barrier(core, d_safe=0.05)

    # 1. Allocation des mémoires STATIQUES
    static_x_obs = torch.rand(300*16, 3, device=device)
    static_pose = torch.eye(4).unsqueeze(0).to(device)
    
    # CRITIQUE : requires_grad=True doit être défini AVANT la capture
    static_theta = torch.zeros(1, 7, device=device, requires_grad=True)
    #update_obstacles_for_graph(camera_points, static_x_obs)

    print("Warm-up de l'Autograd...")
    # On utilise un stream séparé pour le warm-up comme recommandé par NVIDIA
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            _, _ = barrier(static_theta, static_x_obs, static_pose)
    torch.cuda.current_stream().wait_stream(s)

    # 2. CAPTURE DU GRAPHE CUDA (Forward + Backward !)
    print("Capture du graphe CUDA (Distance + Gradient)...")
    g_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_graph):
        static_h, static_grad = barrier(static_theta, static_x_obs, static_pose)
    print("Capture terminée !")

    # 3. BENCHMARK
    print("Benchmark Hautes Performances...")
    n_iters = 1000
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        # Pour une vraie boucle, tu mettrais à jour static_theta ici avec copy_()
        g_graph.replay()
    end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_iters

    print("-" * 50)
    print(f"Distance au point critique h(q) : {static_h.item():.4f} m")
    print(f"Gradient calculé              : {static_grad.detach().cpu().numpy()}")
    print(f"Temps moyen (Dist + Gradient) : {avg_time_ms:.4f} ms")
    print("-" * 50)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    run_barrier_cuda_graph()