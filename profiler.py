import torch
from panda_layer.panda_layer import PandaLayer
from rdf_core import RDFCore

def run_profiler():
    device = 'cuda'
    print("Initialisation du Profiler...")
    
    robot = PandaLayer(device)
    model = torch.load('models/BP_8.pt', map_location=device, weights_only=False)
    core = RDFCore(8, -1.0, 1.0, robot, device, model)

    x_obs = torch.rand(100, 3).to(device)
    theta = torch.zeros(1, 7).to(device)
    pose = torch.eye(4).unsqueeze(0).to(device)

    # 1. Warm-up
    for _ in range(10):
        _ = core.get_whole_body_sdf_batch(x_obs, pose, theta)
    
    torch.cuda.synchronize()

    # 2. Profiling de 100 itérations
    print("Analyse des noyaux CUDA en cours...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(100):
            _ = core.get_whole_body_sdf_batch(x_obs, pose, theta)
            
    # 3. Affichage des 15 opérations les plus lentes sur le GPU
    print("\n--- TOP 15 DES OPÉRATIONS LES PLUS LENTES ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    run_profiler()