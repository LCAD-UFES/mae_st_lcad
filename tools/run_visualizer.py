import argparse
import os.path as osp
import sys
import torch

# Garante que o script encontre a raiz do projeto e o bootstrapper
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from custom import experiment as exp

def main():
    parser = argparse.ArgumentParser(description="Standalone Visualizer using Config and Hooks")
    parser.add_argument("config", help="Caminho para o arquivo de configuração (.py)")
    parser.add_argument("checkpoint", help="Caminho para o arquivo .pth do modelo")
    parser.add_argument("--out_dir", default="vis_output", help="Diretório onde os vídeos serão salvos")
    args = parser.parse_args()

    # 1. Carrega as variáveis do config (input_size, patch_size, batch_size, etc)
    cfg = exp.load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))
    
    # 2. Fixa as sementes para garantir as mesmas máscaras do treinamento
    seed = cfg["seed"]
    exp.seed_everything(seed)

    # 3. Reconstrói o Dataloader de validação usando as regras originais
    print("[vis] Construindo dataset de validação a partir do config...")
    val_ds = exp.build_dataset(cfg["val_dataset"])
    val_clips, val_stems = exp.load_fixed_clips(val_ds, cfg.get("val_max_videos"), seed)
    print(f"[vis] Foram carregados {len(val_clips)} vídeos do Dataloader.")

    # 4. Inicializa o modelo com as dimensões e parâmetros exatos do config
    print(f"[vis] Inicializando arquitetura e carregando checkpoint: {args.checkpoint}...")
    model = exp.build_model(cfg, device)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Se o dict principal for 'model' (padrão MAE), pega ele, senão pega a raiz
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[vis] Status do carregamento: {msg}")
    
    model.to(device)
    model.eval()

    # 5. Roda os ganchos (Hooks) definidos no config
    tag = "manual_vis"
    val_batch = cfg.get("val_batch_size", cfg["batch_size"])
    
    print(f"[vis] Executando ReconstructionVideoHook no device {device}...")
    exp.run_hooks(
        cfg["hooks"],
        model,
        val_clips,
        val_stems,
        tag,
        args.out_dir,
        val_batch,
        device,
        seed
    )
    
    print(f"[vis] Sucesso! Vídeos salvos em: {osp.join(args.out_dir, 'vis_data', tag)}")

if __name__ == "__main__":
    main()