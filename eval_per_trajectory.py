#!/usr/bin/env python3
"""Per-trajectory evaluation script for SEA/GraphSpectralFormer models.

Usage:
  python eval_per_trajectory.py --codebase SEA --config cylinder_flow --model_path PATH --gpu 0
  python eval_per_trajectory.py --codebase SEA-baseline --config cylinder_flow --model_path PATH --gpu 1
"""
import argparse, sys, os, json, csv, importlib
import numpy as np
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebase", required=True, choices=["SEA", "SEA-baseline"])
    parser.add_argument("--config", required=True, help="Config module name, e.g. cylinder_flow")
    parser.add_argument("--model_path", required=True, help="Path to temporal model checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    codebase_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.codebase)
    sys.path.insert(0, codebase_dir)
    os.chdir(codebase_dir)

    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.get_config_temporal()
    config["device"] = str(device)
    config["pretrained_model_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.model_path
    ) if not os.path.isabs(args.model_path) else args.model_path
    config["load_pretrained"] = True
    config["batch_size"] = 1
    config["use_wandb"] = False
    config["perform_initial_test"] = False
    config["test_mesh_structure"] = False
    
    if "ib_num" not in config:
        config["ib_num"] = 1
    
    # For cylinder flow: the checkpoint uses 6-feature polynomial conditioning
    if args.config in ("cylinder_flow", "cylinder_flow_highres"):
        config["ib_num"] = 6
        base_dir = "data_old/CF/all_data" if args.codebase == "SEA" else "data/CF/all_data"
        config["input_path"] = os.path.join(base_dir, "input_data_6feat.npy")

    # Fix encoder_decoder_path: derive from same directory as temporal model
    model_dir = os.path.dirname(config["pretrained_model_path"])
    enc_name = f"encoder_decoder_{config['case_name']}_{config.get('run_name','run1').split('_')[0]}.pt"
    # Try finding encoder in same dir as temporal model
    candidate = os.path.join(model_dir, enc_name)
    if os.path.exists(candidate):
        config["encoder_decoder_path"] = candidate
    else:
        # Fallback: try checkpoints-{case} pattern
        for suffix in [config["case_name"].replace("_", "-"), config["case_name"], ""]:
            cand = os.path.join(codebase_dir, f"checkpoints-{suffix}" if suffix else "checkpoints",
                                f"encoder_decoder_{config['case_name']}_run1.pt")
            if os.path.exists(cand):
                config["encoder_decoder_path"] = cand
                break

    from train.train_temporal import get_model, get_datasets
    from utils.train_utils import relativeMSE, relativeMSE_with_time
    from utils.train_utils import inverse_transform_processed_data

    model, loss_fn, _ = get_model(config, device)
    _, _, testLoader, mesh_processor, processor = get_datasets(config)

    model.eval()
    results = []
    
    inv_re_path = os.path.join(codebase_dir, "data_old", "CF", "all_data", "input_data.npy")
    inv_re_data = None
    if os.path.exists(inv_re_path):
        inv_re_data = np.load(inv_re_path).flatten()

    n_total = len(testLoader.dataset)
    perm = np.random.RandomState(42).permutation(config.get("train_fraction", 0.6) != 0.6 and n_total or n_total)

    print(f"Evaluating {len(testLoader)} test batches on {device}...")
    print(f"Config: {args.config}, Codebase: {args.codebase}")
    print(f"Model: {args.model_path}")

    traj_idx = 0
    with torch.no_grad():
        for batch_i, (data, target, original_data, ib) in enumerate(testLoader):
            data = data.to(device)
            target = target.to(device) 
            original_data = original_data.to(device)
            ib = ib.to(device)
            
            B, L, F, D = data.shape
            if ib.dim() == 2:
                ib = ib.unsqueeze(1).expand(B, L, -1)
            elif ib.shape[1] == 1:
                ib = ib.expand(B, L, -1)
            elif ib.shape[1] == F:
                ib = ib.mean(dim=1, keepdim=True).expand(B, L, -1)

            autoreg_input = data[:, 0:1, :, :]
            for i in range(target.shape[1]):
                ib_slice = ib[:, :i+1, :]
                output = model(autoreg_input, ib_slice)
                next_step = output[:, -1:, :, :]
                autoreg_input = torch.cat((autoreg_input, next_step), dim=1)
            
            autoregressive_output = autoreg_input[:, 1:, :, :]
            
            encoded_rel_mse = ((autoregressive_output - target)**2).sum() / ((target**2).sum() + 1e-12)
            
            tr, T, _, _ = autoregressive_output.shape
            if config.get("dimension", "2D") == "3D":
                n_patches = (config["m"]-1) * (config["n"]-1) * (config.get("k", 2)-1)
            else:
                n_patches = (config["m"]-1) * (config["n"]-1)
            
            autoregressive_output_decode = inverse_transform_processed_data(
                autoregressive_output, tr, T, n_patches, len(config["field_groups"])
            )
            autoregressive_output_decode = processor.decode_data(autoregressive_output_decode)
            if config.get("SEA_mixed"):
                B2, P, F2, C = autoregressive_output_decode.shape
                autoregressive_output_decode = autoregressive_output_decode.reshape(B2, P, C, F2)
            elif config.get("SEA_isolate"):
                autoregressive_output_decode = autoregressive_output_decode.permute(0, 1, 3, 2)
            
            autoregressive_output_decode = autoregressive_output_decode.to("cpu")
            autoregressive_output_decode = mesh_processor.inverse_scale_and_unpatch(autoregressive_output_decode).to(device)
            
            _, C, n_fields = autoregressive_output_decode.shape
            autoregressive_output_decode = autoregressive_output_decode.reshape(tr, T, C, n_fields)
            
            per_field_mse = []
            for f_idx in range(n_fields):
                pred_f = autoregressive_output_decode[:, :, :, f_idx]
                gt_f = original_data[:, :, :, f_idx]
                field_relmse = ((pred_f - gt_f)**2).sum() / ((gt_f**2).sum() + 1e-12)
                per_field_mse.append(field_relmse.item())
            
            macro_relmse = sum(per_field_mse) / len(per_field_mse)
            
            entry = {
                "batch_idx": batch_i,
                "encoded_relmse": encoded_rel_mse.item(),
                "decoded_macro_relmse": macro_relmse,
                "decoded_per_field_relmse": per_field_mse,
                "num_steps": T,
            }
            results.append(entry)
            field_str = " ".join([f"f{i}={v:.6f}" for i, v in enumerate(per_field_mse)])
            print(f"  Batch {batch_i}: macro={macro_relmse:.6f}  {field_str}  enc={encoded_rel_mse.item():.6f}")
            traj_idx += 1

    avg_macro = np.mean([r["decoded_macro_relmse"] for r in results])
    avg_encoded = np.mean([r["encoded_relmse"] for r in results])
    
    summary = {
        "config": args.config,
        "codebase": args.codebase,
        "model_path": args.model_path,
        "num_trajectories": len(results),
        "avg_decoded_macro_relmse": avg_macro,
        "avg_encoded_relmse": avg_encoded,
        "per_trajectory": results,
    }
    
    print(f"\n=== SUMMARY ===")
    print(f"Config: {args.config}, Codebase: {args.codebase}")
    print(f"Trajectories: {len(results)}")
    print(f"Average decoded macro RelMSE: {avg_macro:.6f}")
    print(f"Average encoded RelMSE: {avg_encoded:.6f}")
    
    out_path = args.output or f"eval_results_{args.codebase}_{args.config}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
