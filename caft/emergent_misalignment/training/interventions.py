import torch as t
import torch.nn as nn
from transformers.utils import logging
import numpy as np
import json
from .sae_utils import load_dictionary_learning_batch_topk_sae


def get_intervention(config):
    if config.intervention_type == "pca":
        return get_pca_intervention(config.intervention_kwargs_path)
    elif config.intervention_type == "dirs":
        return get_dirs_intervention(config.intervention_kwargs_path)
    elif config.intervention_type == "sae":
        return get_sae_intervention(config.intervention_kwargs_path)
    elif config.intervention_type == "probe":
        return get_probe_intervention(config.intervention_kwargs_path)
    else:
        raise ValueError(f"Intervention type {config.intervention_type} not supported")


def get_pca_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    pc_path = intervention_kwargs['path']
    intervention_pcs = intervention_kwargs['dict']

    layers = [int(layer) for layer in intervention_pcs.keys()]
    Qs = []
    intervention_indices = []
    for layer,pc_idx in intervention_pcs.items():
        all_pcs = np.load(f"{pc_path}{layer}.npy")
        if "n_pcs" in intervention_kwargs:
            all_pcs = all_pcs[:intervention_kwargs["n_pcs"]]
        if pc_idx == "all":
            pcs = all_pcs.T
            intervention_indices.append(np.arange(pcs.shape[0]))
        elif pc_idx == "random":
            # flip a coin to decide if we should use each PC
            if "p" in intervention_kwargs:
                p = intervention_kwargs["p"]
            else:
                p = 0.5
            idxs = np.random.rand(all_pcs.shape[0]) < p
            # make sure we use at least one PC
            while np.sum(idxs) == 0:
                idxs = np.random.rand(all_pcs.shape[0]) < p
            intervention_indices.append(np.where(idxs)[0])
            pcs = all_pcs[idxs].T
        else:
            intervention_indices.append(pc_idx)
            pcs = all_pcs[pc_idx].T
        pcs = t.from_numpy(pcs)
        pcs = pcs.to(t.bfloat16)
        Qs.append(pcs)

    return layers, Qs, intervention_indices


def get_dirs_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    dirs_path = intervention_kwargs['path']
    intervention_dirs = intervention_kwargs['dict']

    layers = [int(layer) for layer in intervention_dirs.keys()]
    Qs = []
    intervention_indices = []
    for layer,dir_idx in intervention_dirs.items():
        all_dirs = np.load(f"{dirs_path}{layer}.npy")
        if dir_idx == "all":
            dirs = all_dirs.T
            intervention_indices.append(np.arange(all_dirs.shape[0]))
        else:
            dirs = all_dirs[dir_idx].T
            intervention_indices.append(dir_idx)
        dirs = t.from_numpy(dirs)

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(dirs)
        Q = Q.to(t.bfloat16)

        Qs.append(Q)

    return layers, Qs, intervention_indices


def get_probe_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    # Accept either 'pt_path' or 'path' for flexibility
    pt_path = intervention_kwargs.get('pt_path', intervention_kwargs.get('path'))
    if pt_path is None:
        raise ValueError("Probe intervention requires 'pt_path' (or 'path') to the .pt file")

    obj = t.load(pt_path, map_location='cpu')

    # Optional: restrict to specific layers
    selected_layers = intervention_kwargs.get('layers', None)

    def to_Q(dirs_like):
        # Convert various container types into a 2D torch tensor [d_model, k]
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return t.from_numpy(x)
            if t.is_tensor(x):
                return x
            if isinstance(x, (list, tuple)):
                try:
                    return t.tensor(x)
                except Exception:
                    pass
            return None

        dirs_t = to_tensor(dirs_like)
        if dirs_t is None:
            raise ValueError("Loaded directions must be convertible to a numeric tensor (torch/np/list)")
        dirs_t = dirs_t.float()
        # Shape handling: 1D -> [d_model, 1], 2D -> ensure [d_model, k]
        if dirs_t.ndim == 1:
            dirs_t = dirs_t.unsqueeze(-1)
        elif dirs_t.ndim != 2:
            raise ValueError(f"Directions must be 1D or 2D, got shape {tuple(dirs_t.shape)}")
        if dirs_t.shape[0] >= dirs_t.shape[1]:
            pass  # already [d_model, k]
        else:
            dirs_t = dirs_t.T
        # Orthonormalize optional
        if intervention_kwargs.get('orthonormalize', True):
            Q, _ = t.linalg.qr(dirs_t)
        else:
            Q = dirs_t
        return Q.to(t.bfloat16)

    def extract_array_like(x):
        # Try direct conversion first
        try:
            return to_Q(x)
        except Exception:
            pass
        # If dict, try common keys then recurse over values
        if isinstance(x, dict):
            for key in ["directions", "direction", "vector", "mean", "mean_diff", "meandiff", "w", "v"]:
                if key in x:
                    try:
                        return to_Q(x[key])
                    except Exception:
                        pass
            for v in x.values():
                try:
                    return to_Q(v)
                except Exception:
                    # If nested dict/list, recurse
                    res = extract_array_like(v)
                    if isinstance(res, t.Tensor):
                        return res
        # If list/tuple, try elements
        if isinstance(x, (list, tuple)):
            try:
                return to_Q(x)
            except Exception:
                for v in x:
                    try:
                        return to_Q(v)
                    except Exception:
                        res = extract_array_like(v)
                        if isinstance(res, t.Tensor):
                            return res
        raise ValueError("Could not extract a numeric tensor/array from the provided object")

    layers = []
    Qs = []

    if isinstance(obj, dict):
        # Check if keys are numeric (layer-indexed dict)
        keys = list(obj.keys())
        keys_are_numeric = True
        try:
            _ = [int(k) for k in keys]
        except Exception:
            keys_are_numeric = False

        if keys_are_numeric:
            for layer_key, dirs in obj.items():
                layer_idx = int(layer_key)
                if (selected_layers is not None) and (layer_idx not in selected_layers):
                    continue
                Q = to_Q(dirs)
                Qs.append(Q)
                layers.append(layer_idx)
        else:
            if selected_layers is None:
                raise ValueError("When providing a dict without numeric layer keys, specify 'layers' in the kwargs JSON.")
            Q = extract_array_like(obj)
            for layer_idx in selected_layers:
                layers.append(int(layer_idx))
                Qs.append(Q)
    else:
        # Single tensor/array/list case
        if selected_layers is None:
            raise ValueError("When providing a single object, you must specify 'layers' in the kwargs")
        Q = extract_array_like(obj)
        for layer_idx in selected_layers:
            layers.append(int(layer_idx))
            Qs.append(Q)

    return layers, Qs, []


def get_sae_intervention(intervention_kwargs_path):
    with open(intervention_kwargs_path, 'r') as f:
        intervention_kwargs = json.load(f)
    intervention_latents = intervention_kwargs['dict']
    model_name = intervention_kwargs['model']
    if "Mistral" in model_name:
        sae_repo = "adamkarvonen/mistral_24b_saes"
        sae_base_path = "mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k"
        if "trainer" in intervention_kwargs:
            trainer = intervention_kwargs["trainer"]
        else:
            trainer = "trainer_1"
    elif "Qwen" in model_name:
        sae_repo = "adamkarvonen/qwen_coder_32b_saes"
        sae_base_path = "._saes_Qwen_Qwen2.5-Coder-32B-Instruct_batch_top_k"
        if "trainer" in intervention_kwargs:
            trainer = intervention_kwargs["trainer"]
        else:
            trainer = "trainer_1"
    else:
        raise ValueError(f"Model name {model_name} not supported")

    layers = []
    Qs = []
    for layer,latents in intervention_latents.items():
        if latents == "random":
            # load top sae
            with open(intervention_kwargs["top_sae_path"], 'r') as f:
                top_sae = json.load(f)
            latents = top_sae["layer_"+str(layer)]

            if "k" in intervention_kwargs:
                latents = latents[:intervention_kwargs["k"]]

            # flip a coin to decide if we should use each latent
            if "p" in intervention_kwargs:
                p = intervention_kwargs["p"]
            else:
                p = 0.5
            idxs = np.random.rand(len(latents)) < p
            # make sure we use at least one latent
            while np.sum(idxs) == 0:
                idxs = np.random.rand(len(latents)) < p
            # convert latents to numpy array
            latents = np.array(latents)[idxs].tolist()
        elif latents == "top":
            # load top sae
            with open(intervention_kwargs["top_sae_path"], 'r') as f:
                top_sae = json.load(f)
            latents = top_sae["layer_"+str(layer)]

            # get top k latents (all if no k is provided)
            if "k" in intervention_kwargs:
                latents = latents[:intervention_kwargs["k"]]

        if len(latents) == 0:
            continue

        layers.append(int(layer))

        # load saes
        print(f"Loading SAE for layer {layer} with trainer {trainer}")
        sae_path = f"{sae_base_path}/resid_post_layer_{layer}/{trainer}/ae.pt"
        sae = load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=t.device("cuda"),
        dtype=t.float32,
        layer=int(layer),
        local_dir="/root/downloaded_saes",
        )

        # get latent vectors
        dirs = sae.W_dec[latents].T

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(dirs)
        Q = Q.to(t.bfloat16)

        Qs.append(Q)

    return layers, Qs, []



def create_projection_hook(Q):
    Q_per_device = {}
    def hook(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output

        # Ensure Q is on the same device as activations
        dev_key = str(act.device)
        Q_dev = Q_per_device.get(dev_key)
        if Q_dev is None:
            Q_dev = Q.to(act.device)
            Q_per_device[dev_key] = Q_dev

        # Project and subtract projection
        proj = (act @ Q_dev) @ Q_dev.T  # [batch seq d_model]
        act = act - proj

        if isinstance(output, tuple):
            return (act,) + output[1:]
        else:
            return act
    return hook



def add_intervention_hooks(model, layers, Qs):
    """Add custom non-trainable layers based on configuration."""
    
    # Get the device of the model
    device = next(model.parameters()).device

    hook_handles = []
    
    # Create and add custom layers 
    for layer_idx, Q in zip(layers, Qs):
        # Check if the layer index is valid
        if layer_idx >= len(model.model.model.layers):
            logging.warning(f"Layer index {layer_idx} is out of range. Model has {len(model.model.model.layers)} layers.")
            continue
        
        # Detach Q and move to device
        Q = Q.detach().to(device)

        # Create a hook for the layer
        hook_handle = model.model.model.layers[layer_idx].register_forward_hook(create_projection_hook(Q))
        hook_handles.append(hook_handle)
    return model, hook_handles

