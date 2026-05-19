import math

import torch


def patch_grid_distances(num_patches_per_side, device):
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(num_patches_per_side, device=device),
            torch.arange(num_patches_per_side, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    distances = torch.cdist(coords.float(), coords.float(), p=2)
    return distances


def mean_attention_distance(attention_weights, distances):
    """Return mean query-key distance for attention weights shaped B x H x N x N."""
    weighted_distances = attention_weights * distances.view(1, 1, distances.size(0), distances.size(1))
    per_head = weighted_distances.sum(dim=(-1, -2)) / attention_weights.size(-2)
    return per_head.mean(dim=0)


def attention_mass_beyond(attention_weights, distances, threshold):
    far_mask = distances.ge(threshold).view(1, 1, distances.size(0), distances.size(1))
    per_head = attention_weights.masked_fill(~far_mask, 0.0).sum(dim=(-1, -2)) / attention_weights.size(-2)
    return per_head.mean(dim=0)


def grouped_attention_distance(attention_weights, distances, query_mask, key_mask):
    """Average attention distance from a query group to a key group.

    query_mask and key_mask are B x N boolean tensors. Returned values are
    per-head means averaged across examples that contain both groups.
    """
    batch_size, num_heads, num_tokens, _ = attention_weights.shape
    query = query_mask.view(batch_size, 1, num_tokens, 1)
    key = key_mask.view(batch_size, 1, 1, num_tokens)
    pair_mask = query & key
    weights = attention_weights.masked_fill(~pair_mask, 0.0)
    weight_sum = weights.sum(dim=(-1, -2))

    weighted_distances = weights * distances.view(1, 1, num_tokens, num_tokens)
    distance_sum = weighted_distances.sum(dim=(-1, -2))
    valid = weight_sum > 1e-12
    values = distance_sum / weight_sum.clamp(min=1e-12)

    if not valid.any():
        return torch.full((num_heads,), float("nan"), device=attention_weights.device)

    values = torch.where(valid, values, torch.zeros_like(values))
    return values.sum(dim=0) / valid.float().sum(dim=0).clamp(min=1.0)


def _feed_forward(layer, x):
    x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    return layer.dropout2(x)


def build_encoder_tokens(model, x, mask_ratio=0.0):
    patches = model.patchify(x)
    tokens = model.patch_embed(patches)
    mask = None
    if mask_ratio > 0:
        mask = model.random_mask(x.size(0), mask_ratio, x.device)
        mask_tokens = model.mask_token.expand(x.size(0), model.num_patches, -1)
        tokens = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, tokens)
    return tokens + model.pos_embed, mask


def encoder_attention_distances(model, x, mask_ratio=0.0, far_threshold=4.0):
    """Run the encoder and collect mean attention distance for each layer.

    PyTorch's TransformerEncoderLayer does not expose attention weights in its
    default forward path, so this function mirrors its norm-first/no-dropout
    computation while requesting per-head attention weights.
    """
    if not getattr(model.encoder, "layers", None):
        raise ValueError("Expected model.encoder to be a TransformerEncoder with visible layers")

    distances = patch_grid_distances(model.num_patches_per_side, x.device)
    tokens, mask = build_encoder_tokens(model, x, mask_ratio=mask_ratio)
    visible_mask = None if mask is None else ~mask.bool()
    masked_query = None if mask is None else mask.bool()

    layer_rows = []
    with torch.no_grad():
        for layer_index, layer in enumerate(model.encoder.layers):
            if layer.norm_first:
                attn_input = layer.norm1(tokens)
                attn_out, attn_weights = layer.self_attn(
                    attn_input,
                    attn_input,
                    attn_input,
                    need_weights=True,
                    average_attn_weights=False,
                )
                tokens = tokens + layer.dropout1(attn_out)
                tokens = tokens + _feed_forward(layer, layer.norm2(tokens))
            else:
                attn_out, attn_weights = layer.self_attn(
                    tokens,
                    tokens,
                    tokens,
                    need_weights=True,
                    average_attn_weights=False,
                )
                tokens = layer.norm1(tokens + layer.dropout1(attn_out))
                tokens = layer.norm2(tokens + _feed_forward(layer, tokens))

            per_head = mean_attention_distance(attn_weights, distances)
            far_mass = attention_mass_beyond(attn_weights, distances, far_threshold)
            row = {
                "layer": layer_index,
                "mean_attention_distance": per_head.mean().item(),
                "far_attention_mass": far_mass.mean().item(),
                "head_distances": [value.item() for value in per_head],
                "head_far_attention_mass": [value.item() for value in far_mass],
            }

            if masked_query is not None:
                masked_to_visible = grouped_attention_distance(attn_weights, distances, masked_query, visible_mask)
                visible_to_visible = grouped_attention_distance(attn_weights, distances, visible_mask, visible_mask)
                row.update(
                    {
                        "masked_to_visible_distance": masked_to_visible.nanmean().item(),
                        "visible_to_visible_distance": visible_to_visible.nanmean().item(),
                        "head_masked_to_visible_distances": [value.item() for value in masked_to_visible],
                        "head_visible_to_visible_distances": [value.item() for value in visible_to_visible],
                    }
                )
            else:
                row.update(
                    {
                        "masked_to_visible_distance": "",
                        "visible_to_visible_distance": "",
                        "head_masked_to_visible_distances": [],
                        "head_visible_to_visible_distances": [],
                    }
                )

            layer_rows.append(
                row
            )

        if model.encoder.norm is not None:
            tokens = model.encoder.norm(tokens)

    return layer_rows, tokens


def effective_receptive_field(model, x):
    """Compute simple ERF statistics from input gradients of the mean feature norm."""
    model.zero_grad(set_to_none=True)
    inputs = x.detach().clone().requires_grad_(True)
    features = model.encode_features(inputs)
    score = features.pow(2).mean()
    gradients = torch.autograd.grad(score, inputs)[0].detach().abs().mean(dim=1)

    batch_size, height, width = gradients.shape
    flat = gradients.flatten(1)
    weights = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-12)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing="ij",
    )
    coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()
    center = (weights.unsqueeze(-1) * coords.view(1, -1, 2)).sum(dim=1)
    distances_sq = (coords.view(1, -1, 2) - center.unsqueeze(1)).pow(2).sum(dim=-1)
    radius = torch.sqrt((weights * distances_sq).sum(dim=1))

    entropy = -(weights * weights.clamp(min=1e-12).log()).sum(dim=1) / math.log(height * width)
    sorted_weights = torch.sort(weights, dim=1, descending=True).values
    cumulative = sorted_weights.cumsum(dim=1)
    area_50 = (cumulative < 0.5).sum(dim=1).float() + 1.0
    area_90 = (cumulative < 0.9).sum(dim=1).float() + 1.0

    return {
        "erf_radius_px": radius.mean().item(),
        "erf_entropy": entropy.mean().item(),
        "erf_area_50_frac": (area_50 / (height * width)).mean().item(),
        "erf_area_90_frac": (area_90 / (height * width)).mean().item(),
        "gradient_abs_mean": gradients.mean().item(),
    }
