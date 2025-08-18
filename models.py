from torch import Tensor
import torch
from torch import nn
import torch.utils.checkpoint
import contextlib
import torchvision
from einops import rearrange

import math
from stgcn_layers import Graph, get_stgcn_chain
from deformable_attention_2d import DeformableAttention2D
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import warnings
from config import mt5_path


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Uni_Sign(nn.Module):
    def __init__(self, args):
        super(Uni_Sign, self).__init__()
        self.args = args

        # order matters; body computed first, hands/face later
        self.modes = ['body', 'left', 'right', 'face_all']

        # Graphs, adjacency and joints-per-part
        self.graph, A = {}, []
        self.num_nodes = {}

        hidden_dim = args.hidden_dim

        # project (x,y,score) -> 64 before GCN
        self.proj_linear = nn.ModuleDict()

        for mode in self.modes:
            g = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            self.graph[mode] = g
            A_tensor = torch.tensor(g.A, dtype=torch.float32, requires_grad=False)
            A.append(A_tensor)
            # K × V × V; infer V from A
            V = A_tensor.shape[-1]
            self.num_nodes[mode] = getattr(g, 'num_node', V)
            self.proj_linear[mode] = nn.Linear(3, 64)

        # ST-GCN stacks
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)  # K subsets

        gcn_out_dim = None
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                64, 'spatial', (1, spatial_kernel_size), A[index].clone(), True
            )
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim, 'temporal', (5, spatial_kernel_size), A[index].clone(), True
            )
            if gcn_out_dim is None:
                gcn_out_dim = final_dim

        # Tie left/right weights (optional sharing)
        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']
        self.num_nodes['left'] = self.num_nodes['right']

        # Pose feature projection & learned per-part bias
        # Use actual GCN output size to avoid mismatches
        concat_dim = gcn_out_dim * len(self.modes)  # e.g., 256 * 4
        self.pose_proj = nn.Linear(concat_dim, 768)
        self.part_para = nn.Parameter(torch.zeros(concat_dim))

        self.apply(self._init_weights)

        # Language
        self.lang = 'Chinese' if "CSL" in self.args.dataset else 'English'

        # Optional RGB support
        if self.args.rgb_support:
            # NOTE: torchvision >=0.13 uses weights arg; pretrained=True still works for older versions
            self.rgb_support_backbone = torch.nn.Sequential(
                *list(torchvision.models.efficientnet_b0(pretrained=True).children())[:-2]
            )
            self.rgb_proj = nn.Conv2d(1280, hidden_dim, kernel_size=1)
            self.fusion_pose_rgb_linear = nn.Linear(hidden_dim, hidden_dim)

            # PGF
            self.fusion_pose_rgb_DA = DeformableAttention2D(
                dim=hidden_dim,
                dim_head=32,
                heads=8,
                dropout=0.,
                downsample_factor=1,
                offset_scale=None,
                offset_groups=None,
                offset_kernel_size=1,
            )

            self.fusion_gate = nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, 1, 1),
                nn.Tanh(),
                nn.ReLU(),
            )
            for layer in self.fusion_gate:
                if isinstance(layer, nn.Conv1d):
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)

        # Text model
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def maybe_autocast(self, dtype=torch.float32):
        enable_autocast = True
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def gather_feat_pose_rgb(self, gcn_feat, rgb_feat, indices, rgb_len, pose_init):
        b, c, T, n = gcn_feat.shape
        assert rgb_feat.shape[0] == indices.shape[0]
        rgb_feat = self.rgb_proj(rgb_feat)

        assert len(rgb_len) == b
        start = 0
        for batch in range(b):
            index = indices[start:start + rgb_len[batch]].to(torch.long)
            # ignore some invalid rgb clip
            if rgb_len[batch] == 1 and -1 in index:
                start = start + rgb_len[batch]
                continue

            # index selection
            gcn_feat_selected = gcn_feat[batch, :, index]
            rgb_feat_selected = rgb_feat[start:start + rgb_len[batch]]
            pose_init_selected = pose_init[start:start + rgb_len[batch]]

            gcn_feat_selected = rearrange(gcn_feat_selected, 'c t n -> t c n')
            pose_init_selected = rearrange(pose_init_selected, 't n c -> t c n')

            # PGF forward
            with self.maybe_autocast():
                fused_transposed = self.fusion_pose_rgb_DA(
                    pose_feat=gcn_feat_selected,
                    rgb_feat=rgb_feat_selected,
                    pose_init=pose_init_selected,
                )

            fused_transposed = fused_transposed.to(gcn_feat.dtype)
            gate_feature = torch.concat([fused_transposed, gcn_feat_selected], dim=-2)
            gate_score = self.fusion_gate(gate_feature)
            fused_transposed_post = (gate_score) * fused_transposed + (1 - gate_score) * gcn_feat_selected

            gcn_feat = gcn_feat.clone()
            fused_transposed_post = rearrange(fused_transposed_post, 't c n -> c t n')

            # replace gcn feature
            gcn_feat[batch, :, index] = fused_transposed_post
            start = start + rgb_len[batch]

        assert start == rgb_feat.shape[0]
        return gcn_feat

    def _shape_safe_pose_input(self, part: str, x: torch.Tensor) -> torch.Tensor:
        """
        Accept (B,T,V,C) or (B,T, V*C); return (B,T,V,3), padding score channel if needed.
        """
        if x.dim() == 3:
            # (B, T, VC) -> (B, T, V, C)
            B, T, VC = x.shape
            V = self.num_nodes[part]
            if VC % V != 0:
                raise RuntimeError(
                    f"Last dim {VC} not divisible by V={V} for part '{part}'. "
                    f"Check dataset shapes/loader."
                )
            C = VC // V
            x = x.view(B, T, V, C).contiguous()
        elif x.dim() == 4:
            # (B, T, V, C)
            B, T, V, C = x.shape
            # optional sanity check
            if V != self.num_nodes[part]:
                # allow but warn silently; graphs can differ if layouts change
                pass
        else:
            raise RuntimeError(f"Unexpected tensor rank for part '{part}': {tuple(x.shape)}")

        # If only (x,y), pad a zero score channel to reach 3
        if x.shape[-1] == 2:
            pad = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)  # (B, T, V, 3)

        return x

    def forward(self, src_input, tgt_input):
        # RGB branch forward
        if self.args.rgb_support:
            rgb_support_dict = {}
            for index_key, rgb_key in zip(
                    ['left_sampled_indices', 'right_sampled_indices'],
                    ['left_hands', 'right_hands']
            ):
                rgb_feat = self.rgb_support_backbone(src_input[rgb_key])
                rgb_support_dict[index_key] = src_input[index_key]
                rgb_support_dict[rgb_key] = rgb_feat

        # Pose branch forward
        features = []
        body_feat = None

        for part in self.modes:
            # shape-safe: (B,T,V,3)
            x = self._shape_safe_pose_input(part, src_input[part])

            # projection (C=3 -> 64), then to (B,64,T,V) for ST-GCN
            proj_feat = self.proj_linear[part](x).permute(0, 3, 1, 2).contiguous()

            # spatial gcn forward
            gcn_feat = self.gcn_modules[part](proj_feat)

            if part == 'body':
                body_feat = gcn_feat
            else:
                assert body_feat is not None
                if part == 'left':
                    # Pose RGB fusion (optional)
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat,
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm'],
                        )
                    gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()

                elif part == 'right':
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat,
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm'],
                        )
                    gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()

                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()

                else:
                    raise NotImplementedError

            # temporal gcn forward
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)  # (B, C, T, V)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)       # (B, T, C)
            features.append(pool_feat)

        # concat sub-pose feature across channel dimension, add learned bias, project
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds)

        prefix_token = self.mt5_tokenizer(
            [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
            padding="longest",
            truncation=True,
            return_tensors="pt",
            ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token['input_ids'])
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        attention_mask = torch.cat(
            [prefix_token['attention_mask'], src_input['attention_mask']], dim=1
        )

        tgt_input_tokenizer = self.mt5_tokenizer(
            tgt_input['gt_sentence'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )

        labels = tgt_input_tokenizer['input_ids']
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(inputs_embeds.device),
            return_dict=True,
        )

        label = labels.reshape(-1)
        out_logits = out['logits']
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing, ignore_index=-100
        )
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        stack_out = {
            'inputs_embeds': inputs_embeds,   # for inference
            'attention_mask': attention_mask,
            'loss': loss,
        }
        return stack_out

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        inputs_embeds = pre_compute_item['inputs_embeds']
        attention_mask = pre_compute_item['attention_mask']

        out = self.mt5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return out


def get_requires_grad_dict(model):
    param_requires_grad = {name: True for name, param in model.named_parameters()}
    param_requires_grad_right = {}
    for key in param_requires_grad.keys():
        if 'left' in key:
            param_requires_grad_right[key.replace("left", 'right')] = param_requires_grad[key]
    param_requires_grad = {**param_requires_grad, **param_requires_grad_right}
    params_to_update = {k: v for k, v in model.state_dict().items() if param_requires_grad.get(k, True)}
    return params_to_update
