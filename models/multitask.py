import torch
import torch.nn as nn
import torch.nn.functional as F
from .ext_layers import ArcFullyConnected
from . import backbones

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KDLoss(nn.Module):
    """Knowledge distillation loss for embeddings.

    Supports 'cosine' (1 - cos_sim) and 'mse' (Mean Squared Error).
    Optionally normalizes embeddings before computing loss.
    Optionally scales embeddings by a temperature before computing loss.
    """
    def __init__(self, loss_type='cosine', temperature=1.0, normalize_features=True):
        super(KDLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.normalize_features = normalize_features
        if loss_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss(reduction='mean')
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError("Unknown KD loss type: {}".format(loss_type))

    def forward(self, student_feat, teacher_feat):
        if self.normalize_features:
            student_feat = F.normalize(student_feat, p=2, dim=1)
            teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

        if self.temperature != 1.0:
            student_feat = student_feat / self.temperature
            teacher_feat = teacher_feat / self.temperature
            
        if self.loss_type == 'cosine':
            # target = +1 for all pairs (we want them aligned)
            target = torch.ones(student_feat.size(0), device=student_feat.device)
            return self.criterion(student_feat, teacher_feat, target)
        else:
            return self.criterion(student_feat, teacher_feat)


class MultiTaskWithLoss(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, spatial_size,
                 arc_fc=False, feat_bn=False, s=64, m=0.5, is_pw=True,
                 is_hard=False, loss_type='crossentropy', scale=None,
                 backbone_kwargs=None, kd_config=None):
        super(MultiTaskWithLoss, self).__init__()
        self.feat_bn = feat_bn

        # Prepare student backbone kwargs
        b_kwargs = backbone_kwargs.copy() if backbone_kwargs is not None else {}
        if scale is not None:
            b_kwargs['scale'] = scale

        self.basemodel = backbones.__dict__[backbone](
            feature_dim=feature_dim, spatial_size=spatial_size, **b_kwargs)
        
        if feat_bn:
            self.bn1d = nn.BatchNorm1d(feature_dim, affine=False, eps=2e-5, momentum=0.9)
        
        if loss_type == 'focal':
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # --- Knowledge Distillation setup ---
        self.kd_enabled = False
        if kd_config is not None and kd_config.get('enabled', False):
            self.kd_enabled = True
            self.kd_alpha = kd_config.get('alpha', 0.5)

            teacher_backbone = kd_config['teacher_backbone']
            teacher_feature_dim = kd_config.get('teacher_feature_dim', feature_dim)
            teacher_input_size = kd_config.get('teacher_input_size', spatial_size)
            teacher_scale = kd_config.get('teacher_scale', None)
            teacher_b_kwargs = kd_config.get('teacher_backbone_kwargs', {}).copy()
            if teacher_scale is not None:
                teacher_b_kwargs['scale'] = teacher_scale

            temperature = kd_config.get('temperature', 1.0)
            normalize_features = kd_config.get('normalize_features', True)

            # Build frozen teacher backbone
            self.teacher_model = backbones.__dict__[teacher_backbone](
                feature_dim=teacher_feature_dim, spatial_size=teacher_input_size, **teacher_b_kwargs)
            self.teacher_feat_bn = None
            if feat_bn:
                self.teacher_feat_bn = nn.BatchNorm1d(
                    teacher_feature_dim, affine=False, eps=2e-5, momentum=0.9)

            # Load teacher checkpoint
            teacher_ckpt_path = kd_config.get('teacher_checkpoint', '')
            if teacher_ckpt_path and str(teacher_ckpt_path).strip().lower() not in ['', 'none', 'null']:
                self._load_teacher_weights(teacher_ckpt_path)

            # Freeze teacher completely
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            if self.teacher_feat_bn is not None:
                self.teacher_feat_bn.requires_grad_(False)
                self.teacher_feat_bn.eval()

            # Projection layer if dimensions differ
            self.kd_projection = None
            if feature_dim != teacher_feature_dim:
                self.kd_projection = nn.Linear(feature_dim, teacher_feature_dim, bias=False)

            # KD loss
            self.kd_criterion = KDLoss(
                loss_type=kd_config.get('loss_type', 'cosine'), 
                temperature=temperature,
                normalize_features=normalize_features)

        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.arc_fc = arc_fc
            if not arc_fc:
                self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
            else:
                self.fcs = nn.ModuleList([ArcFullyConnected(feature_dim, num_classes[k], s, m, is_pw, is_hard) for k in range(self.num_tasks)])

    def _load_teacher_weights(self, ckpt_path):
        """Load teacher backbone weights from a training checkpoint.

        Handles checkpoints saved by this framework (with 'state_dict' key
        and 'module.' prefix from DataParallel/DDP wrapping). Extracts only
        the backbone (basemodel) and optional batch norm (bn1d) weights.
        """
        import os
        from utils import log
        assert os.path.isfile(ckpt_path), \
            "Teacher checkpoint not found: {}".format(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Support both raw state_dict and wrapped checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Check if it's a full framework checkpoint (has 'basemodel.' keys)
        has_basemodel_prefix = any('basemodel.' in k for k in state_dict.keys())

        # Extract teacher backbone weights, stripping wrapper prefixes
        teacher_state = {}
        bn_state = {}
        for key, val in state_dict.items():
            # Strip 'module.' prefix from DataParallel/DDP
            clean_key = key.replace('module.', '', 1) if key.startswith('module.') else key
            
            if has_basemodel_prefix:
                if clean_key.startswith('basemodel.'):
                    teacher_state[clean_key.replace('basemodel.', '', 1)] = val
                elif clean_key.startswith('bn1d.'):
                    bn_state[clean_key.replace('bn1d.', '', 1)] = val
            else:
                # Assume it's a pure backbone checkpoint
                if clean_key.startswith('bn1d.'):
                    bn_state[clean_key.replace('bn1d.', '', 1)] = val
                else:
                    teacher_state[clean_key] = val

        missing, unexpected = self.teacher_model.load_state_dict(teacher_state, strict=False)
        if missing:
            log("Teacher backbone - missing keys: {}".format(missing))
        if unexpected:
            log("Teacher backbone - unexpected keys: {}".format(unexpected))

        if self.teacher_feat_bn is not None and bn_state:
            self.teacher_feat_bn.load_state_dict(bn_state, strict=False)

        log("=> Loaded teacher weights from '{}'".format(ckpt_path))

    def train(self, mode=True):
        """Override train() to keep teacher frozen in eval mode."""
        super(MultiTaskWithLoss, self).train(mode)
        if self.kd_enabled:
            self.teacher_model.eval()
            if self.teacher_feat_bn is not None:
                self.teacher_feat_bn.eval()
        return self

    def forward(self, input, target=None, slice_idx=None, extract_mode=False):

        feature = self.basemodel(input)
        if self.feat_bn:
            feature = self.bn1d(feature)
        if extract_mode:
            return feature
        else:
            assert feature.size(0) == target.size(0)
            assert(len(slice_idx) == self.num_tasks + 1)
            assert slice_idx[-1] == feature.size(0), "{} vs {}".format(slice_idx[-1], feature.size(0))
            if not self.arc_fc:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            else:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...],
                    target[slice_idx[k]:slice_idx[k+1]]) for k in range(self.num_tasks)]
            target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            task_losses = [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]

            # Compute KD loss if enabled
            kd_loss = None
            if self.kd_enabled:
                with torch.no_grad():
                    teacher_feat = self.teacher_model(input)
                    if self.teacher_feat_bn is not None:
                        teacher_feat = self.teacher_feat_bn(teacher_feat)

                student_feat = feature
                if self.kd_projection is not None:
                    student_feat = self.kd_projection(student_feat)

                kd_loss = self.kd_criterion(student_feat, teacher_feat.detach())

            return task_losses, kd_loss
