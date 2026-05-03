import torch
import torch.nn as nn
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


class KDCosineEmbeddingLoss(nn.Module):
    """Knowledge distillation loss using cosine similarity on embeddings.

    Computes: 1 - cos_sim(student_embedding, teacher_embedding)
    Optionally scales embeddings by a temperature before computing similarity.
    """
    def __init__(self, temperature=1.0):
        super(KDCosineEmbeddingLoss, self).__init__()
        self.temperature = temperature
        # target=+1 means we want the embeddings to be similar
        self.criterion = nn.CosineEmbeddingLoss(reduction='mean')

    def forward(self, student_feat, teacher_feat):
        if self.temperature != 1.0:
            student_feat = student_feat / self.temperature
            teacher_feat = teacher_feat / self.temperature
        # target = +1 for all pairs (we want them aligned)
        target = torch.ones(student_feat.size(0), device=student_feat.device)
        return self.criterion(student_feat, teacher_feat, target)


class MultiTaskWithLoss(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, spatial_size,
                 arc_fc=False, feat_bn=False, s=64, m=0.5, is_pw=True,
                 is_hard=False, loss_type='crossentropy', kd_config=None):
        super(MultiTaskWithLoss, self).__init__()
        self.feat_bn = feat_bn
        self.basemodel = backbones.__dict__[backbone](feature_dim=feature_dim, spatial_size=spatial_size)
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
            temperature = kd_config.get('temperature', 1.0)

            # Build frozen teacher backbone
            self.teacher_model = backbones.__dict__[teacher_backbone](
                feature_dim=teacher_feature_dim, spatial_size=teacher_input_size)
            self.teacher_feat_bn = None
            if feat_bn:
                self.teacher_feat_bn = nn.BatchNorm1d(
                    teacher_feature_dim, affine=False, eps=2e-5, momentum=0.9)

            # Load teacher checkpoint
            teacher_ckpt_path = kd_config['teacher_checkpoint']
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
            self.kd_criterion = KDCosineEmbeddingLoss(temperature=temperature)

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

        # Extract teacher backbone weights, stripping wrapper prefixes
        teacher_state = {}
        bn_state = {}
        for key, val in state_dict.items():
            # Strip 'module.' prefix from DataParallel/DDP
            clean_key = key.replace('module.', '', 1) if key.startswith('module.') else key
            if clean_key.startswith('basemodel.'):
                teacher_state[clean_key.replace('basemodel.', '', 1)] = val
            elif clean_key.startswith('bn1d.'):
                bn_state[clean_key.replace('bn1d.', '', 1)] = val

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
