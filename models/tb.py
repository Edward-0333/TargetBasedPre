import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import AgentEncoder
from models import MapEncoder
from models import LinearScorerLayer
from models import FourierEmbedding
from models import TransformerEncoderLayer


class TB(pl.LightningModule):
    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 state_channel = 6,
                 dim=128,

                 polygon_channel=4,
                 history_channel=2,
                 history_steps=21,
                 encoder_depth=4,
                 decoder_depth=4,
                 drop_path=0.2,
                 state_attn_encoder=True,
                 state_dropout = 0.75,
                 use_ego_history=True,
                 **kwargs):
        super(TB, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )
        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )
        self.linear_scorer_layer = LinearScorerLayer(T=30, d=256)
        self.pos_emb = FourierEmbedding(2, dim, 64)
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
    def forward(self,data):
        agent_pos = data['agent']['position'][:, :, self.historical_steps]
        lane_center = data['map']['lane_center']
        pos = torch.cat((agent_pos, lane_center), dim=1)
        bs, A = agent_pos.shape[0:2]
        # 分别计算agent和polygon的特征
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        pos_embed = self.pos_emb(pos)
        # 合并agent和polygon的特征
        x = torch.cat([x_agent, x_polygon], dim=1)
        x = x + pos_embed
        history_agent_mask = data["agent"]["valid_mask"][:, :, : self.historical_steps]
        history_agent_key_padding = ~(history_agent_mask.any(-1))
        polygon_mask = data["map"]["valid_mask"]
        map_key_padding = ~polygon_mask.any(-1)
        key_padding_mask = torch.cat([history_agent_key_padding, map_key_padding], dim=-1)
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)[:, :A, :]

        agent_mask = data['agent']['valid_mask'][:, :,self.historical_steps:]
        agent_key_padding = ~agent_mask

        polygon_mask = data["map"]["valid_mask"]
        map_key_padding = ~polygon_mask.any(-1)

        logits, probs = self.linear_scorer_layer(
            x,
            x_polygon,
            agent_mask=agent_key_padding,
            lane_mask=map_key_padding,
        )

        return logits, probs
    def training_step(self, data, batch_idx):
        logits, probs  = self(data)
        target_lane_id = data['agent']['agent_lane_id_target'][:,:,self.historical_steps:].long()
        agent_mask = data['agent']['valid_mask'][:,:,self.historical_steps:]
        # 如果agent_mask为False，则对应的target_lane_id设为-100
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(~agent_mask, ignore_index)
        # 将target_lane_id中为-1的值也设为-100
        target_lane_id = target_lane_id.masked_fill(target_lane_id == -1, ignore_index)
        # test1 = target_lane_id.cpu().numpy()
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            logits.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * agent_mask.float()
        loss = (loss * valid).sum() / valid.sum()
        batch_size = data['agent']['position'].size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss


    def validation_step(self, data, batch_idx):
        logits, probs  = self(data)
        target_lane_id = data['agent']['agent_lane_id_target'][:,:,self.historical_steps:].long()
        agent_mask = data['agent']['valid_mask'][:,:,self.historical_steps:]
        # 如果agent_mask为False，则对应的target_lane_id设为-100
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(~agent_mask, ignore_index)
        # 将target_lane_id中为-1的值也设为-100
        target_lane_id = target_lane_id.masked_fill(target_lane_id == -1, ignore_index)
        # test1 = target_lane_id.cpu().numpy()
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            logits.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * agent_mask.float()
        loss = (loss * valid).sum() / valid.sum()
        batch_size = data['agent']['position'].size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_after_backward(self):
        # Monitor gradient norm to catch exploding/vanishing gradients early.
        grads = [
            param.grad.detach()
            for param in self.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        total_norm = torch.linalg.vector_norm(
            torch.stack([g.norm(2) for g in grads]), ord=2
        )
        self.log('grad_norm_l2', total_norm, on_step=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        # return optimizer
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.T_max,
            eta_min=0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
