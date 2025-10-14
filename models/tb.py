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

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
    def forward(self,data):
        agent_pos = data['agent']['position'][:, :, : self.historical_steps]
        bs, A = agent_pos.shape[0:2]
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        agent_mask = data['agent']['valid_mask'][:, :, : self.historical_steps]
        agent_key_padding = ~(agent_mask.any(-1))

        polygon_mask = data["map"]["valid_mask"]
        map_key_padding = ~polygon_mask.any(-1)

        logits, probs = self.linear_scorer_layer(
            x_agent,
            x_polygon,
            agent_mask=agent_key_padding,
            lane_mask=map_key_padding,
        )

        return logits, probs
    def training_step(self, data, batch_idx):
        logits, probs  = self(data)
        target_lane_id = data['agent']['agent_lane_id_target'][:,:,self.historical_steps-1:].long()
        agent_mask = data['agent']['valid_mask'][:,:,self.historical_steps-1:]
        # 如果agent_mask为False，则对应的target_lane_id设为-100
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(~agent_mask, ignore_index)
        # test1 = target_lane_id.cpu().numpy()
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            probs.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * agent_mask.float()
        loss = (loss * valid).sum() / valid.sum()


    def validation_step(self, data, batch_idx):
        logits, probs  = self(data)
        target_lane_id = data['agent']['agent_lane_id_target'][:,:,self.historical_steps-1:].long()
        agent_mask = data['agent']['valid_mask'][:,:,self.historical_steps-1:]
        # 如果agent_mask为False，则对应的target_lane_id设为-100
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(~agent_mask, ignore_index)
        # test1 = target_lane_id.cpu().numpy()
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            probs.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * agent_mask.float()
        loss = (loss * valid).sum() / valid.sum()

        print(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=21)
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
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
