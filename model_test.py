# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pickle
from datasets import ArgoverseV1Dataset
from models.tb import TB
from torch.nn.utils.rnn import pad_sequence
import torch
import matplotlib.pyplot as plt

def collate_fn(batch):
    pad_keys = ["agent", "map"]

    batch_data = {}
    for key in pad_keys:
        batch_data[key] = {
            k: pad_sequence(
                [f[key][k] for f in batch], batch_first=True
            )
            for k in batch[0][key].keys()
        }
    return batch_data

def plot_test(file_path):
    pass

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./lightning_logs/20251029-091154/version_0/checkpoints/epoch=33-step=98769.ckpt')
    args = parser.parse_args()

    model = TB.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='test', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    for batch in dataloader:
        N_index = 4

        _, prob = model.forward(batch)
        # plot_test('./datasets/test/processed/3.pt')
        raw_path = Path('./datasets/test/processed/3.pt')
        with raw_path.open('rb') as f:
            data = torch.load(f)
        map_point = data['map']['point_position'].numpy()
        lane_type = data['map']['lane_type'].numpy()
        agent_position = data['agent']['position'].numpy()
        agent_valid=data['agent']['valid_mask'].numpy()[:,20:]
        now_pos = agent_position[:, 20:]
        lane_num = map_point.shape[0]


        # 绘制第N个车辆的预测轨迹
        prob = prob[0][N_index].detach().cpu().numpy()
        T = prob.shape[0]
        cmap = plt.get_cmap('Reds')
        for j in range(T):

            # 绘制地图
            for i in range(lane_num):
                lane_point = map_point[i]
                if lane_type[i] == 1:
                    color = 'r'
                else:
                    color = 'g'
                plt.plot(lane_point[:, 0], lane_point[:, 1], color=color, linewidth=0.8, alpha=0.5)

            # 绘制车辆位置
            N = now_pos.shape[0]
            for i in range(N):
                if agent_valid[i][j]==0:
                    continue
                plt.scatter(
                    now_pos[i][j][0],
                    now_pos[i][j][1],
                    color='black',
                    marker='o',
                    s=5,
                    alpha=0.8,
                )
            plt.scatter(now_pos[N_index][j][0], now_pos[N_index][j][1], s=10, color='blue', label='Ego')
            prob_j = prob[j]
            nonzero_idx = prob_j.nonzero()[0]
            if nonzero_idx.size:
                lane_scores = prob_j[nonzero_idx]
                max_score = lane_scores.max()
                if max_score <= 0:
                    norm_scores = lane_scores.copy()
                    norm_scores.fill(1.0)
                else:
                    norm_scores = lane_scores / max_score
                for lane_idx, norm_score in zip(nonzero_idx, norm_scores):
                    lane_point = map_point[lane_idx]
                    plt.plot(
                        lane_point[:, 0],
                        lane_point[:, 1],
                        color=cmap(float(norm_score)),
                        linewidth=1.0,
                    )
            plt.savefig(f'./model_test_png/{j}.png', dpi=500)
            plt.close()
            print(1)
