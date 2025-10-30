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

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.tb import TB
from torch.nn.utils.rnn import pad_sequence


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

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./lightning_logs/20251029-091154/version_0/checkpoints/epoch=33-step=98769.ckpt')
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = TB.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    trainer.validate(model, dataloader)
