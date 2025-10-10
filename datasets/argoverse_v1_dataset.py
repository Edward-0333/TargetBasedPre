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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from collections.abc import Sequence
import os.path as osp

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm
from torch.utils.data import Dataset


class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        #如果处理好的文件不存在，则自动调用 process()
        if not all([os.path.exists(path) for path in self._processed_paths]):
            self.process()


    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
        print('处理数据')

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    @property
    def raw_paths(self) -> List[str]:
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]


    def __len__(self) -> int:
        return len(self._raw_file_names)

    def __getitem__(self, idx):
        return torch.load(self.processed_paths[idx])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)
    aaa = df['OBJECT_TYPE'] == 'AV'
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    ego_cur_state = av_df[19][['X', 'Y']].values

    past_ego_traj = av_df[:20][['X', 'Y']].values

    future_ego_traj = av_df[20:][['X', 'Y']].values
    timestamp_dfs = []
    for timestamp in timestamps:
        timestamp_df = df[df['TIMESTAMP'] == timestamp]
        # 去掉OBJECT_TYPE为AV的行
        timestamp_df = timestamp_df[timestamp_df['OBJECT_TYPE'] != 'AV']
        timestamp_dfs.append(timestamp_df)
    build_feature(am, av_df.obj, timestamp_dfs, 20)

    return {}

def get_agent_features(am, agent_df_list, present_idx):
    present_agent_df = agent_df_list[present_idx]
    N, T = len(present_agent_df), len(agent_df_list)
    position = np.zeros((N, T, 2), dtype=np.float64)
    valid_mask = np.zeros((N, T), dtype=np.bool_)

    if N == 0:
        return (
            {
                "position": position,
            },
            [],
            [],
        )
    agent_id = present_agent_df['TRACK_ID'].values
    city = present_agent_df['CITY_NAME'].values[0]
    for t, agent_df in enumerate(agent_df_list):
        agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(agent_df['TRACK_ID'].values)}
        for i, aid in enumerate(agent_id):
            if aid in agent_id_to_idx:
                temp_position = agent_df.iloc[agent_id_to_idx[aid]][['X', 'Y']].values
                position[i, t] = temp_position
                test = am.get_lane_ids_in_xy_bbox(temp_position[0], temp_position[1], city, 5)
                lane, conf, cl = am.get_nearest_centerline(np.array([temp_position[0], temp_position[1]]), city, visualize=False)

                valid_mask[i, t] = True
    print(1)
    pass

def build_feature(am, av_df, timestamp_dfs, present_idx):
    ego_cur_state = av_df.iloc[present_idx][['X', 'Y']].values

    get_agent_features(am, timestamp_dfs, present_idx)
    pass

def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        test = am.get_lane_segment_centerline(lane_id, city)
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)
    ddddddd = list(product(torch.arange(lane_vectors.size(0)), node_inds))
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    ddd = lane_positions.repeat_interleave(len(node_inds), dim=0)
    ccc = node_positions.repeat(lane_vectors.size(0), 1)
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
