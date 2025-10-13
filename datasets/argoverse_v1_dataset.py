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
import math
import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from .utils import sample_discrete_path


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
        os.makedirs(self.processed_dir, exist_ok=True)
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            # aaa = raw_path.split('/')[-1]
            # if raw_path.split('/')[-1] != '3828.csv':
            #     continue
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            idx = os.path.splitext(os.path.basename(raw_path))[0]
            processed_path = os.path.join(self.processed_dir, f'{idx}.pt')
            torch.save(kwargs, processed_path)

            # with open(processed_path, 'wb') as f:
            #     pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(1)
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
        # with open(self.processed_paths[idx], 'rb') as f:
        #     return pickle.load(f)
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
        # timestamp_df = timestamp_df[timestamp_df['OBJECT_TYPE'] != 'AV']
        timestamp_dfs.append(timestamp_df)
    data = build_feature(am, av_df.obj, timestamp_dfs, 20, radius)

    return data

def get_agent_features(am, agent_df_list, present_idx):
    present_agent_df = agent_df_list[present_idx]
    N, T = len(present_agent_df), len(agent_df_list)
    position = np.zeros((N, T, 2), dtype=np.float64)
    valid_mask = np.zeros((N, T), dtype=np.bool_)
    lane_id = np.zeros((N, T), dtype=np.int64)

    if N == 0:
        return (
            {
                "position": position,
            }
        )
    agent_id = present_agent_df['TRACK_ID'].values
    city = present_agent_df['CITY_NAME'].values[0]
    agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(present_agent_df['TRACK_ID'].values)}

    for t, agent_df in enumerate(agent_df_list):
        agent_id_temp = agent_df['TRACK_ID'].values
        for _, aid in enumerate(agent_id_temp):  # i不对!！!
            if aid in agent_id_to_idx:
                idx = agent_id_to_idx[aid]
                temp_position = agent_df[agent_df['TRACK_ID'] == aid][['X', 'Y']].values[0]
                position[idx, t] = temp_position
                # test = am.get_lane_ids_in_xy_bbox(temp_position[0], temp_position[1], city, 5)
                lane, conf, cl = am.get_nearest_centerline(np.array([temp_position[0], temp_position[1]]), city, visualize=False)
                assert lane is not None, "lane is None"
                # if conf < 0.5:
                #     lane_id[idx, t] = -1
                # else:
                lane_id[idx, t] = int(lane.id)
                # if lane.id == 9626043:
                #     print(1)


                valid_mask[idx, t] = True

    agent_features = {
        "position": position,
        "valid_mask": valid_mask,
        "lane_id": lane_id,
    }
    return agent_features

def get_map_features(am, agent_df_list, present_idx, radius, sample_points=20):
    present_agent_df = agent_df_list[present_idx]
    node_positions = np.stack([present_agent_df['X'].values, present_agent_df['Y'].values], axis=-1)
    lane_ids = set()
    city = present_agent_df['CITY_NAME'].values[0]

    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    M, P = len(lane_ids), sample_points
    point_position_raw = np.zeros((M, P + 1, 2), dtype=np.float64)
    point_position = np.zeros((M, P, 2), dtype=np.float64)
    point_vector = np.zeros((M, P, 2), dtype=np.float64)
    lane_type = np.zeros((M,), dtype=np.int64)
    lane_direction = np.zeros((M,), dtype=np.float64)
    lane_control = np.zeros((M,), dtype=np.int64)
    for lane_id in lane_ids:
        idx = list(lane_ids).index(lane_id)
        centerline_point = am.get_lane_segment_centerline(lane_id, city)[:, : 2]
        centerline_point = sample_discrete_path(centerline_point, sample_points + 1)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)

        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')

        point_position_raw[idx] = centerline_point
        point_position[idx] = centerline_point[:-1]
        point_vector[idx] = centerline_point[1:] - centerline_point[:-1]
        lane_type[idx] = int(is_intersection)
        lane_direction[idx] = turn_direction
        lane_control[idx] = int(traffic_control)

    map_features = {
        "point_position_raw": point_position_raw,
        "point_position": point_position,
        "point_vector": point_vector,
        "lane_type": lane_type,
        "lane_direction": lane_direction,
        "lane_control": lane_control,
        "lane_ids": list(lane_ids),
    }
    return map_features

def get_ego_feature(am, av_df, present_idx):
    pass

def normalize(data, av_df, present_idx):
    ego_cur_state = av_df.iloc[present_idx][['X', 'Y']].values
    ego_pre_state = av_df.iloc[present_idx - 1][['X', 'Y']].values
    av_heading_vector = ego_cur_state - ego_pre_state
    theta = math.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]],
                          dtype=np.float64)

    data["map"]['point_position_raw'] = np.matmul(data["map"]['point_position_raw'] - ego_cur_state, rotate_mat)
    data['map']['point_position'] = np.array(np.matmul(data["map"]['point_position'] - ego_cur_state, rotate_mat),dtype=np.float64)
    data['map']['point_vector'] = np.array(np.matmul(data["map"]['point_vector'], rotate_mat),dtype=np.float64)
    data["ego_cur_state"] = np.array([0.0, 0.0])
    data['agent']['position'] = np.array(np.matmul(data['agent']['position'] - ego_cur_state, rotate_mat),dtype=np.float64)

    return data

def build_feature(am, av_df, timestamp_dfs, present_idx, radius):
    ego_cur_state = av_df.iloc[present_idx][['X', 'Y']].values
    data = {}

    agent_features = get_agent_features(am, timestamp_dfs, present_idx)
    map_features = get_map_features(am, timestamp_dfs, present_idx, radius)
    # find_candidate_lanes(am, agent_features, map_features, timestamp_dfs)
    agent_features['agent_lane_id_target'] = target_to_idx(agent_features, map_features)
    data["map"] = map_features
    data["ego_cur_state"] = ego_cur_state
    data["agent"] = agent_features
    data = normalize(data, av_df, present_idx)
    new_data = np_to_torch(data)
    print(1)
    return new_data

def np_to_torch(data):
    new_data = {}
    map_feature = {}
    map_feature['point_position'] = torch.from_numpy(data['map']['point_position'])
    map_feature['point_vector'] = torch.from_numpy(data['map']['point_vector'])
    map_feature['lane_type'] = torch.from_numpy(data['map']['lane_type'])
    map_feature['lane_direction'] = torch.from_numpy(data['map']['lane_direction'])
    map_feature['lane_control'] = torch.from_numpy(data['map']['lane_control'])
    new_data['map'] = map_feature
    agent_features = {}
    agent_features['position'] = torch.from_numpy(data['agent']['position'])
    agent_features['valid_mask'] = torch.from_numpy(data['agent']['valid_mask'])
    agent_features['agent_lane_id_target'] = torch.from_numpy(data['agent']['agent_lane_id_target'])
    new_data['agent'] = agent_features
    return new_data


def target_to_idx(agent_features, map_features):
    all_lane_id = map_features["lane_ids"]
    dict_all_lane_id = {int(lid): idx for idx, lid in enumerate(all_lane_id) if int(lid) > 0}
    agent_lane_id_target = np.zeros_like(agent_features["lane_id"])
    N, T = agent_features["lane_id"].shape
    for i in range(N):
        for t in range(T):
            lid = agent_features["lane_id"][i, t]
            if lid in dict_all_lane_id:
                agent_lane_id_target[i, t] = dict_all_lane_id[lid]
            else:
                agent_lane_id_target[i, t] = -1
    return agent_lane_id_target

def find_candidate_lanes(am: ArgoverseMap,agent_features: Dict, map_features: Dict, timestamp_dfs):
    # 好像不需要mask，因为在测试时也没有mask
    city_name = timestamp_dfs[0].iloc[0]['CITY_NAME']
    all_lane_id = map_features["lane_ids"]
    dict_all_lane_id = {int(lid): idx for idx, lid in enumerate(all_lane_id) if int(lid) > 0}
    # cand_mask: [N, T, K] - 候选车道有效的 mask (1:有效，0:无效)。
    K = 256
    N = agent_features['position'].shape[0]
    T = agent_features['position'].shape[1]# [:,hist_steps:].shape[1]
    cand_mask = np.zeros((N, T, K), dtype=bool)
    for i in range(N):
        for t in range(T):
            now_id = agent_features['lane_id'][i, t]
            adjacent_ids = am.get_lane_segment_adjacent_ids(now_id, city_name)
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
