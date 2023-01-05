from noise_dataset import NoiseDataset
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset, DataLoader

import os
from os import listdir
from os.path import isfile, join
import time
import datetime

from tqdm import tqdm
import yaml
from pathlib import Path
import itertools
from multiprocessing.pool import ThreadPool
from multiprocessing import Manager
torch.multiprocessing.set_sharing_strategy('file_system')


''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

'''
Configure constants
'''
_OBJECT_TYPE = {
    "vehicle": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 3,
    "bus": 4,
    "static": 5,
    "background": 6,
    "construction": 7,
    "riderless_bicycle": 8,
    "unknown": 9,
}
_LANE_MARK_TYPE = {
    "DASH_SOLID_YELLOW": 0,
    "DASH_SOLID_WHITE": 1,
    "DASHED_WHITE": 2,
    "DASHED_YELLOW": 3,
    "DOUBLE_SOLID_YELLOW": 4,
    "DOUBLE_SOLID_WHITE": 5,
    "DOUBLE_DASH_YELLOW": 6,
    "DOUBLE_DASH_WHITE": 7,
    "SOLID_YELLOW": 8,
    "SOLID_WHITE": 9,
    "SOLID_DASH_WHITE": 10,
    "SOLID_DASH_YELLOW": 11,
    "SOLID_BLUE": 12,
    "NONE": 13,
    "UNKNOWN": 14,
}


class Argoverse2Dataset(Dataset):
    def __init__(
        self,
        raw_dir,
        raw_path,
        processed_dir,
        mode=None,
    ):
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.map.map_api import ArgoverseStaticMap
        from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory
        import av2.geometry.polyline_utils as polyline_utils
        import av2.geometry.interpolate as interpolate

        ''' av2 api
        '''
        self.TrackCategory = TrackCategory
        self.scenario_serialization = scenario_serialization
        self.avm = ArgoverseStaticMap
        self.polyline_utils = polyline_utils
        self.interpolate = interpolate

        ''' dataset
        '''
        self.processed_dir = processed_dir

        ''' load data path
        '''
        self.all_scenario_file_list = self.load_from_txt(raw_path)

        ''' Sample mode
        '''
        self.mode = mode

    def load_from_txt(self, raw_path):
        txt_file = open(raw_path, "r")
        file_list = txt_file.read().split("\n")[:-1]
        txt_file.close()
        file_list = [Path(f) for f in file_list]
        return file_list

    def __len__(self):
        return len(self.all_scenario_file_list)

    def downsample(self, polyline, desire_len):
        index = np.linspace(0, len(polyline)-1, desire_len).astype(int)
        return polyline[index]

    def __getitem__(self, idx):
        sample_path = os.path.join(self.processed_dir, f'data_{idx}.pt')

        try:
            #raise ExampleError("[Dataset] DEBUG STAGE!!!")
            sample = torch.load(sample_path)
        except:
            ''' Init Sample
            '''
            sample = {}
            ''' Load Scenario from parquet
            '''
            scenario_path = self.all_scenario_file_list[idx]

            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / \
                f"log_map_archive_{scenario_id}.json"

            scenario = self.scenario_serialization.load_argoverse_scenario_parquet(
                scenario_path)
            static_map = self.avm.from_json(static_map_path)
            tracks_df = self.scenario_serialization._convert_tracks_to_tabular_format(
                scenario.tracks)

            ''' Target Trajectory
            '''
            # [1] Load Target Track
            target_df = tracks_df[tracks_df['object_category'] == 3]
            target_id = target_df['track_id'].to_numpy()[0]
            target_traj = torch.as_tensor(
                target_df[['position_x', 'position_y']].to_numpy()).float()
            # [2] Rotation Normalization
            velocity = torch.as_tensor(
                target_df[['velocity_x', 'velocity_y']].to_numpy()).float()[int(config['constant']['obs_steps'])-1]
            heading = torch.as_tensor(target_df['heading'].to_numpy()).float()[
                int(config['constant']['obs_steps'])-1]
            theta = heading
            rot = torch.Tensor(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # [3] Translation Normalization
            orig = target_traj[int(config['constant']['obs_steps'])-1]

            ''' Dynamic FOV criterion
            '''
            speed = torch.norm(velocity)
            map_radius = min(
                166, max(torch.div(speed*int(config['constant']['pred_steps']), 10, rounding_mode="trunc"), 60))
            ''' Actors Trajectory
            '''
            target_x_track = []
            target_y_track = []
            neighbor_tracks = []
            for track in scenario.tracks:
                actor_timestep = torch.IntTensor(
                    [s.timestep for s in track.object_states])
                observed = torch.Tensor(
                    [s.observed for s in track.object_states])

                if (int(config['constant']['obs_steps'])-1) not in actor_timestep or (int(config['constant']['obs_steps'])-11) not in actor_timestep:
                    continue
                # add point heading
                actor_state = torch.Tensor(
                    [[object_state.position[0], object_state.position[1], object_state.velocity[0], object_state.velocity[1],
                        object_state.heading] for object_state in track.object_states if object_state.timestep < int(config['constant']['total_steps'])]
                )

                actor_state[:, :2] = actor_state[:, :2] - orig
                actor_state[:, :2] = actor_state[:, :2].mm(rot)  # position
                actor_state[:, 2:4] = actor_state[:, 2:4].mm(rot)  # velocity
                actor_state[:, 4] = actor_state[:, 4] - theta  # heading
                if (track.track_id == target_id):
                    target_y_track.append(
                        actor_state[int(config['constant']['obs_steps']):])
                    actor_state = torch.cat([actor_state, torch.empty(
                        int(config['constant']['total_steps']), 1).fill_(_OBJECT_TYPE[track.object_type])], -1)
                    target_x_track.append(
                        actor_state[:int(config['constant']['obs_steps'])])
                else:
                    # apply dynamic fov filtering
                    if (torch.norm(actor_state[-1, :2]) > map_radius):
                        continue
                    start = actor_timestep == int(
                        config['constant']['obs_steps'])-11
                    start_idx = torch.nonzero(start).item()
                    end = actor_timestep == int(
                        config['constant']['obs_steps'])-1
                    end_idx = torch.nonzero(end).item()

                    actor_state = torch.cat(
                        [actor_state[start_idx:end_idx+1], torch.empty(11, 1).fill_(_OBJECT_TYPE[track.object_type])], -1)
                    neighbor_tracks.append(actor_state)

            ''' Lane Centerline
            '''
            lane_segments = static_map.get_nearby_lane_segments(
                orig.cpu().detach().numpy(), map_radius)
            lane_centerlines = torch.Tensor(
                [list(static_map.get_lane_segment_centerline(s.id)) for s in lane_segments])
            lane_centerlines = lane_centerlines[..., :2] - orig
            lane_centerlines = lane_centerlines.reshape(
                -1, 2).mm(rot).reshape(-1, 10, 2)
            lane_polygon = torch.stack([torch.Tensor(self.polyline_utils.centerline_to_polygon(
                c.cpu().detach().numpy(), visualize=False)) for c in lane_centerlines])

            ''' Lane Boundary
            '''
            lane_boundary = []
            filtered_lane_segment_ids = [s.id for s in lane_segments]
            for idx in filtered_lane_segment_ids:
                # Left Boundary
                left_boundary_type = _LANE_MARK_TYPE[str(
                    static_map.vector_lane_segments[idx].left_mark_type).split('.')[1]]
                left_boundary = torch.Tensor([np.append(wpt.xyz[:2], left_boundary_type)
                                             for wpt in static_map.vector_lane_segments[idx].left_lane_boundary.waypoints])
                left_boundary = self.interpolate.interp_arc(10, left_boundary)
                # right Boundary
                right_boundary_type = _LANE_MARK_TYPE[str(
                    static_map.vector_lane_segments[idx].right_mark_type).split('.')[1]]
                right_boundary = torch.Tensor([np.append(wpt.xyz[:2], right_boundary_type)
                                              for wpt in static_map.vector_lane_segments[idx].right_lane_boundary.waypoints])
                right_boundary = self.interpolate.interp_arc(
                    10, right_boundary)
                # stack
                lane_boundary.append(right_boundary.to(torch.float32))
                lane_boundary.append(left_boundary.to(torch.float32))
            lane_boundary = torch.stack(lane_boundary)
            lane_boundary[..., :2] = lane_boundary[..., :2] - orig
            lane_boundary[..., :2] = lane_boundary[...,
                                                   :2].reshape(-1, 2)[:, :2].mm(rot).reshape(-1, 10, 2)

            ''' Crosswalk
            '''
            crosswalk = static_map.get_scenario_ped_crossings()
            crosswalk_polygon_list = [torch.as_tensor(
                s.polygon[:, :2], dtype=torch.float) for s in crosswalk]

            if len(crosswalk_polygon_list) > 0:
                crosswalk_polygon = torch.stack(
                    crosswalk_polygon_list).reshape(-1, 5, 2)
                crosswalk_polygon = crosswalk_polygon[..., :2] - orig
                crosswalk_polygon = crosswalk_polygon.reshape(
                    -1, 2).mm(rot).reshape(-1, 5, 2)
                # filter by dynamic fov
                crosswalk_polygon = [
                    cw_p for cw_p in crosswalk_polygon if torch.norm(cw_p[2, :2]) < map_radius]
                if len(crosswalk_polygon) > 0:
                    crosswalk_polygon = torch.stack(crosswalk_polygon)
                else:
                    crosswalk_polygon = torch.zeros(1, 5, 2)

            crosswalk_waypoint_list = [torch.stack([torch.as_tensor(
                edge, dtype=torch.float) for edge in s.get_edges_2d()]) for s in crosswalk]
            if len(crosswalk_waypoint_list) > 0:
                crosswalk_waypoint = torch.stack(
                    crosswalk_waypoint_list).reshape(-1, 2, 2)
                crosswalk_waypoint = crosswalk_waypoint[..., :2] - orig
                crosswalk_waypoint = crosswalk_waypoint.reshape(
                    -1, 2).mm(rot).reshape(-1, 2, 2)
                # filter by dynamic fov
                crosswalk_waypoint = [
                    cw_w for cw_w in crosswalk_waypoint if torch.norm(cw_w[0, :2]) < map_radius]
                if len(crosswalk_waypoint) > 0:
                    crosswalk_waypoint = torch.stack(crosswalk_waypoint)
                else:
                    crosswalk_waypoint = torch.zeros(1, 2, 2)

            ''' Stack One Scenario
            '''
            # sample['x'].shape -> [x,y,vx,vy,heading,type] (50, 6) -> (250)
            sample['x'] = torch.stack(
                target_x_track).reshape(-1, int(config['constant']['obs_steps']), 6)

            # sample['y'].shape -> [x,y,vx,vy, heading] (60, 5) -> (240)
            sample['y'] = torch.stack(
                target_y_track).reshape(-1, int(config['constant']['pred_steps']), 5)

            # sample['orig'].shape -> (2)
            sample['orig'] = orig

            # sample['rot'].shape -> (2, 2)
            sample['rot'] = rot

            # sample['neighbor_graph'].shape -> (N, 11, 6)
            sample['neighbor_graph'] = torch.zeros(1, 11, 6) if len(neighbor_tracks) == 0 else torch.stack(
                neighbor_tracks).reshape(-1, 11, 6)

            # sample['lane_graph'].shape -> (N, 10, 2)
            sample['lane_graph'] = torch.zeros(
                1, 10, 2) if lane_centerlines.shape[0] == 0 else lane_centerlines.reshape(-1, 10, 2)

            # sample['crossawalk_graph'].shape -> (N, 2, 2)
            sample['crosswalk_graph'] = torch.zeros(
                1, 2, 2) if len(crosswalk_waypoint_list) == 0 else crosswalk_waypoint.reshape(-1, 2, 2)

            # sample['crossawalk_polygon'].shape -> (N, 5, 2)
            sample['crosswalk_polygon'] = torch.zeros(1, 5, 2) if len(
                crosswalk_polygon_list) == 0 else crosswalk_polygon.reshape(-1, 5, 2)

            # sample['lane_polygon'].shape -> (N, 21, 2)
            sample['lane_polygon'] = lane_polygon

            # sample['lane_boundary'].shape -> (N, 10, 3)
            sample['lane_boundary'] = torch.zeros(
                1, 10, 3) if lane_boundary.shape[0] == 0 else lane_boundary.reshape(-1, 10, 3)

            # config
            sample['scenario_id'] = scenario_id
            sample['target_id'] = target_id
            sample['batch_id'] = idx

            torch.save(sample, sample_path)

        if self.mode == 'sampling':
            test_data = NoiseDataset(1)
            sample['noise_data'] = test_data[0].unsqueeze(0)

        return sample

# function called after collecting samples in batch


def argo_multi_agent_collate_fn(batch):
    elem = [k for k, v in batch[0].items()]

    def _collate_util(input_list, key):
        if isinstance(input_list[0], torch.Tensor):
            return torch.cat(input_list, 0)
        if key == 'lane_graph':
            return torch.cat([torch.cat(inp, 0) for inp in input_list], 0)
        return input_list

    def _get_object_type(input_list):
        return input_list[:, 0, -1]

    def _get_idxs(all_list):
        idx = 0
        neighbor_idx = 0
        lane_idx = 0
        lane_boundary_idx = 0
        crosswalk_idx = 0
        neighbor_idxs = []
        lane_idxs = []
        lane_boundary_idxs = []
        crosswalk_idxs = []
        for agent, neighbor, lane, lane_boundary, crosswalk in all_list:
            idx += agent.shape[0]
            neighbor_idx += neighbor.shape[0]
            lane_idx += lane.shape[0]
            lane_boundary_idx += lane_boundary.shape[0]
            crosswalk_idx += crosswalk.shape[0]

            neighbor_idxs.append(neighbor_idx)
            lane_idxs.append(lane_idx)
            lane_boundary_idxs.append(lane_boundary_idx)
            crosswalk_idxs.append(crosswalk_idx)
        return {
            'idxs': [i for i in range(1, idx+1)],
            'neighbor_idxs': neighbor_idxs,
            'lane_idxs': lane_idxs,
            'lane_boundary_idxs': lane_boundary_idxs,
            'crosswalk_idxs': crosswalk_idxs
        }

    def _get_attention_mask(x_idxs, graph_idxs):
        mask = torch.zeros(x_idxs[-1], graph_idxs[-1])
        a_prev = 0
        l_prev = 0
        for a_idx, l_idx in zip(x_idxs, graph_idxs):
            mask[a_prev:a_idx, l_prev:l_idx] = 1
            a_prev = a_idx
            l_prev = l_idx
        return mask

    collate = {key: _collate_util(
        [d[key] for d in batch], key) for key in elem}
    collate.update(_get_idxs(
        [(d['x'], d['neighbor_graph'], d['lane_graph'], d['lane_boundary'], d['crosswalk_graph']) for d in batch]))
    collate.update({'neighbor_mask': _get_attention_mask(
        collate['idxs'], collate['neighbor_idxs'])})
    collate.update({'lane_mask': _get_attention_mask(
        collate['idxs'], collate['lane_idxs'])})
    collate.update({'lane_boundary_mask': _get_attention_mask(
        collate['idxs'], collate['lane_boundary_idxs'])})
    collate.update({'crosswalk_mask': _get_attention_mask(
        collate['idxs'], collate['crosswalk_idxs'])})
    collate.update({'object_type': _get_object_type(collate['x'])})
    return collate


def preprocess_data():
    root = config['data']['root']
    val_dir = Path(root) / Path('raw/validation/')

    val_txt = config['data']['validation_txt']

    processed_val_dir = Path(root) / Path('processed/validation/')
    processed_val_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    dataset = Argoverse2Dataset(
        val_dir,
        val_txt,
        processed_val_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=24,
        collate_fn=argo_multi_agent_collate_fn,
        num_workers=0,
    )
    dataiter = tqdm(dataloader)
    for i, data in enumerate(dataiter):
        print('[info] - dataset.py - test looping')


if __name__ == '__main__':
    preprocess_data()
