import os
from pathlib import Path
import math
import datetime
import yaml

import numpy as np
from scipy import interpolate

import torch.nn.functional as F
import torch

import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colors

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

''' Dict
'''
_manuever_type = {
    0: 'left-turn',
    1: 'right-turn',
    2: 'u-turn',
    3: 'go straight',
    4: 'static',
}
_is_pass_intersection = {
    0: '',
    1: 'pass through intersection'
}
_lane_mark_type = {
    0: {"mark_type": "DASH_SOLID_YELLOW", "line_style": "-", "color": "yellow"},
    1: {"mark_type": "DASH_SOLID_WHITE", "line_style": "-", "color": "black"},
    2: {"mark_type": "DASHED_WHITE", "line_style": "--", "color": "black"},
    3: {"mark_type": "DASHED_YELLOW", "line_style": "--", "color": "yellow"},
    4: {"mark_type": "DOUBLE_SOLID_YELLOW", "line_style": "-", "color": "yellow"},
    5: {"mark_type": "DOUBLE_SOLID_WHITE", "line_style": "-", "color": "black"},
    6: {"mark_type": "DOUBLE_DASH_YELLOW", "line_style": "--", "color": "yellow"},
    7: {"mark_type": "DOUBLE_DASH_WHITE", "line_style": "--", "color": "black"},
    8: {"mark_type": "SOLID_YELLOW", "line_style": "-", "color": "yellow"},
    9: {"mark_type": "SOLID_WHITE", "line_style": "-", "color": "black"},
    10: {"mark_type": "SOLID_DASH_WHITE", "line_style": "-", "color": "black"},
    11: {"mark_type": "SOLID_DASH_YELLOW", "line_style": "-", "color": "yellow"},
    12: {"mark_type": "SOLID_BLUE", "line_style": "-", "color": "blue"},
    13: {"mark_type": "NONE", "line_style": ":", "color": "darkgrey"},
    14: {"mark_type": "UNKNOWN", "line_style": ":", "color": "darkgrey"},
}
_size = {
    "vehicle": [4.0, 2.0],
    "pedestrian": [0.7, 0.7],
    "motorcyclist": [2.0, 0.7],
    "cyclist": [2.0, 0.7],
    "bus": [7.0, 3.0],
    "static": [0.0, 0.0],
    "background": [0.0, 0.0],
    "construction": [0.0, 0.0],
    "riderless_bicycle": [2.0, 0.7],
    "unknown": [0.0, 0.0],
}


class VisualizeInterface:
    def __init__(self):
        pass

    def check_interrupt(self):
        os.system('clear')
        control = input("[Utils] visualize next? [y/n]: ")
        if (control == 'n'):
            exit()

    def prepare_argo_data(
        self,
        input_batch,
        batch_idx,
        dataset,
        pred_traj
    ):
        ''' meta data
        '''
        self.scenario_id = input_batch['scenario_id'][0]
        ''' trajectory-related
        '''
        self.x = input_batch['x'][0].detach().cpu().numpy()
        self.y = input_batch['y'][0].detach().cpu().numpy()

        self.pred_traj = pred_traj.reshape(-1, 60, 4).detach().cpu().numpy()
        xheading = input_batch['x'][...,
                                    4][-1].detach().cpu().numpy() * (180/3.14159)
        yheading = input_batch['y'][...,
                                    4][-1].detach().cpu().numpy() * (180/3.14159)
        self.heading = np.concatenate((xheading, yheading), axis=0)
        ''' map-related
        '''
        self.lane_graph = input_batch['lane_graph'].detach().cpu().numpy()
        self.crosswalk_polygon = input_batch['crosswalk_polygon'].detach(
        ).cpu().numpy()
        self.lane_polygon = input_batch['lane_polygon'].detach().cpu().numpy()
        self.lane_boundary = input_batch['lane_boundary'].detach(
        ).cpu().numpy()

        ''' neighbor-related
        '''
        self.neighbor_graph = input_batch['neighbor_graph'].detach(
        ).cpu().numpy()
        self.neighbor_tracks = self.prepare_neighbor(
            input_batch, batch_idx, dataset)

    def prepare_neighbor(self, input_batch, batch_idx, dataset):
        scenario_path = dataset.all_scenario_file_list[batch_idx]
        scenario = dataset.scenario_serialization.load_argoverse_scenario_parquet(
            scenario_path)
        tracks_df = dataset.scenario_serialization._convert_tracks_to_tabular_format(
            scenario.tracks)
        target_df = tracks_df[tracks_df['object_category'] == 3]
        target_id = target_df['track_id'].to_numpy()[0]
        target_heading = torch.as_tensor(target_df[['heading']].to_numpy()).float()[
            config['constant']['obs_steps']-1]

        orig = input_batch['orig'].detach().cpu()
        rot = input_batch['rot'].detach().cpu()
        # dynamic fov filter
        velocity = torch.as_tensor(
            target_df[['velocity_x', 'velocity_y']].to_numpy()).float()[config['constant']['obs_steps']-1]
        speed = torch.norm(velocity)
        map_radius = min(
            166, max(torch.div(speed*config['constant']['pred_steps'], 10, rounding_mode="trunc"), 60))

        neighbor_tracks = []
        index = 0
        for track in scenario.tracks:
            actor_timestep = torch.IntTensor(
                [s.timestep for s in track.object_states])
            observed = torch.Tensor([s.observed for s in track.object_states])

            if (config['constant']['obs_steps']-1) not in actor_timestep or (config['constant']['obs_steps']-11) not in actor_timestep:
                continue
            actor_state = torch.Tensor(
                [list(object_state.position+(object_state.heading, ))
                 for object_state in track.object_states if object_state.timestep < config['constant']['total_steps']]
            )

            actor_state[:, :2] = actor_state[:, :2] - orig
            actor_state[:, :2] = actor_state[:, :2].mm(rot)  # position
            actor_state[:, 2] = actor_state[:, 2] - target_heading
            if (track.track_id == target_id):
                edgecolor = config['plot']['agent']['fc']
                color = config['plot']['agent']['ec']
                target = True
            else:
                # apply dynamic fov filtering
                if (torch.norm(actor_state[-1, :2]) > map_radius):
                    continue
                start = actor_timestep == config['constant']['obs_steps']-11
                start_idx = torch.nonzero(start).item()
                end = actor_timestep == config['constant']['obs_steps']-1
                end_idx = torch.nonzero(end).item()

                edgecolor = config['plot']['neighbor']['fc']
                color = config['plot']['neighbor']['ec']
                target = False
                actor_state = actor_state[start_idx:end_idx+1]
                index += 1
            if not track.object_type in ['vehicle', 'pedestrian', 'motorcyclist', 'bus', 'cyclist']:
                continue
            state = {
                "l": _size[track.object_type][0],
                "w": _size[track.object_type][1],
                "actor_state": actor_state,
                "object_type": track.object_type,
                "color": color,
                "edgecolor": edgecolor,
                "target": target
            }
            neighbor_tracks.append(state)
        return neighbor_tracks

    def argo_matplot(
        self,
        title,
        lane_graph, lane_polygon,
        crosswalk_polygon,
        lane_boundary,
        neighbor_tracks,
        x, y,
        pred_traj,
    ):
        fig = plt.figure()
        axes1 = fig.add_subplot(111)
        ''' plot metadata
        '''
        title = f'scenario_id:{self.scenario_id}\n' + title
        axes1.set_title(title, fontsize=11)
        axes1.set_facecolor(config['plot']['bg']['color'])
        ''' plot lane
        '''
        for i in range(len(lane_graph)):
            color_lg = config['plot']['lane']['color']
            color_lp = config['plot']['lane_polygon']['color']
            axes1.plot(
                lane_graph[i, :, 0].T, lane_graph[i, :, 1].T,
                ':',
                color=color_lg,
                linewidth=config['plot']['lane']['lw'],
                zorder=998
            )
            ''' plot lane polygon
            '''
            axes1.plot(
                lane_polygon[i, :, 0].T, lane_polygon[i, :, 1].T,
                color=color_lp,
                linewidth=config['plot']['lane_polygon']['lw'],
                zorder=998
            )
        ''' plot crosswalk
        '''
        axes1.fill(
            crosswalk_polygon[..., 0].T,
            crosswalk_polygon[..., 1].T,
            color=config['plot']['crosswalk_polygon']['color'],
            zorder=998,
            alpha=0.5
        )
        ''' plot lane boundary
        '''

        for mark in lane_boundary:
            if _lane_mark_type[int(mark[-1, 2])]['mark_type'] in ['NONE', 'UNKNOWN']:
                continue
            axes1.plot(
                mark[..., 0],
                mark[..., 1],
                _lane_mark_type[int(mark[-1, 2])]['line_style'],
                color=_lane_mark_type[int(mark[-1, 2])]['color'],
                zorder=999,
                alpha=0.9,
            )
        ''' plot neighbor
        '''
        # Draw neighbors
        for index, neighbor in enumerate(neighbor_tracks):
            color = neighbor['color']
            edgecolor = neighbor['edgecolor']
            target = neighbor['target']
            object_type = neighbor['object_type']
            l = neighbor['l']
            w = neighbor['w']
            state = neighbor['actor_state'][:, :2]
            heading = neighbor['actor_state'][-1, 2]
            theta = heading*180 / math.pi
            axes1.plot(
                state[:, 0].T,
                state[:, 1].T,
                color=edgecolor,
                zorder=999,
                lw=config['plot']['traj']['lw'],
            )
            if object_type == 'pedestrian':
                axes1.plot(
                    state[-1, 0].T,
                    state[-1, 1].T,
                    'o',
                    color=color,
                    zorder=999,
                )
            else:
                if target:
                    continue
                else:
                    ts = axes1.transData
                    tr = matplotlib.transforms.Affine2D().rotate_deg_around(
                        state[-1, 0], state[-1, 1], theta)
                    t = tr+ts
                    axes1.plot(
                        state[-1, 0].T,
                        state[-1, 1].T,
                        'o',
                        color=edgecolor,
                        zorder=999,
                    )
                    axes1.add_patch(patches.Rectangle(
                        (state[-1, 0]-l/2, state[-1, 1]-w/2),
                        l, w,
                        facecolor=color,
                        edgecolor=edgecolor,
                        alpha=0.4,
                        zorder=98,
                        fill=True,
                        transform=t,
                        lw=config['plot']['traj']['lw'],
                    ))
        ''' plot history 
        '''
        axes1.plot(
            x[0], x[1],
            color=config['plot']['history']['color'],
            linewidth=config['plot']['history']['lw'],
            zorder=999,
            label='history'
        )
        ''' plot future groundtruth 
        '''
        axes1.plot(
            y[0], y[1],
            color=config['plot']['gt']['color'],
            linewidth=config['plot']['gt']['lw'],
            alpha=config['plot']['gt']['alpha'],
            zorder=999,
            label='future groundtruth'
        )
        ''' plot prediction 
        '''
        axes1.plot(
            pred_traj[0, :, 0], pred_traj[0, :, 1],
            color=config['plot']['traj']['color'],
            linewidth=config['plot']['traj']['lw'],
            alpha=config['plot']['traj']['alpha'],
            zorder=1000,
            label='prediction'
        )

        """ draw car model
        """
        #car_image = plt.imread('./assets/car.png')
        #offset_image = OffsetImage(car_image, zoom=0.05)
        #box = AnnotationBbox(offset_image, (0, 0), frameon=False, zorder=999)
        # axes1.add_artist(box)

        axes1.axis('equal')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys()
        )
        save_dir = Path('./images/')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")

        plt.savefig(f"{save_dir}/{fname}.png", dpi=600)
        plt.clf()

    def argo_forward(
        self,
        batch, batch_idx, dataset, traj
    ):
        # self.check_interrupt()
        self.prepare_argo_data(batch, batch_idx, dataset, traj)
        self.argo_matplot(
            "Diffusion of trajectory prediction",
            self.lane_graph, self.lane_polygon,
            self.crosswalk_polygon,
            self.lane_boundary,
            self.neighbor_tracks,
            np.transpose(self.x, (1, 0)), np.transpose(self.y, (1, 0)),
            self.pred_traj,
        )


def path_smoothing(waypoints, custom_num=120):
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x = x[okay]
    y = y[okay]
    tck, *rest = interpolate.splprep([x, y])
    u = np.linspace(0, 1, num=custom_num)
    smooth = interpolate.splev(u, tck)
    smooth = np.asarray(smooth)
    return smooth[:, 1::2]


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class IntentionLogger:
    def __init__(
        self,
        ckpt_name,
        matrix
    ):
        self.ckpt_name = ckpt_name
        self.matrix = matrix

    def save_to_txt(self):
        metric_path = f'./analysis_logger/{self.ckpt_name}_intention_matrix.txt'
        f = open(metric_path, 'w')
        for i in range(5):
            for j in range(5):
                count = self.matrix['count'][i][j]
                hit = self.matrix['hit'][i][j]
                ratio = hit/count
                f.write(
                    f'{hit}/{count}/{ratio*100:0.2f}% ')
            f.write('\n')
        f.close()


class PerformanceLogger:
    def __init__(
        self,
        ckpt_name,
        buckets
    ):
        self.ckpt_name = ckpt_name
        self.bucket_keys = ['all', '0-1', '1-2', '2-3', '3-4', '4-5', '5-']
        self.buckets = buckets
        self.metric_keys = ['ade_1', 'fde_1',
                            'ade_6_min', 'fde_6_min', 'b_fde_6_min']

    def calculate_ratio(self, numerator, denominator):
        ratio_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if denominator[i][j] != 0:
                    ratio_matrix[i][j] = numerator[i][j] / denominator[i][j]
        return ratio_matrix

    def extract_mean(self, outputs, key):
        return 0 if len(outputs) == 0 else np.array([output[key].item() for output in outputs]).mean()

    def save_to_txt(self):
        metric_path = f'./analysis_logger/{self.ckpt_name}_metric.txt'
        f = open(metric_path, 'w')
        for bucket_key in self.bucket_keys:
            bucket = self.buckets[bucket_key]
            id_list = bucket['id']
            metric = bucket['metric']
            matrix = bucket['matrix']

            ''' save metric
            '''
            f.write(f'------ b_fde_6_min range: {bucket_key} -----\n')
            for metric_key in self.metric_keys:
                metric_mean = self.extract_mean(metric, metric_key)
                f.write(f'{metric_key}: {metric_mean}\n')
            percentage = (len(id_list)/len(self.buckets['all']['id'])) * 100
            f.write(f'total_scenario: {len(id_list)} ({percentage:0.2f}%)\n')

            f.write('- matrix : describe the ratio of scenarios\n')
            ratio_matrix = self.calculate_ratio(
                matrix, self.buckets['all']['matrix'])
            for i in range(5):
                for j in range(5):
                    f.write(
                        f'{matrix[i][j]:0.2f}({ratio_matrix[i][j]*100:0.2f}%) ')
                f.write('\n')
            f.write('\n')
        f.close()

        '''
        Save ids
        '''
        for bucket_key in self.bucket_keys:
            bucket = self.buckets[bucket_key]
            id_list = bucket['id']
            id_path = f'./analysis_logger/{self.ckpt_name}_id_{bucket_key}.txt'
            f = open(id_path, 'w')
            for idx in id_list:
                f.write(f'{idx}\n')
            f.close()
