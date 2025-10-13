import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def plot_test():
    data_path = Path(__file__).resolve().parent / 'train' / 'processed' / '3828.pt'
    with data_path.open('rb') as f:
        data = pickle.load(f)

    map_data = data['map']
    agent_data = data['agent']
    ego_cur_state = data['ego_cur_state']
    aaa = 9626043 in map_data['lane_ids']

    N, T = agent_data['position'].shape[0:2]
    agent_colors = plt.get_cmap('hsv', N)(range(N))
    marker_options = ['o', 's', '^', 'v', 'P', '*', 'X', 'D', '<', '>', 'H', '8']
    agent_markers = [marker_options[i % len(marker_options)] for i in range(N)]

    for t in range(T):
        fig, ax = plt.subplots(figsize=(20, 20))

        for i, centerline in enumerate(map_data['point_position_raw']):
            aaaaaaa = map_data['lane_ids'][i]
            if map_data['lane_type'][i] == 1:
                color = 'r'
            elif map_data['lane_type'][i] == 0:
                color = 'g'
            # if map_data['lane_ids'][i] == 9626043:
            #     color = 'orange'
            ax.plot(centerline[:, 0], centerline[:, 1], color=color, linewidth=0.8, alpha=0.5)

        now_pos = agent_data['position'][:, t]
        valid = agent_data['valid_mask'][:, t]
        now_lane = agent_data['lane_id'][:, t]
        for i in range(N):
            if not valid[i]:
                continue

            ax.scatter(
                now_pos[i][0],
                now_pos[i][1],
                # color=agent_colors[i],
                color='orange' if now_lane[i] == -1 else 'black',
                marker=agent_markers[i],
                s=5,
                alpha=0.8,
            )

        ax.scatter(ego_cur_state[0], ego_cur_state[1], s=10, color='red', label='Ego')


        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Scene 2645')
        # 保存图片到plot_test目录
        plt.savefig(Path(__file__).resolve().parent / 'plot_test' / f'scene_2645_t{t}.png', dpi=300)
        plt.close()






if __name__ == '__main__':
    plot_test()
