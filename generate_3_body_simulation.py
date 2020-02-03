#! /usr/bin/env python3
import argparse
import telegram

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple
from matplotlib.animation import FFMpegWriter

Body = namedtuple('Body', ['m', 'x', 'y', 'dot_x', 'dot_y'])


def _create_random_body():
    m = 10 ** np.random.uniform(-0.5, 1.0)
    x0, y0 = np.random.uniform(-10.0, 10.0), np.random.uniform(-10.0, 10.0)
    dot_x0, dot_y0 = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

    return Body(m, x0, y0, dot_x0, dot_y0)


def _calc_force(b0, b1):
    # I'm sorry
    G = 1.0

    dx = b1.x - b0.x
    dy = b1.y - b0.y

    r = (dx ** 2 + dy ** 2) ** 0.5
    F = G * b0.m * b1.m / r ** 2

    if dx != 0:
        Fx_01 = F * dx / r
    else:
        Fx_01 = 0

    if dy != 0:
        Fy_01 = F * dy / r
    else:
        Fy_01 = 0

    return Fx_01, Fy_01


def _move_body(b, Fx, Fy, dt):
    dot_dot_x = Fx / b.m
    new_x = b.x + b.dot_x * dt + dot_dot_x * (dt ** 2) * 0.5
    new_dot_x = b.dot_x + dot_dot_x * dt

    dot_dot_y = Fy / b.m
    new_y = b.y + b.dot_y * dt + dot_dot_y * (dt ** 2) * 0.5
    new_dot_y = b.dot_y + dot_dot_y * dt

    new_b = Body(b.m, new_x, new_y, new_dot_x, new_dot_y)
    return new_b


def _calc_center_of_mass(bodies):
    sum_mass = np.sum([b.m for b in bodies])
    x = np.sum([b.m * b.x for b in bodies]) / sum_mass
    y = np.sum([b.m * b.y for b in bodies]) / sum_mass
    return x, y


def _calc_trajectories(b0, b1, b2, dt, n_steps):
    trj_0 = [[b0.x, b0.y]]
    trj_1 = [[b1.x, b1.y]]
    trj_2 = [[b2.x, b2.y]]
    trj_cm = [_calc_center_of_mass([b0, b1, b2])]

    for step in range(n_steps):
        # Forces
        Fx_01, Fy_01 = _calc_force(b0, b1)
        Fx_02, Fy_02 = _calc_force(b0, b2)
        Fx_12, Fy_12 = _calc_force(b1, b2)

        Fx_0 = Fx_01 + Fx_02
        Fy_0 = Fy_01 + Fy_02

        Fx_1 = - Fx_01 + Fx_12
        Fy_1 = - Fy_01 + Fy_12

        Fx_2 = - Fx_02 - Fx_12
        Fy_2 = - Fy_02 - Fy_12

        # Move everything
        b0 = _move_body(b0, Fx_0, Fy_0, dt)
        b1 = _move_body(b1, Fx_1, Fy_1, dt)
        b2 = _move_body(b2, Fx_2, Fy_2, dt)

        # Save positions
        trj_0.append([b0.x, b0.y])
        trj_1.append([b1.x, b1.y])
        trj_2.append([b2.x, b2.y])

        trj_cm.append(_calc_center_of_mass([b0, b1, b2]))

    trj_cm = np.array(trj_cm)

    trj_0 = np.array(trj_0) - trj_cm
    trj_1 = np.array(trj_1) - trj_cm
    trj_2 = np.array(trj_2) - trj_cm

    return trj_0, trj_1, trj_2


def _draw_animation(t0, t1, t2, b0, b1, b2, fps, duration):
    data_len = t0.shape[0]

    fig = plt.figure(figsize=(7, 7))

    # Finding correct borders for the graph
    x_min, x_max = np.min((t0[:, 0], t1[:, 0], t2[:, 0])), np.max((t0[:, 0], t1[:, 0], t2[:, 0]))
    y_min, y_max = np.min((t0[:, 1], t1[:, 1], t2[:, 1])), np.max((t0[:, 1], t1[:, 1], t2[:, 1]))

    x_mean = (x_min + x_max) * 0.5
    y_mean = (y_min + y_max) * 0.5

    graph_size = np.max(((x_max - x_min), (y_max - y_min)))
    plt.xlim(x_mean - graph_size * 0.55, x_mean + graph_size * 0.55)
    plt.ylim(y_mean - graph_size * 0.55, y_mean + graph_size * 0.55)

    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title('Gravitational interaction of 3 random bodies', fontsize=20)
    plt.grid()

    # Initial points
    plt.plot(t0[0, 0], t0[0, 1], marker='x', color='k')
    plt.plot(t1[0, 0], t1[0, 1], marker='x', color='navy')
    plt.plot(t2[0, 0], t2[0, 1], marker='x', color='lime')

    # Trajectories
    line_00, = plt.plot(t0[0, 0], t0[0, 1], color='k', alpha=0.2)
    line_01, = plt.plot(t0[0, 0], t0[0, 1], color='k', alpha=0.8)

    line_10, = plt.plot(t1[0, 0], t1[0, 1], color='navy', alpha=0.2)
    line_11, = plt.plot(t1[0, 0], t1[0, 1], color='navy', alpha=0.8)

    line_20, = plt.plot(t2[0, 0], t2[0, 1], color='lime', alpha=0.2)
    line_21, = plt.plot(t2[0, 0], t2[0, 1], color='lime', alpha=0.8)

    trace_len = int(t0.shape[0] * 0.1)

    # Final points
    point_0, = plt.plot(t0[0, 0], t0[0, 1], marker='o',
                        markersize=int(10 * b0.m ** (1.0/3)), color='k')
    point_1, = plt.plot(t1[0, 0], t1[0, 1], marker='o',
                        markersize=int(10 * b1.m ** (1.0/3)), color='k')
    point_2, = plt.plot(t2[0, 0], t2[0, 1], marker='o',
                        markersize=int(10 * b2.m ** (1.0/3)), color='k')

    n_frames = int(fps * duration)

    def aimation_step(n_frame):
        # Finding correct point to visualise
        i = int(data_len * float(n_frame + 1) / n_frames)

        if i > trace_len:
            line_01.set_data(t0[i - trace_len:i, 0], t0[i - trace_len:i, 1])
            line_11.set_data(t1[i - trace_len:i, 0], t1[i - trace_len:i, 1])
            line_21.set_data(t2[i - trace_len:i, 0], t2[i - trace_len:i, 1])

            line_00.set_data(t0[:i, 0], t0[:i, 1])
            line_10.set_data(t1[:i, 0], t1[:i, 1])
            line_20.set_data(t2[:i, 0], t2[:i, 1])
        else:
            line_01.set_data(t0[:i, 0], t0[:i, 1])
            line_11.set_data(t1[:i, 0], t1[:i, 1])
            line_21.set_data(t2[:i, 0], t2[:i, 1])

            line_00.set_data(t0[:i, 0], t0[:i, 1])
            line_10.set_data(t1[:i, 0], t1[:i, 1])
            line_20.set_data(t2[:i, 0], t2[:i, 1])

        point_0.set_data(t0[i - 1, 0], t0[i - 1, 1])
        point_1.set_data(t1[i - 1, 0], t1[i - 1, 1])
        point_2.set_data(t2[i - 1, 0], t2[i - 1, 1])

        return line_00, line_01, line_10, line_11, line_20, line_21, point_0, point_1, point_2

    trj_anim = animation.FuncAnimation(fig, aimation_step, blit=True, frames=range(n_frames))
    return trj_anim


def _downscale_trajectory(trj, min_x, min_y, dx, dy):
    trj_ds = (trj - np.array([[min_x, min_y]])) / np.array([[dx, dy]])
    trj_ds = trj_ds.astype(np.int32)
    return np.unique(trj_ds, axis=0)


def _draw_low_res_trajectory(downscaled_trj, n_bins):
    img = np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32)
    for p in downscaled_trj:
        img[p[0], p[1]] += 1
    return img


def _calc_interestness_score(t0, t1, t2, n_bins=30):
    min_x, max_x = np.min((t0[:, 0], t1[:, 0], t2[:, 0])), np.max((t0[:, 0], t1[:, 0], t2[:, 0]))
    min_y, max_y = np.min((t0[:, 1], t1[:, 1], t2[:, 1])), np.max((t0[:, 1], t1[:, 1], t2[:, 1]))

    dx = (max_x - min_x) / n_bins
    dy = (max_y - min_y) / n_bins

    all_trjs_img =\
        _draw_low_res_trajectory(_downscale_trajectory(t0, min_x, min_y, dx, dy), n_bins) +\
        _draw_low_res_trajectory(_downscale_trajectory(t1, min_x, min_y, dx, dy), n_bins) +\
        _draw_low_res_trajectory(_downscale_trajectory(t2, min_x, min_y, dx, dy), n_bins)

    score = np.where(all_trjs_img > 1)[0].size
    return score


def main(args):
    is_verbose = args.verbose

    draft_dt = args.dt * 10
    score = 0
    while(score < args.min_score or score > args.max_score):
        b0, b1, b2 = _create_random_body(), _create_random_body(), _create_random_body()
        t0, t1, t2 = _calc_trajectories(b0, b1, b2, draft_dt, int(args.duration / draft_dt))
        score = _calc_interestness_score(t0, t1, t2)
        if is_verbose:
            print ('Interestness score: {}'.format(score))

    t0, t1, t2 = _calc_trajectories(b0, b1, b2, args.dt, int(args.duration / args.dt))

    if is_verbose:
        print ('Drawing animation')
    new_animation = _draw_animation(t0, t1, t2, b0, b1, b2, args.fps, args.duration)

    writer = FFMpegWriter(fps=args.fps, metadata=dict(artist='robolamp'), bitrate=1800)
    new_animation.save("./simulation.mp4", writer=writer)

    if args.token is None:
        return
    if is_verbose:
        print('Uploading')

    info_msg = 'Initial states:\n'

    for b in [b0, b1, b2]:
        info_msg += '    m: {:.3f} x: {:.3f} y: {:.3f} vx: {:.3f} vy: {:.3f}\n'.format(
            b.m, b.x, b.y, b.dot_x, b.dot_y)
    info_msg += 'Interest-ness score: {}'.format(score)

    bot = telegram.Bot(args.token)
    bot.send_animation(chat_id=args.channel_name, animation=open('./simulation.mp4', 'rb'),
                       caption=info_msg, timeout=120)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Script which is generating random but "interesting" 3 bodies simulation')
    p.add_argument('--dt', type=float, default=0.001, help='Simulation step')
    p.add_argument('--fps', type=int, default=30, help='Frames per second')
    p.add_argument('--duration', type=float, default=60.0, help='Simulation duration')
    p.add_argument('--min-score', type=int, default=20, help='Minimal "interest" score')
    p.add_argument('--max-score', type=int, default=100, help='Maximal "interest" score')
    p.add_argument('-V', '--verbose', action='store_true', help='Print debug info')
    p.add_argument('-T', '--token', type=str, default=None, help='Token for your bot')
    p.add_argument('-N', '--channel-name', type=str, default=None, help='Channel name')

    args = p.parse_args()
    main(args)
