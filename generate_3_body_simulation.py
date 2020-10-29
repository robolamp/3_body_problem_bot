#! /usr/bin/env python3
import argparse
import telegram

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple
from matplotlib.animation import FFMpegWriter
from scipy.integrate import solve_ivp

class Body(object):
    def __init__(self, m, x0, y0, dot_x0, dot_y0):
        self.m = m
        self.dot_x0 = dot_x0
        self.dot_y0 = dot_y0

        self.trj_x = np.array([x0])
        self.trj_y = np.array([y0])

    def reset_trj(self):
        self.trj_x = np.array([self.trj_x[0]])
        self.trj_y = np.array([self.trj_y[0]])


def _create_random_body():
    m = 10 ** np.random.uniform(-1.0, 0.0)
    x0, y0 = np.random.uniform(-10.0, 10.0), np.random.uniform(-10.0, 10.0)

    theta = np.random.uniform(0.0, 2.0 * np.pi)
    v0 = np.random.uniform(0.0, 1.0)

    dot_x0, dot_y0 = v0 * np.cos(theta), v0 * np.sin(theta)

    return Body(m, x0, y0, dot_x0, dot_y0)


def _calc_center_of_mass_trjs(bodies):
    sum_mass = np.sum([b.m for b in bodies])
    trj_x = np.sum(np.array([b.trj_x * b.m for b in bodies]), axis=0) / sum_mass
    trj_y = np.sum(np.array([b.trj_y * b.m for b in bodies]), axis=0) / sum_mass
    return trj_x, trj_y


def _calc_trajectories(bodies, dt, n_steps):
    for b in bodies:
        b.reset_trj()

    n_bodies = len(bodies)

    masses_vec = np.array([body.m for body in bodies])
    masses_matrix = np.array([masses_vec, ] * n_bodies)
    M = masses_matrix * masses_matrix.transpose()

    X_0 = np.array([body.trj_x[-1] for body in bodies])
    Y_0 = np.array([body.trj_y[-1] for body in bodies])

    dot_X_0 = np.array([body.dot_x0 for body in bodies])
    dot_Y_0 = np.array([body.dot_y0 for body in bodies])

    timestamps = np.linspace(0, dt * n_steps, n_steps)

    init_state = np.concatenate([X_0, Y_0, dot_X_0, dot_Y_0])

    def iter(t, state):
        X = state[0:n_bodies]
        Y = state[n_bodies:2 *n_bodies]

        dot_X = state[2 * n_bodies:3 *n_bodies]
        dot_Y = state[3 * n_bodies:]

        X_matrix = np.array([X, ] * n_bodies)
        Y_matrix = np.array([Y, ] * n_bodies)

        dist_X = X_matrix - X_matrix.transpose()
        dist_Y = Y_matrix - Y_matrix.transpose()

        R = np.sqrt(dist_X ** 2 + dist_Y ** 2)

        F_x = 1. * M * dist_X / (R ** 3)
        F_x[np.isnan(F_x)] = 0.
        F_x[np.isinf(F_x)] = 0.
        dot_dot_X = np.sum(F_x, axis=1) / masses_vec

        F_y = 1. * M * dist_Y / (R ** 3)
        F_y[np.isnan(F_y)] = 0.
        F_y[np.isinf(F_y)] = 0.
        dot_dot_Y = np.sum(F_y, axis=1) / masses_vec

        return np.concatenate([dot_X, dot_Y, dot_dot_X, dot_dot_Y])

    sol = solve_ivp(iter, [0, dt * n_steps], init_state, t_eval=timestamps, method='DOP853')
    for i in range(n_bodies):
            bodies[i].trj_x = sol.y[i, :]
            bodies[i].trj_y = sol.y[n_bodies + i, :]

    trj_cm_x, trj_cm_y = _calc_center_of_mass_trjs(bodies)

    trjs = np.array([np.array([
        b.trj_x - trj_cm_x, b.trj_y - trj_cm_y]).transpose() for b in bodies])
    return trjs


def _draw_animation(trjs, bodies, fps, duration, time_scale, track_indices):
    n_bodies = len(bodies)
    colors = ['k', 'navy', 'lime']
    colors = (colors * (n_bodies // len(colors) + 1))[:n_bodies]

    data_len = trjs.shape[1]

    fig = plt.figure(figsize=(7, 7))

    # Finding correct borders for the graph
    x_min, x_max = np.min((trjs[track_indices, :, 0])), np.max((trjs[track_indices, :, 0]))
    y_min, y_max = np.min((trjs[track_indices, :, 1])), np.max((trjs[track_indices, :, 1]))

    x_mean = (x_min + x_max) * 0.5
    y_mean = (y_min + y_max) * 0.5

    graph_size = np.max(((x_max - x_min), (y_max - y_min)))
    plt.xlim(x_mean - graph_size * 0.55, x_mean + graph_size * 0.55)
    plt.ylim(y_mean - graph_size * 0.55, y_mean + graph_size * 0.55)

    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title('Gravitational interaction of {} random bodies\ntime x{}'.format(
              n_bodies, time_scale), fontsize=15)
    plt.grid()

    # Initial points
    for i in range(n_bodies):
        plt.plot(trjs[i, 0, 0], trjs[i, 0, 1], marker='x', color=colors[i])

    # Trajectories and final points
    points = []
    lines_full = []
    lines_fin = []
    for i in range(n_bodies):
        line_full, = plt.plot(trjs[i, 0, 0], trjs[i, 0, 1],
                             alpha=0.2, color=colors[i])
        line_fin, = plt.plot(trjs[i, 0, 0], trjs[i, 0, 1],
                             alpha=0.8, color=colors[i])
        point, = plt.plot(trjs[i, 0, 0], trjs[i, 0, 1], marker='o', color='k',
                          markersize=int(10 * bodies[i].m ** (1.0/3)))
        lines_full.append(line_full)
        lines_fin.append(line_fin)
        points.append(point)

    trace_len = int(trjs.shape[1] * 0.1)

    n_frames = int(fps * duration)

    def aimation_step(n_frame):
        # Finding correct point to visualise
        i = int(data_len * float(n_frame + 1) / n_frames)

        if i > trace_len:
            for j in range(n_bodies):
                lines_fin[j].set_data(trjs[j, i - trace_len:i, 0],
                                      trjs[j, i - trace_len:i, 1])
                lines_full[j].set_data(trjs[j, :i, 0], trjs[j, :i, 1])
        else:
            for j in range(n_bodies):
                lines_fin[j].set_data(trjs[j, :i, 0], trjs[j, :i, 1])
                lines_full[j].set_data(trjs[j, :i, 0], trjs[j, :i, 1])

        for j in range(n_bodies):
            points[j].set_data(trjs[j, i - 1, 0], trjs[j, i - 1, 1])

        return lines_full + lines_fin + points

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


def _calc_interestness_score(trajectories, n_bins=30):
    min_x, max_x = np.min(trajectories[:, :, 0]), np.max(trajectories[:, :, 0])
    min_y, max_y = np.min(trajectories[:, :, 1]), np.max(trajectories[:, :, 1])

    field_size = np.max(((max_x - min_x), (max_y - min_y)))
    step = field_size / n_bins

    all_trjs_img = np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32)
    for i in range(trajectories.shape[0]):
        downscaled_trj = _downscale_trajectory(
            trajectories[i, :, :], min_x, min_y, step, step)
        all_trjs_img += _draw_low_res_trajectory(downscaled_trj, n_bins)

    score = np.where(all_trjs_img > 1)[0].size
    return score, field_size

def _calc_mass_in_field(bodies, mass_keep_in_field):
    if mass_keep_in_field >= 1.0:
        return list(range(len(bodies)))

    mass_list = [{'mass': body.m, 'index': i} for body, i in zip(bodies, range(len(bodies)))]
    
    mass_list = sorted(mass_list, key=lambda mass: -mass['mass'])


    mass_sum = 0
    for mass in mass_list:
        mass_sum += mass['mass']
        mass.update({'mass_sum': mass_sum})

    
    track_indices = []
    for mass in mass_list:
        sum_mass_share = mass['mass_sum'] / mass_sum

        print('sum_mass_share: {}'.format(sum_mass_share))
        if sum_mass_share < mass_keep_in_field:
            track_indices.append(mass['index'])

    return sorted(track_indices)


def main(args):
    is_verbose = args.verbose

    draft_dt = args.dt * 10
    score = 0
    field_size = args.max_field_size * 2
    while((score <= args.min_score or score > args.max_score) or field_size > args.max_field_size):
        bodies = [_create_random_body() for _ in range(args.n_bodies)]
        trjs = _calc_trajectories(bodies, draft_dt, int(args.duration / draft_dt))
        track_indices = _calc_mass_in_field(bodies, args.mass_keep_in_field)
        score, field_size = _calc_interestness_score(trjs[track_indices])

        if is_verbose:
            print ('Interestness score: {} Size: {}'.format(score, field_size))

    if is_verbose:
        print ('Final simulation')
    trjs = _calc_trajectories(bodies, args.dt, int(args.duration / args.dt))

    if is_verbose:
        print ('Drawing animation')

    time_scale = int(args.duration / args.video_duration)
    trjs = trjs[:, ::time_scale, :]

    new_animation = _draw_animation(trjs, bodies, args.fps, args.video_duration,
                                    time_scale, track_indices)

    writer = FFMpegWriter(fps=args.fps, metadata=dict(artist='robolamp'), bitrate=1800)
    new_animation.save("./simulation.mp4", writer=writer)

    if args.token is None:
        return
    if is_verbose:
        print('Uploading')

    info_msg = 'Initial states:\n'

    for body in bodies:
        info_msg += '    m: {:.3f} x: {:.3f} y: {:.3f} vx: {:.3f} vy: {:.3f}\n'.format(
            body.m, body.trj_x[0], body.trj_y[0], body.dot_x0, body.dot_y0)
    info_msg += 'Interest-ness score: {}'.format(score)

    bot = telegram.Bot(args.token)
    bot.send_video(chat_id=args.channel_name, video=open('./simulation.mp4', 'rb'),
                   caption=info_msg, timeout=120)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Script which is generating random but "interesting" N bodies simulation')
    p.add_argument('--dt', type=float, default=0.001, help='Simulation step')
    p.add_argument('--fps', type=int, default=30, help='Frames per second')
    p.add_argument('--n-bodies', type=int, default=3, help='Number of bodies')
    p.add_argument('--duration', type=float, default=30.0, help='Simulation duration')
    p.add_argument('--video-duration', type=float, default=30.0, help='Video duration')
    p.add_argument('--min-score', type=int, default=20, help='Minimal "interest" score')
    p.add_argument('--max-score', type=int, default=75, help='Maximal "interest" score')
    p.add_argument('--max-field-size', type=float, default=50.0, help='Maximal field size')
    p.add_argument('--mass-keep-in-field', type=float, default=1.0, help='Share of mass to keep in field')
    p.add_argument('-V', '--verbose', action='store_true', help='Print debug info')
    p.add_argument('-T', '--token', type=str, default=None, help='Token for your bot')
    p.add_argument('-N', '--channel-name', type=str, default=None, help='Channel name')

    args = p.parse_args()
    main(args)
