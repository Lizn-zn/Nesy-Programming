import argparse
import os
import platform
import random
import torch
import torch.optim as optimizer
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from scipy.interpolate import make_interp_spline


from smt_solver import maxsat_solver, maxsat_solver_gpt
from joblib import Parallel, delayed

comp_list = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
size = 10
device = "cuda:0"

def get_point_mat(grids):
    # boundary and dimensions
    xmin, ymin, xmax, ymax = int(np.min(grids[0]['bx'])), int(np.min(grids[0]['by'])), \
        int(np.max(grids[0]['bx'])), int(np.max(grids[0]['by']))
    width, height = xmax - xmin, ymax - ymin

    sp = torch.zeros((len(grids), 10, self.cfg.OUTPUT_WIDTH)).cuda()
    tp = torch.zeros((len(grids), 10, self.cfg.OUTPUT_WIDTH)).cuda()

    for i, grid in enumerate(grids):
        sx_idx, sy_idx = max(0, min(round(grid['sx'] - xmin), width - 1)), max(0, min(height - 1 - round(
            grid['sy'] - ymin), height - 1))
        tx_idx, ty_idx = max(0, min(round(grid['tx'] - xmin), width - 1)), max(0, min(height - 1 - round(
            grid['ty'] - ymin), height - 1))
        sp[i, sy_idx, sx_idx] = 1.
        tp[i, ty_idx, tx_idx] = 1.

    return sp, tp
    
def add_argue(parser):
    parser.add_argument("--config-file", default="./data/kitti-smoke/configs/smoke_gn_vector.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    plat = platform.system().lower()
    if plat == 'windows':
        port = 100
    else:
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def save(net, logic, file_name, epoch=0):
    state = {
            'net': net.state_dict(),
            'logic': logic,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net


def eval_grid(model, dataset="kitti", device="cuda:0"):
    """evaluate the model that outputs the obstalce matrix and then """
    pass

def get_point_mat(grids):
    # boundary and dimensions
    xmin, ymin, xmax, ymax = int(np.min(grids[0]['bx'])), int(np.min(grids[0]['by'])), \
        int(np.max(grids[0]['bx'])), int(np.max(grids[0]['by']))
    width, height = xmax - xmin, ymax - ymin

    sp = torch.zeros((len(grids), 10, 10)).to(device)
    tp = torch.zeros((len(grids), 10, 10)).to(device)

    for i, grid in enumerate(grids):
        sx_idx, sy_idx = max(0, min(round(grid['sx'] - xmin), width - 1)), max(0, min(height - 1 - round(
            grid['sy'] - ymin), height - 1))
        tx_idx, ty_idx = max(0, min(round(grid['tx'] - xmin), width - 1)), max(0, min(height - 1 - round(
            grid['ty'] - ymin), height - 1))
        sp[i, sy_idx, sx_idx] = 1.
        tp[i, ty_idx, tx_idx] = 1.

    return sp, tp

def eval_traj(logits, per_y, grids, sol_y):
    # logits: network prediction of obs+path (batch_size, 2, 10, 10, 2)
    # per_y: grid mat 
    # grids: batch['grids']
    # sol_y: traj mat
    """evaluate the model that directly outputs the trajectory matrix"""

    def gen_pred(logits, target, grids, gt_trajs, check_show=False,):
        """generate the smoothed predicted trajectory 01 matrix"""
        logit = logits[:, 0, :, :].squeeze(dim=1)  # shape = (batch_size, 10, 10, 2)
        pred = logit.argmax(dim=-1)  # shape = (batch_size, 10, 10)
        num_batch = target.shape[0]
        sample_ratio = 100
        smooth_method = ["interpolate", "Astar"][1]

        for i in range(num_batch):
            p, t, grid = pred[i], target[i], grids[i]  # p/t.shape = (10,10), grid is a dict
            if check_show:
                plt.imshow(p.cpu().numpy())
                plt.savefig(f"{i}_original.png")
                plt.close()
                plt.imshow(gt_trajs[i].cpu().numpy())
                plt.savefig(f"{i}_gt_traj.png")
                plt.close()
                plt.imshow(t.cpu().numpy())
                plt.savefig(f"{i}_gt_grid.png")
                plt.close()
            # to assign the start point and target point on the predicted trajectory map
            xmin, ymin, xmax, ymax = int(np.min(grid['bx'])), int(np.min(grid['by'])), \
                int(np.max(grid['bx'])), int(np.max(grid['by']))
            width, height = xmax - xmin, ymax - ymin
            sx_idx, sy_idx = max(0, min(round(grid['sx'] - xmin), width - 1)), max(0, min(height - 1 - round(
                grid['sy'] - ymin), height - 1))
            tx_idx, ty_idx = max(0, min(round(grid['tx'] - xmin), width - 1)), max(0, min(height - 1 - round(
                grid['ty'] - ymin), height - 1))
            p[sy_idx, sx_idx] = 1
            p[ty_idx, tx_idx] = 1
            if check_show:
                plt.imshow(p.cpu().numpy())
                plt.savefig(f"{i}_addst.png")
                plt.close()
            # to smooth the predicted trajectory
            if smooth_method == "interpolate":  # by interpolation
                w = p.nonzero().cpu().numpy()
                wx, wy = w[:, 1], w[:, 0]
                param = np.linspace(0, 1, wy.size)  # parameterized  input
                f = make_interp_spline(param, np.c_[wy, wx], k=2)  # function
                cxs, cys = f(np.linspace(0, 1, wy.size * sample_ratio)).T  # output
                cxs_idx, cys_idx = cxs.round().astype(np.int32), cys.round().astype(np.int32)
                for cx_idx, cy_idx in zip(cxs_idx, cys_idx):
                    p[cx_idx, cy_idx] = 1
            elif smooth_method == "Astar":  # by Astar
                find_consecutive_path(p, (sy_idx, sx_idx))
            if check_show:
                plt.imshow(p.cpu().numpy())
                plt.savefig(f"{i}_smoothed.png")
                plt.close()

        return pred

    def get_acc(pred, target):
        """get the precision and recall for the pred(generated by logits) with the target"""
        tp = torch.logical_and(pred == target, target > 0).sum().item()
        t, p = (target > 0).sum().item(), (pred > 0).sum().item()

        pre = tp / p if p > 0 else 0  # get precision
        rec = tp / t if t > 0 else 0  # get recall

        return tp, p, t

    def check_collide_length(pred, target):
        """check if the predicted trajectory(generated by logits) will collide on the grid with obstacles(presented by target)"""
        total_cnt = target.shape[0]
        collided_cnt = 0
        diff_len_cnt = 0

        for i in range(total_cnt):
            p, t = pred[i], target[i]  # p/t.shape = (10,10)
            # check if the smoothed predicted trajectory collided on the target grid
            collided = torch.any(p[t == 1] == 1)
            if collided:
                collided_cnt += 1
            else:
                diff_len_cnt += torch.abs(p.sum() - t.sum()).item()

        return collided_cnt, diff_len_cnt ,total_cnt

    def find_consecutive_path(matrix, start):
        import queue
        def is_valid(x, y, matrix):
            return 0 <= x < len(matrix) and 0 <= y < len(matrix[0])

        def bfs(matrix, ones, start):
            rows, cols = len(matrix), len(matrix[0])
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            visited[start[0]][start[1]] = True
            q = queue.Queue()
            q.put(start)
            parent = {start: (-1, -1)}

            while not q.empty():
                (x, y) = q.get()

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny, matrix) and not visited[nx][ny]:
                        if (nx, ny) in ones:
                            parent[(nx, ny)] = (x, y)
                            return (nx, ny), parent
                        if matrix[nx][ny] == 1:
                            continue
                        visited[nx][ny] = True
                        parent[(nx, ny)] = (x, y)
                        q.put((nx, ny))
            return -1, -1

        ones = set([(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == 1])
        while len(ones) != 0:
            ones.remove(start)
            end, parent = bfs(matrix, ones, start)
            if parent != -1:
                p = end
                while p != (-1, -1):
                    matrix[p[0]][p[1]] = 1
                    p = parent[p]
            start = end


    pred = gen_pred(logits, per_y, grids, sol_y)
    # get solving acc
    sol_tp, sol_p, sol_t = get_acc(pred, sol_y)
    # check if collided
    col_cnt, dl_cnt, tot_cnt = check_collide_length(pred, per_y)
    return sol_tp, sol_p, sol_t, col_cnt, dl_cnt, tot_cnt

def plot(grid, trajs, figsize=(6, 6), times=1, sample_ratio=100, save_path=None):
    """
    draw several trajs on the grid, but currently trajs only contain at most two trajs
    one is ground-truth traj, the other is predicted traj
    """

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)

    ## draw grid
    bx, by, oxs, oys, sx, sy, tx, ty = (
        np.array(grid['bx']) * times, np.array(grid['by']) * times,
        np.array(grid['ox']) * times, np.array(grid['oy']) * times,
        np.array(grid['sx']) * times, np.array(grid['sy']) * times,
        np.array(grid['tx']) * times, np.array(grid['ty']) * times,
    )

    # black boundary as cross
    ax1.scatter(bx, by, marker='+', s=20, c='k')
    # other colors as X
    for ox, oy in zip(oxs, oys):
        ax1.scatter(ox, oy, marker='x', s=20)
    # blue start point shaped as triangle
    ax1.scatter(sx, sy, c='slategray', marker='^', s=70)
    # red goal point shaped as star
    ax1.scatter(tx, ty, c='tomato', marker='*', s=70)

    ## draw trajactory
    labels = ['ground-truth', 'predicted']
    clist = ['darkorange', 'steelblue']  # now only support 2 trajs
    for i, traj in enumerate(trajs):
        px, py = np.array(traj['pathx']) * times, np.array(traj['pathy']) * times
        if len(px) == 1:  # no path
            continue
        while py[0] > ty:  # get rid of the waypoints that overhead the target point
            px, py = px[1:], py[1:]
        px, py = px[:-1], py[:-1]
        # create an interpolation function f
        wx, wy = np.append(np.append(px, sx)[::-1], tx), np.append(np.append(py, sy)[::-1], ty)  # waypoints
        param = np.linspace(0, 1, wy.size)  # parameterized  input
        f = make_interp_spline(param, np.c_[wy, wx], k=2)  # function
        cx, cy = f(np.linspace(0, 1, wy.size * sample_ratio)).T  # output

        # draw waypoints
        ax1.scatter(px, py, alpha=0.6, c=clist[i % len(clist)])
        # draw connecting plots
        ax1.plot(cy, cx, '-', alpha=0.7, c=clist[i % len(clist)], label=labels[i % len(labels)])

    # draw legend
    plt.legend(loc=(0.64, 0.83))
    if save_path:
        plt.savefig(save_path, dpi=100)


def show_grid(model, dataset="kitti", device="cuda:0"):
    pass


def show_traj(model, dataset="kitti", device="cuda:0"):
    """show the predicted trajectory and the ground-truth trajectory on the ground-truth grid
    for the model that directly outputs the trajectory matrix"""
    if dataset == "kitti":
        dataset = KittiDataset(cfg, split='test', is_train=False)
    elif dataset == "nuscenes":
        dataset = NuscenesDataset(cfg, split='test', is_train=False)

    idx = np.random.randint(len(dataset))
    data = dataset[idx]

    x, gt_grid, gt_traj = data[0].unsqueeze(0).to(device), data[3], data[4]
    preds, _, _ = model(x)  # shape=(1, 2, 10, 10, 2)
    pr_traj_mat = preds[0, 0, :, :, :].argmax(dim=-1)  # shape=(10,10)

    xmin, ymin, xmax, ymax = int(np.min(gt_grid['bx'])), int(np.min(gt_grid['by'])), \
        int(np.max(gt_grid['bx'])), int(np.max(gt_grid['by']))
    width, height = xmax - xmin, ymax - ymin
    pr_traj_path = pr_traj_mat.nonzero().cpu().numpy()
    pathx, pathy = list(map(lambda x: min(x + xmin, xmax - 1), pr_traj_path[:, 1])), \
        list(map(lambda y: min((height - 1 - y) + ymin, ymax - 1), pr_traj_path[:, 0]))
    pr_traj = {
        "pathx": pathx, "pathy": pathy,
        "mat": pr_traj_mat.cpu().numpy().tolist()
    }
    trajs = [gt_traj, pr_traj]
    plot(gt_grid, trajs, save_path="plot_{}.png".format(idx))


if __name__ == "__main__":
    ######      simple evaluation   #######
    cfg = setup_cfg()
    device = "cuda:0"
    dataset = ["kitti", "nuscenes"][0]
    model_flag = 0
    ckpt_paths = [
        "./logs/{}_rt_sup_model_st_e99.pth".format(dataset[:3]),
        "./logs/{}_rt_nesy_model_st_e99.pth".format(dataset[:3]),
        "./logs/{}_resnet10_model_st_e99.pth".format(dataset[:3])
    ]

    ## load the model
    if model_flag == 0:  # evaluate the model that directly outputs the trajectory matrix, which trained under a supervised mode
        model = build_model_rt(cfg, device)
        model.load_state_dict(torch.load(ckpt_paths[model_flag], map_location=device))
    elif model_flag == 1:  # evaluate the model that directly outputs the trajectory matrix, which trained under a neuro-symbolic mode
        model = build_model_rt(cfg, device)
        model.load_state_dict(torch.load(ckpt_paths[model_flag], map_location=device))
    elif model_flag == 2:  # evaluate the model that outputs the obstacle matrix, and gives the trajectory using A* planner
        model = build_model_resnet(cfg, device)
        model.load_state_dict(torch.load(ckpt_paths[model_flag], map_location=device))
    else:
        model = None

    ## evaluate
    if model_flag == 2:  # the model that outputs the obstacle matrix first
        eval_grid(model, dataset, device)
        show_grid(model, dataset, device)
    else:  # the model that directly outputs the trajectory matrix
        eval_traj(model, dataset, device)
        # show_traj(model, dataset, device)



def evaluate_batch_cnn(net, W, b, bmin, bmax, dataloader, threshold=0.0):

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    net.eval()
    symbol_acc = 0
    pred_acc = 0
    solving_acc = 0
    total = 0
    size = 10
    device = 'cuda:0'

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # forward the model and get output
            images, img_infos, gt_targets, gt_grids, gt_trajs = batch["images"].to(device), batch["img_infos"], \
                                                                batch["targets"], batch["grids"], batch["trajs"]

            output = net(images)
            output[output > 0.5] = 1.0
            output[output < 0.5] = 0.0

            trajs = batch["trajs"]
            y = torch.stack([torch.Tensor(traj["mat"]) for traj in trajs]).cuda()
            
            grids = batch["grids"]
            sp, tp = get_point_mat(grids)
            gt = torch.stack([torch.Tensor(grid["mat"]) for grid in grids]).cuda()

            num = output.size(0)
            def solve_(idx):
                ind1 = torch.where(sp[idx].reshape(-1) == 1)[0]
                ind2 = torch.where(tp[idx].reshape(-1) == 1)[0]
                _, res = maxsat_solver(W[:,100:200], bmin-W[:,0:100]@output[idx].reshape(1,-1).T, bmax-W[:,0:100]@output[idx].reshape(1,-1).T, ind1, ind2)
                return torch.Tensor(res).reshape(1,-1).cuda()

            res = Parallel(n_jobs=16)(delayed(solve_)(idx) for idx in range(num))
            paths = torch.cat(res, dim=0)
            output = output.reshape(num, -1)
            logits = torch.cat([paths, output], dim=-1).long()
            logits = F.one_hot(logits.reshape(num, 2, 10, 10), num_classes=2)

            print('pred obs and gt path')
            print(W.shape)
            print((W[:,0:100]@output[0].reshape(1,-1).T + W[:,100:200]@y[0].reshape(1,-1).T >= bmin).sum())
            print((W[:,0:100]@output[0].reshape(1,-1).T + W[:,100:200]@y[0].reshape(1,-1).T <= bmax).sum())
            
            print('pred obs and pred path')
            print((W[:,0:100]@output[0].reshape(1,-1).T + W[:,100:200]@res[0].reshape(1,-1).T >= bmin).sum())
            print((W[:,0:100]@output[0].reshape(1,-1).T + W[:,100:200]@res[0].reshape(1,-1).T <= bmax).sum())
            print(res[0].reshape(10,10), y[0].reshape(10,10))

            sol_tp, sol_p, sol_t, col_cnt, dl_cnt, tot_cnt = eval_traj(logits, gt, grids, y)
            print(sol_tp/sol_p, sol_tp/sol_t, col_cnt/tot_cnt, dl_cnt/(tot_cnt-col_cnt))
            # print(col_cnt)

def evaluate_batch_gpt(net, W, b, bmin, bmax, dataloader, threshold=0.0):

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    net.eval()
    total_sol_p = 0.0
    total_sol_t = 0.0
    total_sol_tp = 0.0
    total_col_cnt = 0.0
    total_dl_cnt = 0.0
    total = 0.0
    size = 10
    device = 'cuda:0'

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # forward the model and get output
            images, img_infos, gt_targets, gt_grids, gt_trajs = batch["images"].to(device), batch["img_infos"], \
                                                                batch["targets"], batch["grids"], batch["trajs"]


            trajs = batch["trajs"]
            y = torch.stack([torch.Tensor(traj["mat"]) for traj in trajs]).cuda()

            
            grids = batch["grids"]
            sp, tp = get_point_mat(grids)
            gt = torch.stack([torch.Tensor(grid["mat"]) for grid in grids]).cuda()

            sp, tp = get_point_mat(grids)
            logits, loss, _ = net(images, sp=sp, tp=tp)
            output = torch.sigmoid(logits)
            num = output.size(0)
            output[output > threshold] = 1.0
            output[output < threshold] = 0.0
            
            output = output.reshape(num, -1)
            tmp_y = y.reshape(num, -1)

            def solve_(idx):
                ind1 = torch.where(sp[idx].reshape(-1) == 1)[0]
                ind2 = torch.where(tp[idx].reshape(-1) == 1)[0]
                _, res = maxsat_solver_gpt(output[idx], W, bmin, bmax, ind1, ind2)
                return torch.Tensor(res).reshape(1,-1).cuda()

            res = Parallel(n_jobs=16)(delayed(solve_)(idx) for idx in range(num))
            paths = torch.cat(res, dim=0)
            # paths = output

            logits = torch.cat([paths, tmp_y], dim=-1).long()
            logits = F.one_hot(logits.reshape(num, 2, 10, 10), num_classes=2)

            for i in range(num):
                print(output[i].reshape(10,10), y[i].reshape(10,10))

            sol_tp, sol_p, sol_t, col_cnt, dl_cnt, tot_cnt = eval_traj(logits, gt, grids, y)
            print(sol_tp/sol_p, sol_tp/sol_t, col_cnt/tot_cnt, dl_cnt/(tot_cnt-col_cnt))

            total_sol_tp += sol_tp
            total_sol_p += sol_p
            total_sol_t += sol_t
            total_col_cnt += col_cnt
            total_dl_cnt += dl_cnt
            total += tot_cnt
    print('final result | precision: %.3f recall %.3f coll rate %.3f dist error %.3f' \
                           %(total_sol_tp/total_sol_p, total_sol_tp/total_sol_t, total_col_cnt/total, total_dl_cnt/(total-total_col_cnt)))


