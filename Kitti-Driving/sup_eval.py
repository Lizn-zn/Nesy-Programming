from dataset import build_dataset_kitti, default_arg_parse, setup_cfg
from model import build_model_resnet
from nn_utils import *

test_num_batch = 100
grid_resolution = 1.0
robot_radius = 2.0
threshold = 0.35
device = 'cuda:0'
ckpt = './checkpoint/sup_cv_best_0.9_2.0.pth_0.t7'

# init model
opt = default_arg_parse()
cfg = setup_cfg(opt)
net = build_model_resnet(cfg, device)

# init chektpoint
if ckpt != None and ckpt != "":
    static_dict = torch.load(ckpt)
    net.load_state_dict(static_dict['net'])

# init dataloader
test_dataloader = build_dataset_kitti(cfg, is_train=False)

net.eval()
prec, recall, f1_score, collision_rate, avg_delta_dist, test_cnt, no_solve_cnt = 0., 0., 0., 0., 0., 0.0, 0.0

################################ start testing ################################

for batch_idx, batch in enumerate(test_dataloader):  # default batch_size=1
    # forward the model and get output
    images, img_infos, gt_targets, gt_grids, gt_trajs = batch["images"].to(device), batch["img_infos"], \
                                                                batch["targets"], batch["grids"], batch["trajs"]
    output = net(images)
    target = torch.Tensor(gt_grids[0]["mat"]).float().to(device)

    # compute the keypoints RMSE loss

    with torch.no_grad():
        output[output >= threshold] = 1.
        output[output < threshold] = 0.
        tp = torch.sum(output * target).item()
        p, t = torch.sum(output).item(), torch.sum(target).item()
        if p != 0:
            t1 = tp / p
            prec += t1
        if t != 0.:
            t2 = tp / t
            recall += t2
        if t1 + t2 != 0.:
            f1_score += 2 * t1 * t2 / (t1+t2)


        # generate the predicted grid
        pr_grid = gen_predicted_grid(output, gt_grids[0])

        # generate the trajectory on the predicted grid
        pr_traj = gen_traj(pr_grid, grid_resolution, robot_radius)

        # compute the delta_traj_dist = | |gt_traj| - |pr_traj| |
        delta_traj_dist, flag = compute_delta_traj_dist(gt_trajs[0], pr_traj)
        if flag == 1:
            print("We ignore the situation when the goal is not reachable for the ground-truth for now")
        elif flag == 2:
            no_solve_cnt += 1
        else:  # flag == 0
            avg_delta_dist += delta_traj_dist

        # check whether the trajectory on the predicted grid will collide on the ground-truth grid
        collided = check_collision(gt_grids[0], pr_traj, robot_radius, device)
        if collided:
            collision_rate += 1
        test_cnt += 1

        if test_num_batch:
            if test_cnt > test_num_batch:
                break


        # plot for detection result
        # plot for the predicted trajectory on the ground-truth grid
        # if plot_show:
        #     idx = img_infos[0]["idx"]
        #     idx = ("000000" + str(idx))[-6:]

        #     if not self.args.save_plot:
        #         self.__plot_grids_and_trajs(gt_grids[0], gt_trajs[0], pr_grid, pr_traj)
        #     else:
        #         planning_save_path = os.path.join(self.args.save_root, "planning", idx + ".png")
        #         self.__plot_grids_and_trajs(gt_grids[0], gt_trajs[0], pr_grid, pr_traj,
        #                                         save=True, save_path=planning_save_path)


# for classification
prec = prec / test_cnt
recall = recall / test_cnt
f1_score = f1_score / test_cnt

# for planning
collision_rate = collision_rate / test_cnt
avg_delta_dist = avg_delta_dist / (test_cnt - no_solve_cnt) 

# for classification
print("The average precision, recall, f1-score over {} samples for binary classification is {:.2f}% | {:.2f}% | {:.2f}%"
        .format(test_cnt, prec * 100, recall * 100, f1_score * 100))

# for planning
print("The collision rate over {} samples for trajectory planning is {:.2f} %".format(test_cnt, collision_rate * 100))
print("The average distance between the ground-truth trajectories and the predicted ones " +
        "over {} solved samples is {:.4f}".format(test_cnt - no_solve_cnt, avg_delta_dist))
print("The no-solution rate over {} samples with the detection threshold {} is {:.2f}%"
        .format(test_cnt, threshold, no_solve_cnt * 100 / test_cnt))

