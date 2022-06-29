import numpy as np


def calc_IOU(curr_pc, goal_pc_path, n_grid=64):
    goal_pc = np.load(goal_pc_path)
    goal_pc = np.round(goal_pc * n_grid).astype(np.int32)
    goal_voxels = np.zeros((n_grid, n_grid, n_grid), dtype=np.int32)
    goal_voxels[goal_pc[:, 0], 0, goal_pc[:, 2]] = 1

    curr_pc = np.round(curr_pc * n_grid).astype(np.int32)
    curr_voxels = np.zeros((n_grid, n_grid, n_grid), dtype=np.int32)
    curr_voxels[curr_pc[:, 0], 0, curr_pc[:, 2]] = 1

    # goal_pcd_o3d = o3d.geometry.PointCloud()
    # goal_pcd_o3d.points = o3d.utility.Vector3dVector(goal_pc)
    # goal_pcd_o3d.paint_uniform_color(np.array([0, 1, 0]))
    # curr_pcd_o3d = o3d.geometry.PointCloud()
    # curr_pcd_o3d.points = o3d.utility.Vector3dVector(curr_pc)
    # curr_pcd_o3d.paint_uniform_color(np.array([1, 0, 0]))
    # o3d.visualization.draw_geometries([goal_pcd_o3d, curr_pcd_o3d])

    merged_voxels = goal_voxels + curr_voxels
    intersection = (merged_voxels == 2).astype(np.int32).sum()
    union = (merged_voxels > 0).astype(np.int32).sum()
    iou = 1.0 * intersection / union

    return iou
