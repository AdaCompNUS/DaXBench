from daxbench.core.envs.fold_cloth1_env import FoldCloth1Env
from daxbench.core.envs.fold_cloth3_env import FoldCloth3Env
from daxbench.core.envs.fold_cloth_tshirt_env import FoldTshirtEnv
from daxbench.core.envs.pour_soup_env import PourSoupEnv
from daxbench.core.envs.pour_water_env import PourWaterEnv
from daxbench.core.envs.shape_rope_env import ShapeRopeEnv
from daxbench.core.envs.shape_rope_hard_env import ShapeRopeHardEnv
from daxbench.core.envs.unfold_cloth1_env import UnfoldCloth1Env
from daxbench.core.envs.unfold_cloth3_env import UnfoldCloth3Env
from daxbench.core.envs.whip_rope_env import WhipRopeEnv

env_functions = {
    "fold_cloth1": FoldCloth1Env,
    "fold_cloth3": FoldCloth3Env,
    "fold_tshirt": FoldTshirtEnv,
    "shape_rope": ShapeRopeEnv,
    "push_rope": ShapeRopeEnv,
    "shape_rope_hard": ShapeRopeHardEnv,
    "push_rope_hard": ShapeRopeHardEnv,
    "unfold_cloth1": UnfoldCloth1Env,
    "unfold_cloth3": UnfoldCloth3Env,
    "pour_water": PourWaterEnv,
    "pour_soup": PourSoupEnv,
    "whip_rope": WhipRopeEnv,
}
