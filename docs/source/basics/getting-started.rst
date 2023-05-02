Getting Started
===============

Installation
------------

(Optional) Python Environment
.............................

To get started, we suggest to try ``daxbench`` in a separate python environment.
You can do so using Python's built-in ``venv`` package:

.. code:: sh

    python -m venv ./daxbench_venv
    . ./daxbench_venv/bin/activate

Alternatively, use ``conda`` if you prefer so.

Python Dependencies
...................

There are a few python dependencies that are tricky to install. You would need
to install them manually. Install ``jax`` corresponding to your devices by
following the `official instruction <https://github.com/google/jax#installation>`_.

Install ``sdf``:

.. code:: sh

    pip install git+https://github.com/fogleman/sdf.git


Install Daxbench
................

Simply run:

.. code:: sh

    pip install .

Or install in development (editable) mode:

.. code:: sh

    pip install -e .

Simple Example
--------------

Let's verify the installation using a simple example. We will show how to create
multiple parallel ``daxbench`` environments, perform forward simulation, and
calculate gradients with respect to an objetive function.

We first create 3 parallel ``shape_rope`` environments, and three actions
corresponding to the three envrionments.

.. code:: python

    import jax
    import jax.numpy as jnp
    from daxbench.core.envs import ShapeRopeEnv

    # Crreate the environments
    env = ShapeRopeEnv(batch_size=3, seed=1)
    obs, state = env.reset(env.simulator.key)

    # Actions to be simulated in each environment
    actions = jnp.array(
        [
            [0.4, 0, 0.4, 0.6, 0, 0.6],
            [0.6, 0, 0.6, 0.4, 0, 0.4],
            [0.4, 0, 0.6, 0.6, 0, 0.4],
        ]
    )

Then we apply the actions to the environments. We use the method
``step_with_render`` to visualize the effect of the first action in the first
environment; note that this method is not accelerated by ``Jax``.

.. code:: python

    obs, reward, done, info = env.step_with_render(actions, state)
    next_state = info["state"]

To take advantage of Jax, we suggest to separate rendering from the forward
simulation. ``step_diff`` method is accelerated by Jax's just-in-time (jit)
compilation.

.. code:: python

    obs, reward, done, info = env.step_diff(actions, state)
    next_state = info["state"]
    image = env.render(next_state, visualize=True)

To compute the gradient of the actions to maximize the reward, we use `jax.grad
<https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html>`_ as a
decorator. Instead of returning the objective value, the decorated fuction
returns the gradient of the objective with respect to the specified (by default
the first one) arguments.

.. code:: python

    @jax.jit
    @jax.grad
    def compute_grad(actions, state):
        obs, reward, done, info = env.step_diff(actions, state)
        objective_to_be_minimized =  - reward.sum()
        return objective_to_be_minimized

    print("action gradients:", compute_grad(actions, state))

DaxBench Environments
---------------------

``DaxBench`` implements the following environments, you may also implement your own customized environment using them as examples,

- ``daxbench.core.envs.PourWaterEnv``: Pour a bowl of water into the target
  bowl.
- ``daxbench.core.envs.PourSoupEnv``: Pour a bowl of soup with various solid
  ingredients into the target bowl.
- ``daxbench.core.envs.ShapeRopeEnv``: Push the rope to the pre-specified
  configuration.
- ``daxbench.core.envs.ShapeRopeHardEnv``: Push the rope to the pre-specified
  configuration. The initial configuration is more complicated.
- ``daxbench.core.envs.WhipRopeEnv``: Whip the rope into a target configuration.
- ``daxbench.core.envs.FoldCloth1Env``: Fold a piece of flattened cloth and move
  it to a target location. The target location requires 1 fold.
- ``daxbench.core.envs.FoldCloth3Env``: Fold a piece of flattened cloth and move
  it to a target location. The target location requires 3 fold.
- ``daxbench.core.envs.FoldTshirtEnv``: Fold a T-shirt to a target location.
- ``daxbench.core.envs.UnfoldCloth1Env``: Flatten a piece of folded cloth to a
  target location. The cloth is initial folded once.
- ``daxbench.core.envs.UnfoldCloth3Env``: Flatten a piece of folded cloth to a
  target location. The cloth is initial folded for 3 times.

In addition, the dictionary ``daxbench.core.envs.registration`` maps
strings to the environment classes

.. code:: python

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


Interactive Scripts
...................

We implemented interactive scripts in the source code of each environment.

.. code:: sh

    python daxbench/core/envs/shape_rope_env.py

Here are the interfaces for each environment:

- ``daxbench/core/envs/shape_rope_env.py``: A image would pop up. Click on the
  image to specify the start and end of a push.
- ``daxbench/core/envs/shape_rope_hard_env.py``: Same as ``shape_rope_env.py``.
- ``daxbench/core/envs/fold_cloth1_env.py``: A image would pop up. Click on the
  image to specify the pick and place locations.
- ``daxbench/core/envs/fold_cloth3_env.py``: Same as ``fold_cloth1_env.py``.
- ``daxbench/core/envs/fold_cloth_tshirt_env.py``: Same as ``fold_cloth1_env.py``.
- ``daxbench/core/envs/unfold_cloth1_env.py``: Same as ``fold_cloth1_env.py``.
- ``daxbench/core/envs/unfold_cloth3_env.py``: Same as ``fold_cloth1_env.py``.
- ``daxbench/core/envs/pour_water_env.py``: Focus on the black opencv image
  window with name ``control pad`` and use the keyboard to control the bowl.
    + ``w``: Forward.
    + ``s``: Backward.
    + ``a``: Left.
    + ``d``: Right.
    + ``shift``: Tile towards the right.
    + ``tab``: Tile towards the left.
    + ``enter``: Quit.
- ``daxbench/core/envs/pour_soup_env.py``: Same as ``pour_water_env.py``.
- ``daxbench/core/envs/whip_rope_env.py``: Focus on the black opencv image
  window with name ``control pad`` and use the keyboard to control the rope.
    + ``w``: Forward.
    + ``s``: Backward.
    + ``a``: Left.
    + ``d``: Right.
    + ``shift``: Downward.
    + ``tab``: Upward.


Environment Configuration
.........................

Each environment comes with a configuration `dataclass
<https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass>`_. We
follow the naming convention of ``{TaskName}Env`` and ``{TaskName}Config``. In
general, you don't need to change the configuration unless something went wrong.

The following configuration attributes are important:

- ``E``: Rigidity of the object, in the range of :math:`[0, 1e6]`. The object
  behaves like liquid when :math:`E \in [0, 1]`.
- ``nu``: Poisson ratio. It determines how the object deforms under stresses.
  Its in the range of :math:`[-1, 0.5)`.
- ``ngrid``: Number of grids. Smaller grids result in higher simulation fidelity
  and longer computation time.
- ``res``: ratio of the region for Lazy Dynamic Update.
- ``dt``: The simulation time step length of the physics. It does not correspond
  to how long a ``step_diff`` or ``step_with_render`` call takes. The more rigid
  the object is, the smaller the ``dt`` needs to be.
- ``primitive_action_duration``: Duration of the macro actions. Longer durations
  require more computations per ``step_diff`` (resp. ``step_with_render``) call.

The environment can be configured using the keyword argument ``conf``

.. code:: python

    from daxbench.core.envs import ShapeRopeEnv, ShapeRopeConfig
    conf = ShapeRopeConfig(dt=0.4e-4)
    conf.E = 101
    env = ShapeRopeEnv(batch_size=2, seed=1, conf=conf)
    obs, state = env.reset(env.simulator.key)

Train a Policy with DaxBench
----------------------------

When training a policy with DaxBench, consider these essential configuration attributes:

- ``env``: Environment for training and evaluation.
- ``ep_len``: Length of each episode.
- ``num_envs``: Number of environments used in training.
- ``lr``: Learning rate.
- ``gpus``: Number of GPUs for training and evaluation.
- ``seed``: Random seed for initializing training and evaluation environments, as well as policy parameters.
- ``eval_freq``: Policy evaluation frequency.
- ``max_grad_norm``: The maximum gradient to perform gradient clip. 

For example, you may use the following script to train a policy using Analytical Policy Gradient (APG) on the ``fold_cloth3`` environment with 4 environments. The policy will be trained for 2000 iterations at a learning rate of 0.0001.

.. code:: sh 

   python -m daxbench.algorithms.apg.apg \
          --env fold_cloth3 \
          --ep_len 3 \
          --num_envs 4 \
          --lr 1e-4 \
          --gpus 1 \
          --max_grad_norm 0.3 \
          --seed 0 \
          --eval_freq 20 

To see all available options and their default values, run

.. code:: sh

   python -m daxbench.algorithms.apg.apg --help
