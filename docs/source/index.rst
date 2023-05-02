:github_url: https://github.com/AdaCompNUS/DaXBench/tree/HEAD/docs

DaxBench
--------

Deformable object manipulation (DOM) is a long-standing challenge in robotics
and has attracted significant interest recently. This work presents
**DaXBench**, a differentiable simulation framework for DOM. While existing
work often focuses on a specific type of deformable objects, DaXBench supports
fluid, rope, cloth . . . ; it provides a general-purpose benchmark to evaluate
widely different DOM methods, including planning, imitation learning, and
reinforcement learning. DaXBench combines recent advances in deformable object
simulation with JAX, a high-performance computational framework. All DOM tasks
in DaXBench are wrapped with the OpenAI Gym API for easy integration with DOM
algorithms. We hope that DaXBench provides to the research community a
comprehensive, standardized benchmark and a valuable tool to support the
development and evaluation of new DOM methods.

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    basics/getting-started.rst


Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/metaopt/daxbench/issues>`_.

License
-------

DaxBench is licensed under the Apache 2.0 License.

Citing
------

If you find DaxBench useful, please cite it in your publications.

.. code-block:: bibtex

    @inproceedings{chen2023daxbench,
        title={DaXBench: Benchmarking Deformable Object Manipulation with Differentiable Physics},
        author={Siwei Chen* and Yiqing Xu* and Cunjun Yu* and Linfeng Li and Xiao Ma and Zhongwen Xu and David Hsu},
        year={2023},
        booktitle={ICLR}
    }
