# Learned centroid CVs with mlcolvar

PyRPMD can evaluate an exported Deep-LDA or Deep-TICA model through
`pyrpmd.advanced_cv.TorchScriptCV`. Training happens in mlcolvar; production
sampling uses a fixed TorchScript graph and its Cartesian derivatives. The
same CV can be supplied to umbrella sampling, metadynamics or the new centroid
enhanced-sampling APIs, subject to their individual CV restrictions.

The runnable route is `examples/mlcolvar_cv.py`. The example targets
`mlcolvar>=1.2,<2`; its constructors and data keys were checked against the
1.3.1 release. It uses the public `DeepLDA`, `DeepTICA`, `DictDataset` and
`DictModule` APIs documented in the official
[Deep-LDA tutorial](https://mlcolvar.readthedocs.io/en/stable/notebooks/tutorials/cvs_DeepLDA.html)
and [Deep-TICA tutorial](https://mlcolvar.readthedocs.io/en/stable/notebooks/tutorials/cvs_DeepTICA.html).

## Train and export

Install the optional training dependencies in a separate environment:

```bash
python -m pip install -e '.[test]'
python -m pip install 'mlcolvar>=1.2,<2'
python examples/mlcolvar_cv.py --method deep-lda --output deep-lda.pt
python examples/mlcolvar_cv.py --method deep-tica --output deep-tica.pt
```

Both commands explicitly train and export on the CPU in float64. They do not
select an accelerator automatically. The example restores the calling
process's default torch dtype, RNG state and deterministic-algorithm setting.
For molecular GPU training, choose the training device and precision explicitly
and validate the final float64 deployment graph on the intended sampling device.

Deep-LDA uses labelled samples from two synthetic basins; every training batch
contains both classes. Deep-TICA uses time-adjacent samples from two stationary
Ornstein–Uhlenbeck processes with different relaxation times. It provides
`data`, `data_lag`, `weights` and `weights_lag`; both weights are one because
these toy trajectories are unbiased and uniformly sampled. These small datasets
exercise genuine training and deployment. They do not establish the quality of
a reaction coordinate for a molecule.

The inference graph contains descriptor extraction, fitted input normalization,
the trained network and the final LDA/TICA projection. After the last optimizer
step the projection is refitted on the training partition, so validation does
not choose the deployed projection. Two independent probe geometries verify
that serialization preserves both values and nonzero coordinate derivatives.

## Match the coordinate contract

mlcolvar models normally receive a batch of descriptors. A centroid PyRPMD CV
receives one `[N,3]` float64 array of positions in **bohr** and must return one
`[D]` tensor. The example wraps the model to extract atom zero's x and y
coordinates and add/remove the batch dimension. For a real molecule replace
that descriptor function with differentiable distances, angles or another
symmetry-respecting representation. Keep the feature order, reference geometry,
units, normalization and atom order in the exported graph or its explicit
immutable configuration.

```python
import numpy as np
from pyrpmd.advanced_cv import TorchScriptCV
from pyrpmd.bias import UmbrellaBias

cv = TorchScriptCV(
    "deep-tica.pt", dimension=1, units=("dimensionless",),
    periods=(None,), estimator="centroid", device="cpu",
)
q = np.array([[[0.2, -0.1, 0.0]], [[0.4, 0.1, 0.0]]], dtype=np.float64)
result = cv.value_and_jacobian(q)
assert result.value.shape == (1,)
assert result.jacobian.shape == (1, 2, 1, 3)

# Harmonic bias energy is physical; UmbrellaBias applies its own P scaling.
bias = UmbrellaBias(cv, center=[0.0], stiffness=0.1)
ring_bias_energy, ring_bias_force = bias.energy_force(q)
```

Autograd differentiates through the centroid, so each bead derivative contains
the required factor `1/P`. Do not divide the model output or bias force by P
again. The serialized artifact's SHA-256 participates in existing TorchScript
CV identities. Keep the deployed model immutable during a run; retraining
creates a different scientific input.

## Training and validation for molecular data

For Deep-LDA, labels identify sampled metastable basins. Separation of those
labels does not certify a transition-state coordinate. For Deep-TICA, choose a
physical lag time, preserve time pairs within each trajectory, and keep training
and validation trajectories or long blocks separate. The random split in this
small deployment example is not a molecular kinetic-validation protocol.

Biased trajectories need the appropriate reweighting and time treatment; their
unit weights cannot simply be copied from the unbiased toy example. The
[official biased Deep-TICA example](https://mlcolvar.readthedocs.io/en/stable/notebooks/examples/ex_DeepTICA.html)
shows the corresponding mlcolvar route. Validate slow-mode stability across lag
times, feature choices and independent trajectories. Check held-out derivatives
and bias forces, including molecular symmetries and any periodic representation.
Avoid detached NumPy preprocessing inside the deployed model, which removes
the gradients needed for biased sampling.

Run the optional end-to-end checks with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_mlcolvar_cv.py
```

Each test trains the selected real mlcolvar model, exports it, loads it with
`TorchScriptCV`, compares the predicted CV, checks the complete ring Jacobian
against finite differences and verifies a biased Cartesian force. Missing
PyTorch, Lightning or mlcolvar dependencies are guarded with `pytest.importorskip`.
Passing these deployment checks does not imply physical convergence.

## References

Catalog entry: [samp-mlcolvar](manual/references.md#samp-mlcolvar).

- Bonati, Trizio, Rizzi and Parrinello, *A unified framework for machine learning
  collective variables for enhanced sampling simulations: mlcolvar*, J. Chem.
  Phys. **159**, 014801 (2023), [doi:10.1063/5.0156343](https://doi.org/10.1063/5.0156343).
- Bonati, Rizzi and Parrinello, *Data-driven collective variables for enhanced
  sampling*, J. Phys. Chem. Lett. **11**, 2998–3004 (2020),
  [doi:10.1021/acs.jpclett.0c00535](https://doi.org/10.1021/acs.jpclett.0c00535).
- Bonati, Piccini and Parrinello, *Deep learning the slow modes for rare events
  sampling*, Proc. Natl. Acad. Sci. USA **118**, e2113533118 (2021),
  [doi:10.1073/pnas.2113533118](https://doi.org/10.1073/pnas.2113533118).
