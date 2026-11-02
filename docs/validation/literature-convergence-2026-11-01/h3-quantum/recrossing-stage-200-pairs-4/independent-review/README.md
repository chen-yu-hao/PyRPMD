# Independent replay of the complete four-pair P=128 cohort

The review independently reconstructed all **800 raw child checkpoints**, the eight exact parent checkpoint envelopes, keyed channels and Maxwell momenta, signed flux arrays, both timestep records, the complete 200-point transmission curves and the eight-chain statistics. No producer or production analyzer function was called; no PES or dynamics was evaluated.

The full rate is **1.967856365280176e-16 cm³ molecule⁻¹ s⁻¹**, with nominal eight-chain 95% interval **[1.932938606756291e-16, 2.003404899068246e-16]**. These values reproduce the completed analysis. All eleven registered fixed-P primary criteria reproduce as passed. The transmission-coefficient relative 95% halfwidth is **2.317249%**, below the registered 3% target.

The independent uncertainty unit remains one complete equilibrium chain, including its correlated BAR surface partition and children. The four pair numerator/denominator sums are combined within a chain before taking its transmission ratio. The paired timestep calculations share exact momenta and channels and are not extra independent chains. The same-chain log-partition/log-transmission covariance is retained in the rate interval.

Two earlier or supplementary failures remain explicit. The original two-pair stage has a **3.272200%** transmission halfwidth and fails its precision gate. The complete four-pair stage's supplementary first/second-parent-half difference has interval **[-0.02310073441, 0.03625536217]**, which is not wholly inside ±0.03. Extra momenta do not provide additional parent equilibration. This latter diagnostic was supplementary and was not silently promoted or discarded as a stopping rule.

The nominal t intervals do not independently correct finite-chain ratio/geometric-mean bias or conditional-extension coverage. Minimum pair distances were recorded at 1-fs boundaries, not every physical step. This review checks the stored estimator and data identities; it does not re-integrate forces, rerun BAR, establish dynamic bead convergence, or validate the production MD umbrella sampler. The separate P=256 control remains necessary for the dynamic bead question.

`replay_complete.py` is the reproducible independent implementation. `original/` holds the full numerical replay, all input hashes, per-chunk identities and parent-file bindings. `original-execution.json` and `original-replay.log` preserve the actual command and outcome. The report in `independent-review.json` also binds and verifies all 63 files in root's separate frozen integration publication inventory; no prior inventory was edited.

The incremental raw archive and its fresh extraction replay are deliberately separate from this review's inventory. They must preserve the immutable first-stage archive and the exact parent archive, and reproduce these same results from their union.
