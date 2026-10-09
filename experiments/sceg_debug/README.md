# Four-image overfit debug

Four real CUB images from four classes, deterministically resized/cropped. Those same images are reused as support and query: four unique originals, eight tensor rows. This deliberately tests memorization; it is not a held-out episode or paper result.

200 steps; augmentation, dropout, weight decay off. All four official loss terms used. Four-class global head is a documented debug override. Strict inactive-parameter gradient failures remain visible.

Pass criterion: final train-mode loss <=0.05; episodic AND global accuracy 100%. Official loss cannot reach exactly zero: four positive terms alone have floor `4*softplus(-38*(1-0.85)) ≈0.01336`. Eval-mode accuracy is additionally reported.
