# object_LVDM
final project for genAI course in SNU (2023 Fall)

## Aim
object centric video generation via slot attention and consistency loss

## Consistency loss (slot loss in codebase)
1. High Temporal Similarity: We assume that frames close in time (e.g., frame $i$ and $j$) are highly similar. Therefore, the base loss (the -log(fraction) term, before applying the penalty) should be small for these pairs.
2. Penalty Mechanism: The penalty term's role is to enforce this temporal smoothness. The closer the frames $i$ and $j$ are (i.e., the smaller the temporal distance $|i-j|$), the smaller the penalty term should be. This weighting mechanism strongly incentivizes the model to make adjacent frames even more similar (thus achieving a minimal -log(fraction) value).

## Results
![OLVDM_example](https://github.com/Sangyoon-Bae/object_LVDM/assets/90450600/f7237c84-8212-4e5d-8da9-4df6d9d2db12)
![LVDM_example](https://github.com/Sangyoon-Bae/object_LVDM/assets/90450600/e30d106a-03b0-49af-bad2-bd1cfd5890db)
(Left : Object LVDM results, Right: LVDM results)


