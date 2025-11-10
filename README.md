# object_LVDM
This is final project for genAI course in SNU (2023 Fall)

## Aim
Object-centric video generation via slot attention and consistency loss

## Consistency loss (slot loss in codebase)
1. High Temporal Similarity: We assume that frames close in time (e.g., frame $i$ and $j$) are highly similar. Therefore, the base loss (the -log(fraction) term, before applying the penalty) should be small for these pairs.
2. Penalty Mechanism: The penalty term's role is to enforce this temporal smoothness. The closer the frames $i$ and $j$ are (i.e., the smaller the temporal distance $|i-j|$), the smaller the penalty term should be. This weighting mechanism strongly incentivizes the model to make adjacent frames even more similar (thus achieving a minimal -log(fraction) value).

## Results
![OLVDM_example](https://github.com/Sangyoon-Bae/object_LVDM/assets/90450600/f7237c84-8212-4e5d-8da9-4df6d9d2db12)
![LVDM_example](https://github.com/Sangyoon-Bae/object_LVDM/assets/90450600/e30d106a-03b0-49af-bad2-bd1cfd5890db)

(Left : Object LVDM results, Right: LVDM results)

The GIF above directly compares the output of OLVDM (ours) with the LVDM (baseline). It's clear that the baseline model (LVDM) produces significant flickering and blurring artifacts. In contrast, our OLVDM generates a much more temporally consistent video, where the difference in the object's appearance between adjacent frames is far smoother and more natural.

<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/75228a32-39a8-42fc-9419-b152bccf13d9" />
The figure below shows a frame-by-frame (t=0 to 14) comparison of video generation results from our OLVDM (Ours) and the LVDM (Baseline) model.

As shown in the qualitative results:
OLVDM (Top Rows): Our model successfully generates objects (the aurora in the first row, clouds in the second) that are clearly visible and consistent throughout the video. The highlighted objects change and move smoothly and naturally between frames.
LVDM (Bottom Rows): The baseline model struggles with object consistency. The generated frames are often darker , and the objects are difficult to differentiate. These frames exhibit significant blurring and flickering artifacts , which are common obstacles in generating temporally consistent video.

This comparison highlights how our object-centric approach, using Slot Attention and the proposed consistency loss , effectively reduces these artifacts to produce higher-quality, more natural-looking videos.

