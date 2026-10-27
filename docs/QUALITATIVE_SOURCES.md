# External qualitative image checks

These two photographs were chosen before the final checkpoint and deployment resolution were available. They are outside the project's Oxford-IIIT Pet splits and were used only to inspect new-image behavior with the frozen model. Neither has a segmentation annotation here. Two examples cannot establish general pet segmentation or pet-absence performance; their outputs were not used to tune the model.

The original PNG files are under `artifacts/qualitative/inputs/`; `artifacts/qualitative/sources.json` records download URLs, image dimensions, SHA-256 hashes, rights sources and visual observations. The files were downloaded unchanged from the scikit-image project's `v0.25.2` tag on October 24, 2026. Their decoded RGB hashes were checked against all project train, validation and test records; neither is an exact content match. This check does not rule out every near-duplicate.

| Input | Credit and rights | Intended check |
|---|---|---|
| `chelsea.png`, 451 × 300 | Photograph by Stefan van der Walt, dedicated under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), documented by [scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.chelsea). [Pinned source file](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/chelsea.png). | One close-up tabby cat, with whiskers and part of the head near the frame edges. Inspect pet coverage and visible background leakage. |
| `astronaut.png`, 512 × 512 | NASA photograph of Eileen Collins, distributed as public domain in [scikit-image's provenance](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut). The individual photographer is not named there. Educational reuse follows [NASA's image and media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/). [Pinned source file](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/astronaut.png). | A person, helmet, flag and shuttle model; no cat or dog is visible. Inspect any false pet foreground. |

The input images were visually inspected by the Codex agent. This does not claim that the candidate has personally reviewed them. Scikit-image identifies the astronaut's upstream source as [NASA's Flickr image](https://flic.kr/p/r9qvLn); that link was unavailable to the browsing tool, so the downloaded bytes came from the pinned scikit-image source instead.

Credit NASA for the astronaut input photograph. Masks, overlays and background composites are this project's predictions and are not NASA findings or endorsements. The photographs are used as factual educational examples, without implying the photographers' endorsement.

Input SHA-256 values:

- `chelsea.png`: `596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb`
- `astronaut.png`: `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`

The prediction CLI ran both inputs with the frozen experiment C checkpoint at the retained 256 input resolution. Each `outputs/<image>/` directory contains the original-size mask, overlay, composite and metadata. All output dimensions and the 0/255 mask encoding were checked. Metadata identifies checkpoint `9a4a8d23e09c29479a6969a9f4fba4533c7998da71b35b82576a07e86ed4ce16` and decision `db611c75e96f5dd20195aba19663e8345edfde7375e235839ae1dc4db8b7b07c`. `outputs/qualitative_examples.png` presents the two cases together.

The Codex agent visually inspected the actual masks, overlays and composites:

- **Chelsea:** the central face and most of the visible head remain in the mask. The composite replaces the surrounding background, but its smooth binary contour removes thin whiskers extending to the right. This is usable foreground separation for this example, with a visible fine-detail limitation. The mask contains 107,655 foreground pixels out of 135,300 (79.57%).
- **Astronaut:** small disconnected regions on the hair, suit collar, background shuttle area and lower edge are incorrectly retained as pet foreground. The no-pet input produces 1,051 foreground pixels out of 262,144 (0.401%). The composite retains those isolated fragments. A mostly empty mask in this single image does not establish reliable pet-absence rejection.

These pixel fractions describe only how much of each output mask is foreground. They are not IoU, accuracy or confidence scores: no reference segmentation masks were supplied. No model, threshold or deployment setting was changed in response to these observations. The candidate's personal review remains pending.
