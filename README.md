> Eurographics 2026
<p align="center">
<h1 align="center"><strong>Physics-Based Motion Tracking of Contact-Rich Interacting Characters</strong></h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=MHQRNggAAAAJ" target="_blank">Xiaotang Zhang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=gHhQNlYAAAAJ&hl" target="_blank">Ziyi Chang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=t1hraiAAAAAJ&hl" target="_blank">Qianhui Men</a><sup>2</sup>,
    <a href="http://hubertshum.com/" target="_blank">Hubert Shum</a><sup>1&dagger;</sup>
    <br>
      <sup>1</sup>Durham University  
      <sup>2</sup>University of Bristol
    <br>
      &dagger; Corresponding Author
  </p>
</p>

<div id="top" align="center">
  
[[Paper]](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70336) [[Video]](https://youtu.be/37nH8QE2ycE) [[arXiv]](https://arxiv.org/abs/2604.07984)

</div>

![Teaser](/materials/teaser.png)
### Abstract
Motion tracking has been an important technique for imitating human-like movement from large-scale datasets in physics-based motion synthesis. However, existing approaches focus on tracking either single character or a particular type of interaction, limiting their ability to handle contact-rich interactions. Extending single-character tracking approaches suffers from the instability due to the challenge of forces transferred through contacts. Contact-rich interactions requires levels of control, which places much greater demands on model capacity. To this end, we propose a robust tracking method based on progressive neural network (PNN) where multiple experts are specialized in learning skills of various difficulties. Our method learns to assign training samples to experts automatically without requiring manually scheduling. Both qualitative and quantitative results show that our method delivers more stable motion tracking in densely interactive movements while enabling more efficient model training.

### Usage
- This project is tested on `IsaacSim 4.5`, `IsaacLab v2.0.X` and `Ubuntu 22.04`. 
- Please kindly install the IsaacLab framework from: https://github.com/isaac-sim/IsaacLab.
- Clone this repository to `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct`.
- Create virtural environment and install packages from `requirements.txt`.
- Run the project with `bash run.bash`

### BibTex
```
@inproceedings{zhang2026physics,
  title={Physics-Based Motion Tracking of Contact-Rich Interacting Characters},
  author={Zhang, Xiaotang and Chang, Ziyi and Men, Qianhui and Shum, Hubert PH},
  booktitle={Computer Graphics Forum},
  pages={e70336},
  year={2026},
  organization={Wiley Online Library}
}
```
