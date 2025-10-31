# [NeurIPS2025] Bootstrap Off-policy with World Model (BOOM)

## Notes
Our paper reports the results after 2 million (2M) iterations.  
- For DMC: Action repeat is 2, corresponding to 4 million (4M) environmental steps.  
- For Humanoid-Bench: Action repeat is 1, corresponding to 2 million (2M) environmental steps.

## Quick Start Instructions

```bash
conda create -n boom python=3.9
conda activate boom
pip install -r requirements.txt
pip install -e .  # Ensure setup.py exists in the current directory
cd boom
python train.py
```


## Citation
If you find our work useful, please cite our paper:
```text
@inproceedings{boom2025neurips,
  title={Bootstrap Off-policy with World Model (BOOM)},
  author={Guojian Zhan, Likun Wang, Xiangteng Zhang, Jiaxin Gao, Masayoushi Tomizuka, Shengbo Eben Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```