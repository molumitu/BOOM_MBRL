# [NeurIPS2025] Bootstrap Off-policy with World Model (BOOM)

## Notes
Our paper reports the results after 2 million (2M) iterations. The default action repeat is 2, corresponding to 4 million (4M) environmental steps.  


## Quick Start Instructions

```bash
conda create -n boom python=3.10
conda activate boom
pip install -e .  # Ensure setup.py exists in the current directory
cd boom
python train.py # Default is dog-run, with an expected score of ~800.
# For Humanoid-bench, you can set task=humanoid_h1hand-run-v0 as an example.
```


## Bibtex citation
If you find our work useful, please cite our paper:
```text
@inproceedings{boom2025neurips,
  title={Bootstrap Off-policy with World Model},
  author={Zhan, Guojian and Wang, Likun and Zhang, Xiangteng and Gao, Jiaxin, Tomizuka, Masayoshi and Li, Shengbo Eben},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```