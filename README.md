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
python train.py # Default is dog-run, with an expected score of ~800.
# For Humanoid-bench, you can set task=humanoid_h1-run as an example.
```


## Bibtex citation
If you find our work useful, please cite our paper:
```text
@inproceedings{boom2025neurips,
  title={Bootstrap Off-policy with World Model (BOOM)},
  author={Guojian Zhan, Likun Wang, Xiangteng Zhang, Jiaxin Gao, Masayoshi Tomizuka, Shengbo Eben Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```