This repo contains the codebase of the paper _**Tuning vision-language models with candidate labels by prompt alignment**_, which is accepted in DASFAA 2025. Below are instructions to reproduce our experiments.

# Environment installation
Please refer to [Dassl.](https://github.com/KaiyangZhou/Dassl.pytorch)

# implementation
You can run the following command to train models with our algorithm.

```
bash scripts/coop_pll.sh [DATASET] [CONFIG] [CTX_LEN] [PARTIAL_RATE] [TASK_NAME] [PLL_ALG] [GPU_ID]

bash scripts/coop_pll.sh imagenet rn50 16 0.1 test_PiCO PiCO 0
```
We support these partial label learning training objectives: CC, RC, LWS, CAVL, PlCR, PiCO, NAIVE

# Citation
If you find our work useful, please cite it using the following BibTeX entry:
```
@article{zhang2025tuning,
    title={Tuning Vision-Language Models with Candidate Labels by Prompt Alignment},
    author={Zhifang Zhang and Yuwei Niu and Xin Liu and Beibei Li},
    journal={DASFAA},
    year={2025}
}
```