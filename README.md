# MLP Can Be A Good Transformer Learner (CVPR2024)

## Overview
- The implementation of Base architecture (**deit_base_patch16_224_attn**) can be found in [models_v4.py](models_v4.py). For Small (**deit_small_patch16_224_copy_lambda**) and Tiny (**deit_tiny_patch16_224_copy_lambda**) architectures, they are in [models_small_timm3_lambda.py](models_small_timm3_lambda.py).
- We copy the code from timm 0.3.2 to implement the baseline for small and tiny archs in [models_small_timm3.py](models_small_timm3.py), which further results in our model for small and tiny archs in [models_small_timm3_lambda.py](models_small_timm3_lambda.py). However, the model *deit_base_patch16_224_copy_lambda* has not been tested.

## Note
- There is a ***BUG*** when saving the best models in [main_lambda.py](main_lambda.py) because it will save the model with lambda>0. It's easy to fix this bug if you want to save the best checkpoint with lambda=0.
- Additionally, we save the checkpoints at 280, 290 and 295 epochs (lambda>0). If you don't need them, just comment the code snippets.
- Hence, we use the final checkpoint for evaluation. Again, if you want to use the best checkpoint with lambda=0, it's easy to fix the bug on your own.

## Requirements
- timm==0.4.12
- einops
- torchprofile
- fvcore

Note that we use torch==1.7.1 for training. To incorparate with [ToMe](https://github.com/facebookresearch/ToMe), we use torch==1.12.1.

## Checkpoints
We provide some checkpoints for reference. Here the prefix indicates the architectures while the suffix indicates which attention layers are removed.
- [base_01346.pth](https://drive.google.com/file/d/1kpN-yZKI2RAirD5GkG1tjyJMMyhwyZvC/view?usp=drive_link)
- [base_013469.pth](https://drive.google.com/file/d/16MCFOl6MSpACtFZ-VnCesLPoD2051y-d/view?usp=drive_link)
- [small_024.pth](https://drive.google.com/file/d/1hAlrazQHPmoouxll_uNCvMnYCmfdLay4/view?usp=drive_link)
- [small_0246.pth](https://drive.google.com/file/d/1F9rggdlcILbLz5UZr4fNxw_wuOdFLu50/view?usp=drive_link)
- [tiny_024.pth](https://drive.google.com/file/d/188xpKUKUfG-ks5jH_yiGYmczknPBCqnV/view?usp=drive_link)
- [tiny_0246.pth](https://drive.google.com/file/d/16HgVf_MgmEGXXek8F9Y1R__wsSxZAVYb/view?usp=drive_link)

## Performance
We found that the same code and checkpoint would produce different inference results using different pytorch versions. We still cannot figure out and welcome discussions.
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">Arch</th>
    <th class="tg-0pky" rowspan="2">Baseline</th>
    <th class="tg-c3ow" colspan="2">25%</th>
    <th class="tg-c3ow" colspan="2">30%</th>
    <th class="tg-c3ow" colspan="2">40%</th>
    <th class="tg-c3ow" colspan="2">50%</th>
  </tr>
  <tr>
    <th class="tg-c3ow">1.7.1</th>
    <th class="tg-c3ow">1.12.1</th>
    <th class="tg-c3ow">1.7.1</th>
    <th class="tg-c3ow">1.12.1</th>
    <th class="tg-c3ow">1.7.1</th>
    <th class="tg-c3ow">1.12.1</th>
    <th class="tg-c3ow">1.7.1</th>
    <th class="tg-c3ow">1.12.1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Base</td>
    <td class="tg-c3ow">81.8</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">81.83</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">81.77</span></td>
    <td class="tg-c3ow">81.33</td>
    <td class="tg-c3ow">81.46</td>
  </tr>
  <tr>
    <td class="tg-0pky">Small</td>
    <td class="tg-c3ow">79.9</td>
    <td class="tg-c3ow">80.31</td>
    <td class="tg-c3ow">80.33</td>
    <td class="tg-c3ow">79.90</td>
    <td class="tg-c3ow">79.89</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">Tiny</td>
    <td class="tg-c3ow">72.2</td>
    <td class="tg-c3ow">72.94</td>
    <td class="tg-c3ow">72.79</td>
    <td class="tg-c3ow">71.90</td>
    <td class="tg-c3ow">71.88</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</tbody>
</table>

We deploy the [ToMe](https://github.com/facebookresearch/ToMe) over the normal blocks (indexed by 0, 1, 2, ...). Typically, we use this technique on the normal block started by index 1 and its subsequent normal blocks. The model is evaluated with torch==1.12.1 .
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Arch</th>
    <th class="tg-0lax">Remove Ratio</th>
    <th class="tg-0pky">w/o ToMe</th>
    <th class="tg-c3ow">Started idx</th>
    <th class="tg-c3ow">r</th>
    <th class="tg-0lax">w ToMe</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">Base </td>
    <td class="tg-baqh" rowspan="2">40%</td>
    <td class="tg-c3ow" rowspan="2"><span style="font-weight:400;font-style:normal">81.77</span></td>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">24</td>
    <td class="tg-baqh">81.58</td>
  </tr>
  <tr>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">28</td>
    <td class="tg-baqh">81.42</td>
  </tr>
  <tr>
    <td class="tg-baqh">50%</td>
    <td class="tg-baqh">81.46</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">14</td>
    <td class="tg-baqh">81.28</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Small</td>
    <td class="tg-baqh">25%</td>
    <td class="tg-c3ow">80.33</td>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">22</td>
    <td class="tg-baqh">79.86</td>
  </tr>
  <tr>
    <td class="tg-baqh">30%</td>
    <td class="tg-baqh">79.89</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">19</td>
    <td class="tg-baqh">79.62</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Tiny</td>
    <td class="tg-baqh">25%</td>
    <td class="tg-c3ow">72.79</td>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">19</td>
    <td class="tg-baqh">72.35</td>
  </tr>
  <tr>
    <td class="tg-baqh">30%</td>
    <td class="tg-baqh">71.88</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">14</td>
    <td class="tg-baqh">71.7</td>
  </tr>
</tbody>
</table> 


## Before getting started
- Please download the [DeiT](https://github.com/facebookresearch/deit) checkpoints.
- Modify the ImageNet path in shell scripts, e.g. [script/shrink_base.sh](script/shrink_base.sh).

## Training
We use 8 GPUs with 256 images per GPU.

E.g.

```
./script/shrink_base.sh
```

## Testing
```
./script/test.sh
```

## Speed, Params & FLOPs
Please refer to [benchmark.py](benchmark.py) and run 

```
python benchmark.py
```

## To-Do
- [ ] Code for segmentation.
- [x] Upload checkpoints.

## Issues / Contact
Feel free to create an issue if you get a question or just drop
me emails ( sihao.lin@student.rmit.edu.au ). 

## Acknowledgement
This work is built upon [DeiT](https://github.com/facebookresearch/deit). Thanks to their awesome work.