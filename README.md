# DiffWMA
This is the ource code of DiffWMA and our paper is submitted into "NeurIPS2025".
# Prerequisites
## Install dependencies
See ``requirments.txt`` file for more information about how to install the dependencies.
## Environments
The environments used in our paper are based on the [SMAC](https://github.com/oxwhirl/smac), [SMACv2](https://github.com/oxwhirl/smacv2), and [Google Research Football](https://link.zhihu.com/?target=https%3A//github.com/google-research/football). Please refer to them to know how to install.
# Usages

```python
python3 main.py --config=DiffWMA --env-config=sc2 with env_args.map_name=corridor
```
All results will be stored in the Results folder.

You can save the learnt models to disk by setting save_model = True, which is set to False by default. Learnt models can be loaded using the checkpoint_path parameter, after which the learning will proceed from the corresponding timestep.


# Acknowledgments
We want to express our gratitude to the authors of  [PyMARL](https://github.com/oxwhirl/pymarl) and [PyMARL2](https://github.com/hijkzzz/pymarl2) for publishing the source code. These are excellent source code for studying MARL. 


