# RLSD sources and modifications

- **SDAR**, ZJU-REAL: https://github.com/ZJU-REAL/SDAR , Apache-2.0.
  Fixed commit: `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
  Based on veRL and verl-agent. Their notices remain in fetched source.
  Reproduction changes touch `agent_system/environments/env_manager.py`,
  `agent_system/environments/env_package/alfworld/envs.py`, and
  `verl/trainer/main_rlsd.py`: fixed-task validation with native one-actor-per-environment execution,
  and a read-only evaluation/logging subclass. An additional helper selects
  existing games. Native RLSD token advantage, warmdown, Actor/Teacher scoring, GRPO, skills and prompts
  are unmodified. See `patches/runtime.patch` and `configs/upstream.json`.
- **RLSD / Self-Distilled RLVR**: https://arxiv.org/abs/2604.03128 ,
  authors' repository https://github.com/iie-ycx/RLSD . This release uses SDAR's
  public ALFWorld reproduction of RLSD, including lambda mixing and its
  100-update warmdown. It does not redistribute the authors' visual-RL framework
  or claim that the authors provided the ALFWorld adaptation.
- **Qwen2.5-1.5B-Instruct**, Qwen: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , Apache-2.0.
- **Qwen2.5-3B-Instruct**, Qwen: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct , Qwen Research License.
  Model files are downloaded from pinned official revisions, not redistributed here.
- **ALFWorld**: https://github.com/alfworld/alfworld , obtain assets with the
  official downloader and follow the source code/data licenses.
- **Fixed seen140 task manifest**: same public manifest as
  https://github.com/leiye302/rethink-alfworld-qwen25-1.5b ; contains only public
  game-relative identities, hashes and environment seeds, not private trajectories.

Please cite Self-Distilled RLVR (arXiv:2604.03128) for the RLSD method and
SDAR for the ALFWorld baseline implementation. SDAR's citation is:

```bibtex
@article{lu2026sdar,
  title={Self-Distilled Agentic Reinforcement Learning},
  author={Lu, Zhengxi and Yao, Zhiyuan and Han, Zhuowen and Wang, Zi-Han and others},
  journal={arXiv preprint arXiv:2605.15155},
  year={2026}
}
```
