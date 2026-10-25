import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import checkpoint_complete, devices, load, protocol, tasks
from fixed_validation import fixed_validation_envs
from run import config
from setup import verify


def test_native_source_and_task_locks():
    lock = verify()
    assert len(lock['unchanged_files']) >= 10
    assert 'verl/workers/actor/dp_actor.py' in lock['unchanged_files']
    assert len(tasks()) == 140


def test_entrypoint_uses_rlsd_trainer_and_native_reweighting():
    launcher = (ROOT/'src/run.py').read_text(encoding='utf-8')
    adapter = (ROOT/'src/rlsd_eval.py').read_text(encoding='utf-8')
    entry = (ROOT/'.deps/sdar/verl/trainer/main_rlsd.py').read_text(encoding='utf-8')
    native = (ROOT/'.deps/sdar/verl/trainer/ppo/rlsd_ray_trainer.py').read_text(encoding='utf-8')
    assert 'from verl.trainer.main_rlsd import run_rlsd' in launcher
    assert 'from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer' in adapter
    assert 'class FixedSeenTrainer(RLSDRayTrainer)' in adapter
    assert 'from rlsd_eval import FixedSeenTrainer as RLSDRayTrainer' in entry
    assert 'token_advantages = compute_rlsd_token_advantage(' in native
    assert 'batch.batch["advantages"] = token_advantages' in native
    assert 'use_sdar_loss = True' not in entry


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_composed_native_recipe(size, tmp_path):
    cfg = config(size, tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    assert (cfg.trainer.n_gpus_per_node, cfg.ray_init.num_gpus) == (4, 4)
    assert cfg.algorithm.rlsd.rlsd_lambda == 0.5
    assert cfg.algorithm.rlsd.warmdown_steps == 100 and cfg.algorithm.rlsd.clip_eps == 0.2
    assert 'sdar' not in cfg.algorithm
    assert not cfg.actor_rollout_ref.actor.get('use_sdar_loss', False)
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert cfg.env.max_steps == 50 and cfg.env.history_length == 2
    assert cfg.data.max_response_length == 512
    assert cfg.data.val_batch_size == 140
    assert cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == 10 and cfg.trainer.save_freq == 20
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert 'local_hints' not in cfg.algorithm
    assert cfg.actor_rollout_ref.model.get('lora_rank', 0) == 0


@pytest.mark.parametrize('spec', ['0,1', '0,1,2,2', '0,1,2,3,4', '', '0,1,2,GPU-xxx'])
def test_invalid_gpu_allocation(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_four_gpu_allocation():
    assert devices('4,5,6,7') == ['4', '5', '6', '7']


def test_partial_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        checkpoint_complete(tmp_path)
    for rank in range(4):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    assert checkpoint_complete(tmp_path) == tmp_path


def test_telemetry_does_not_accept_duplicate_actions():
    assert protocol('<think>Plan.</think><action>look</action>')['strict_protocol']
    assert not protocol('<think></think><action>look</action>')['strict_protocol']
    assert not protocol('<think>Plan.</think><action>look</action><action>look</action>')['strict_protocol']


def test_fixed_task_uses_only_seen_and_preserves_base(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    game = tmp_path/'json_2.1.1/valid_seen/task/trial/game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_bytes(b'game')
    manifest = tmp_path/'manifest.json'
    row = {'task_id': game.relative_to(tmp_path).as_posix(), 'seed': 2026,
           'game_sha256': hashlib.sha256(b'game').hexdigest()}
    manifest.write_text(json.dumps([row]))
    base = SimpleNamespace(game_files=[str(game)], num_games=140)
    selected = fixed_validation_envs(base, manifest, 1, 1)
    assert selected[0].fixed_validation_seed == 2026
    assert base.num_games == 140 and selected[0] is not base
    row['game_sha256'] = '0'*64
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        fixed_validation_envs(base, manifest, 1, 1)


def native_rlsd_advantage():
    torch = pytest.importorskip('torch')
    source = ROOT/'.deps/sdar/verl/trainer/ppo/rlsd_utils.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_rlsd_token_advantage')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['compute_rlsd_token_advantage']


def test_native_rlsd_weight_clipping_direction_and_detach():
    torch = pytest.importorskip('torch')
    function = native_rlsd_advantage()
    advantage = torch.tensor([[2., 2., 2.], [-2., -2., -2.], [0., 0., 0.]])
    student = torch.full_like(advantage, -3., requires_grad=True)
    teacher = torch.tensor([[-1., -5., -1.]]*3, requires_grad=True)
    mask = torch.tensor([[1., 1., 0.]]*3)
    result = function(advantage, student, teacher, mask, rlsd_lambda=0.5, rlsd_clip_eps=0.2)
    assert torch.allclose(result, torch.tensor([[2.2,1.8,0.],[-1.8,-2.2,0.],[0.,0.,0.]]))
    assert not result.requires_grad and teacher.grad is None and student.grad is None
    # Exercise a policy update using these detached advantages: Teacher cannot receive a gradient.
    (-student*result).sum().backward()
    assert teacher.grad is None
    assert student.grad[0,0] < 0 and student.grad[1,0] > 0 and student.grad[:,2].sum() == 0


@pytest.mark.parametrize('coefficient', [0., 0.25, 0.5])
def test_native_rlsd_zero_teacher_gap_is_grpo(coefficient):
    torch = pytest.importorskip('torch')
    advantage = torch.tensor([[1., 1., 1.], [-1., -1., -1.]])
    log_probs = torch.full_like(advantage, -2.)
    mask = torch.tensor([[0.,1.,1.], [1.,1.,0.]])
    result = native_rlsd_advantage()(advantage, log_probs, log_probs, mask, rlsd_lambda=coefficient)
    assert torch.equal(result, advantage*mask)


@pytest.mark.parametrize('step,expected', [(0,0.5),(1,0.495),(50,0.25),(100,0.),(150,0.)])
def test_native_rlsd_warmdown(step, expected):
    source = ROOT/'.deps/sdar/verl/trainer/ppo/rlsd_ray_trainer.py'
    cls = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='RLSDRayTrainer')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rlsd_lambda')
    namespace = {}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'), namespace)
    instance = SimpleNamespace(rlsd_lambda_init=0.5, rlsd_lambda_warmdown_steps=100)
    assert namespace['_get_rlsd_lambda'](instance, step) == pytest.approx(expected)


def test_native_workers_preserve_seeds_order_and_results():
    source = ROOT/'.deps/sdar/agent_system/environments/env_package/alfworld/envs.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'AlfworldWorker', 'AlfworldEnvs'}]

    class NativeEnvironment:
        def seed(self, seed): self.value = seed
        def reset(self): return [str(self.value)], {'admissible_commands': [['look']]}
        def step(self, actions):
            return [str(self.value)+actions[0]], [0], [False], {'admissible_commands': [['look']], 'won': [False]}

    class Base:
        def init_env(self, batch_size): return NativeEnvironment()

    class Method:
        def __init__(self, fn): self.remote = fn

    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return Method(getattr(self.obj, name))

    class Remote:
        def __init__(self, cls): self.cls = cls
        def remote(self, *args): return Handle(self.cls(*args))

    ray = SimpleNamespace(is_initialized=lambda: True, get=lambda x:x,
                          remote=lambda **kwargs: lambda cls:Remote(cls))
    namespace = {'gym': SimpleNamespace(Env=object), 'ray': ray, 'get_environment': lambda _: lambda *a,**k:Base(),
                 'load_config_file': lambda _: {'env': {'type': 'AlfredTWEnv'}},
                 'compute_reward': lambda info,multi_modal: 0}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    env = namespace['AlfworldEnvs']('unused', 17, 3, 2, {})
    assert len(env.workers) == 6
    assert not hasattr(env, 'pool_size')
    reset = copy.deepcopy(env.reset())
    step = copy.deepcopy(env.step(['look'] * 6))
    assert reset[0] == ['17', '17', '18', '18', '19', '19']
    assert step[0] == ['17look', '17look', '18look', '18look', '19look', '19look']


def test_dry_run_needs_no_model_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train', '--size', '1.5b',
                             '--run-dir', str(tmp_path/'unused'), '--dry-run'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'rlsd_lambda: 0.5' in result.stdout and 'warmdown_steps: 100' in result.stdout


def test_models_share_all_scientific_settings(tmp_path):
    from omegaconf import OmegaConf
    left = config('1.5b', tmp_path/'model15', tmp_path/'assets', tmp_path/'run15')
    right = config('3b', tmp_path/'model3', tmp_path/'assets', tmp_path/'run3')
    for cfg in (left, right):
        cfg.actor_rollout_ref.model.path = 'model'
        cfg.critic.model.tokenizer_path = 'model'
        cfg.reward_model.model.input_tokenizer = 'model'
        cfg.trainer.experiment_name = 'experiment'
        cfg.data.train_files = cfg.data.val_files = 'parquet'
        cfg.trainer.default_local_dir = cfg.trainer.validation_data_dir = cfg.trainer.rollout_data_dir = 'output'
        cfg.ray_init.runtime_env.env_vars.RLSD_RUN_DIR = 'output'
    assert OmegaConf.to_container(left, resolve=True) == OmegaConf.to_container(right, resolve=True)


def test_evaluation_hook_preserves_native_success_and_all_tasks(tmp_path, monkeypatch):
    import importlib.util
    import types
    torch = pytest.importorskip('torch')
    rows = tasks()
    gamefiles = [str(tmp_path/row['task_id']) for row in rows]
    batch = SimpleNamespace(
        batch={'responses': torch.ones((140, 2), dtype=torch.long),
               'attention_mask': torch.ones((140, 4), dtype=torch.long)},
        non_tensor_batch={'gamefile': gamefiles, 'traj_uid': [str(i) for i in range(140)],
                          'turn_step': [0]*140, 'episode_rewards': [10.0]*70+[0.0]*70,
                          'is_action_valid': [True]*140})

    class NativeTrainer:
        def _validate(self):
            self.traj_collector.multi_turn_loop()
            return {'val/success_rate': 0.5}

    native_module = types.ModuleType('verl.trainer.ppo.rlsd_ray_trainer')
    native_module.RLSDRayTrainer = NativeTrainer
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.rlsd_ray_trainer', native_module)
    spec = importlib.util.spec_from_file_location('tested_rlsd_eval', ROOT/'src/rlsd_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    monkeypatch.setenv('RLSD_RUN_DIR', str(tmp_path/'run'))
    trainer = module.FixedSeenTrainer()
    trainer.global_steps = 10
    original = lambda:batch
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=original)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kw:'<think>Plan.</think><action>look</action>')
    assert trainer._validate() == {'val/success_rate':0.5}
    assert trainer.traj_collector.multi_turn_loop is original
    summary = json.loads((tmp_path/'run/evaluation/update_010/summary.json').read_text())
    assert summary['successes'] == 70 and len(summary['task_results']) == 140
    assert summary['reasoning_complete_rate'] == 1
