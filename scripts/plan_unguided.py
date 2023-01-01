import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import torch
import einops
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

#value_experiment = utils.load_diffusion(
#    args.loadbase, args.dataset, args.value_loadpath,
#    epoch=args.value_epoch, seed=args.seed,
#)

## ensure that the diffusion model and value function are compatible with each other
#utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
#value_function = value_experiment.ema
#guide_config = utils.Config(args.guide, model=value_function, verbose=False)
#guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
"""
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
"""
logger = logger_config()
#policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
# modified begin added
@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    if type(cond)==dict:
      cond = cond[0]
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values

diffusion.normalizer=dataset.normalizer

def format_conditions(conditions, batch_size):
  conditions = utils.apply_dict(
    dataset.normalizer.normalize,
    conditions,
    'observations',
  )
  conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
  conditions = utils.apply_dict(
    einops.repeat,
    conditions,
    'd -> repeat d', repeat=batch_size,
  )
  return conditions

DEVICE = 'cuda:0'
DTYPE = torch.float

def normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)
    return (x_in - means) / stds


def de_normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)

def to_torch(x_in, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x_in) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x_in.items()}
	elif torch.is_tensor(x_in):
		return x_in.to(device).type(dtype)
	return torch.tensor(x_in, dtype=dtype, device=device)

preprocess_fn = get_policy_preprocess_fn(args.preprocess_fns)
## end added
env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()

    ## format current observation for conditioning
    #conditions = {0: observation}
    #conditions = {k: preprocess_fn(v) for k, v in conditions.items()}
    observation = dataset.normalizer(observation, 'observations')
    observation = observation[None].repeat(args.batch_size, axis=0)
    conditions = utils.to_torch(observation, dtype=torch.float32, device='cuda:0')
    #conditions = {0: observation}
    #conditions = format_conditions(conditions, args.batch_size)
    #obs = obs[None].repeat(n_samples, axis=0)
    #conditions = {
    #  0: to_torch(observation, device=DEVICE)
    #}
    #print("\n\n conditions.shape:", conditions.shape, "\n\n")
    ##action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    # run reverse diffusion process: samples=[a_t,...,a_{t+horizon}]
    ##samples = diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)
    samples = diffusion(
        conditions, sample_fn=default_sample_fn
    )
    
    actions = utils.to_np(samples.actions)


    ## extract action [ batch_size x horizon x transition_dim ]
    actions = dataset.normalizer.unnormalize(actions, 'actions')

    ## extract first action
    action = actions[0, 0]

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    #rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    #logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish_unguided(t, score, total_reward, terminal, diffusion_experiment)
