import diffuser.utils as utils

STATE_EMBEDDING_DIM = 50
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

if args.dataset in ['Push-v0', 'push-v0']:
    termination_penalty=None 
    args.loader = 'datasets.SequenceDatasetPush'
else:
    termination_penalty=0

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    termination_penalty=termination_penalty,
    obs_type=args.obs_type, 
    data_dir= args.data_dir,
)

dataset = dataset_config()

if args.dataset not in ['Push-v0', 'push-v0']:
  render_config = utils.Config(
      args.renderer,
      savepath=(args.savepath, 'render_config.pkl'),
      env=args.dataset,
  )
else:
  renderer_config = renderer= dataset.env.scene


if args.dataset not in ['Push-v0', 'push-v0']:
  renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

if args.dataset in ['Push-v0', 'push-v0']:
    termination_penalty=None 
    args.loader = 'datasets.SequenceDatasetPush'
    cond_dim = STATE_EMBEDDING_DIM 
else:
    termination_penalty=0
    cond_dim = observation_dim

print(f'\n\n action_dim : {action_dim }, observation_dim: {observation_dim }\n\n' )

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#


model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    ## begin modified (changing transition dim to make it equal to action_dim)
    #transition_dim=observation_dim + action_dim,
    transition_dim=action_dim,
    ## enf added
    cond_dim=cond_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    obs_type = args.obs_type,
    obs_embed_dim=STATE_EMBEDDING_DIM ,
    path_pretrained_encoder=args.path_pretrained_encoder,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#


model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
#print("shape data set", dataset[0]['actions'].shape, dataset[0]['states'].shape)
batch = utils.batchify(dataset[0])
#print("batch observation", batch[0].shape, batch[1].shape)
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

