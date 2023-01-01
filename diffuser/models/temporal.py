import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):
    ## begin modified: added as input the observation  dimension (obs_embed_dim)
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, obs_dim=10):
        super().__init__()
        #print("\n\n \tin_channels, out_channels \n\n", inp_channels, out_channels)
        # to be modified ?: replace Conv1dBlock(inp_channels, ...) by Conv1dBlock(inp_channels+obs_embed_dim,...)
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        ## begin modified: adding observation embedding
        self.observation_mlp = nn.Sequential(
            nn.Linear(obs_dim, obs_dim//2),
            nn.Mish(),
            nn.Linear(obs_dim//2, 1+obs_dim//3),
            Rearrange('batch t -> batch t 1'),
        )
        ## end added
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    # modified added input obs 
    def forward(self, x, obs, t):
        '''
            x : [ batch_size x inp_channels x horizon ] 
            t : [ batch_size x embed_dim ]

            # modified added input obs: obs: [batch_size x inp_obs_dim]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        ## begin modified (adding the observation embedding in the Residual block)
        #print("\n\n obs.shape, t.shape\n\n", obs.shape, t.shape)
        #obs= self.observation_mlp(obs) # [batch_size x (1+obs_dim//3) x 1]
        #horizon = x.size(2)
        #obs = obs.repeat(1,1,horizon)  # [batch_size x (1+obs_dim//3) x horizon]
        #out = torch.cat((x, obs), dim=1)
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

# begin modified: transition_dim = action_dim (and not anymore: transition_dim = action_dim+observation_dim)
class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim, # begin modified : cond_dim must be equal to observation_dim
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        obs_embed_dim=10, # modified : added observation embedding dimension
    ):
        super().__init__()
        ## modified: replaced 'dims = [transition_dim+obs_embed_dim,..]' by 'dims = [transition_dim+obs_embed_dim,...]'
        dims = [transition_dim+1+cond_dim//3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_dim = cond_dim
        ## begin modified (adding an embdding for observation)
        self.observation_mlp =nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.Mish(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, 1+cond_dim//3),
        )
        ## end modified
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon, obs_dim=cond_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )
    # modified: add 'obs' as input (observation at the first time step of the trajectory)
    def forward(self, x, obs, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        ## begin modified (added observation embedding)
        obs = self.observation_mlp(obs)
        obs = einops.rearrange(obs, 'batch t -> batch t 1')
        horizon = x.size(2)
        obs = obs.repeat(1,1,horizon)
        x = torch.cat((x, obs), dim=1)
        ## end modified
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, obs, t)
            x = resnet2(x, obs, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, obs, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, obs, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, obs, t)
            x = resnet2(x, obs, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
