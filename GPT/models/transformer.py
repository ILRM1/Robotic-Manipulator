import torch
import torch.nn as nn
import torchvision
import transformers
from GPT.models.model import TrajectoryModel
from GPT.models.trajectory_gpt2 import GPT2Model

class Transformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.act_dim=act_dim
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # embedding network for inputs of transformer
        self.embed_color = torchvision.models.densenet121(pretrained=True)
        self.embed_color = torch.nn.Sequential(*(list(self.embed_color.children())[:-1]))

        self.embed_state = torch.nn.Sequential(*([nn.BatchNorm2d(1024),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(1024, 512, (3,5), stride=1, bias=False),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 256, (3,5), stride=1, bias=False),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, hidden_size, 2, stride=1, bias=False)
                                                  ])
                                               )


        # embedding layers for trajectories and timesteps
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_action = torch.nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(nn.Linear(hidden_size, act_dim))

    def forward(self, states, actions, timesteps, attention_mask=None):
        batch_size, seq_length, channel, width, height = \
            states.shape[0], states.shape[1], states.shape[2], states.shape[3], states.shape[4]
        states=states.reshape(-1,channel,width,height)

        # embed each image, trajectory and timestep
        color_embeddings = self.embed_color(states)
        state_embeddings = self.embed_state(color_embeddings)
        state_embeddings = state_embeddings.view(-1, self.hidden_size)
        state_embeddings = state_embeddings.reshape(batch_size, seq_length, self.hidden_size)
        action_embeddings = self.embed_action(actions)
        # time embeddings are treated similar to positional embeddings
        time_embeddings = self.embed_timestep(timesteps)

        # this makes the sequence look like (s_1, s_2, s_2, ..., s_10, a_! )
        # which works nice in an autoregressive sense since images predict trajectories
        stacked_inputs = torch.cat((state_embeddings, action_embeddings), dim=1)
        stacked_inputs = stacked_inputs + time_embeddings
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # get predictions
        action_preds = self.predict_action(x[:, -2])  # predict next action given state

        return action_preds

    def get_action(self, states, actions, timesteps, **kwargs):
        states = states.unsqueeze(0)
        actions = actions.reshape(1, -1, self.act_dim)
        timesteps = timesteps.reshape(1, -1)

        attention_mask = torch.cat([torch.ones(self.max_length)])
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

        action_preds = self.forward(states, actions, timesteps,attention_mask=attention_mask, **kwargs)

        return action_preds[0]