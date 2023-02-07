import torch
import torch.nn.functional as F
import json
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from transformers import (AdamW, BartTokenizer, WEIGHTS_NAME, CONFIG_NAME, get_linear_schedule_with_warmup, BartForSequenceClassification)
from random import shuffle
import re
import time
from tqdm import tqdm
import traceback
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), 'damd_multiwoz'))
from utils import Vocab, CustomizedBartClassificationHead, CustomizedBartEncoder
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random
from damd_multiwoz.config import global_config as cfg

from typing import Optional

DEFAULT_EPS = 1e-10


class BartTokenizer(BartTokenizer):
    def encode(self,text,add_special_tokens=False):
        encoded_inputs = self.encode_plus(text,add_special_tokens=False)
        return encoded_inputs["input_ids"]


class BartRewardModel(BartForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.classification_head = CustomizedBartClassificationHead(
            config.d_model, config.d_model, config.num_labels,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.model.encoder = CustomizedBartEncoder(config, self.model.shared)

    def resize_token_embeddings(self, new_num_tokens: int):
        print(f"\nUse customized resize_token_embeddings", flush=True)
        new_embeddings = self.model.resize_token_embeddings(new_num_tokens)
        self.shared = new_embeddings

    def tie_decoder(self):
        print(f"\nUse customized tie_decoder", flush=True)
        self.shared.padding_idx = self.config.pad_token_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        output = self.model.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )[0]                                                            # last hidden state, (batch_size x MAX_TIME_STEP, MAX_STATE_LEN + MAX_ACT_LEN + MAX_GOAL_LEN, embedding_size)
        sentence_representation = output.mean(axis=-2)                  # sentence_representation: (batch_size x MAX_TIME_STEP, embedding_size)
        logits = self.classification_head(sentence_representation)      # (batch_size x MAX_TIME_STEP, 1)
        logits = torch.sigmoid(logits).nan_to_num(nan=0.0)              # scale rewards to [0,1], replace nan with 0.0

        return (logits,)

    def get_embeddings(self, inputs):
        """
        Transform inputs to the BART embedding
        Example:
            inputs = torch.LongTensor([[2,3,4],[1,5,6]])
            emb = self.model.shared
            oh = F.one_hot(inputs, emb.weight.shape[0]).float()
            print(oh.shape)
            assert (oh.argmax(-1) == inputs).all()
            assert (oh @ emb.weight == emb(inputs)).all()
        """
        if len(inputs.shape) == 2:
            # inputs are token ids (integer)
            # inputs: (batch_size, seq_len)
            assert not torch.is_floating_point(inputs)
            input_embeds = self.model.shared(inputs)
        elif len(inputs.shape) == 3:
            # inputs are one-hot tensor over vocab (float)
            # inputs: (batch_size, seq_len, vocab_size)
            assert torch.is_floating_point(inputs)
            # input_embeds: (batch_size, seq_len, vocab_size) x (vocab_size, hidden_size) = (batch_size, seq_len, hidden_size)
            input_embeds = inputs @ self.model.shared.weight
        else:
            raise ValueError(f"inputs should have dimension 2 or 3, received shape {inputs.shape}")

        return input_embeds


def numpy2torch(x):
    if np.issubdtype(x.dtype, np.int):
        return torch.LongTensor(x)
    elif np.issubdtype(x.dtype, np.float):
        return torch.FloatTensor(x)
    else:
        raise NotImplementedError(f"Error: type of x is {x.dtype}")


def numpyBinaryTuple2torch(x):
    return numpy2torch(x[0]).cuda(), numpy2torch(x[1]).cuda()


def binaryTupleList2Tensor(tuple_list):
    return torch.stack([x[0] for x in tuple_list]), torch.stack([x[1] for x in tuple_list])


def listNetLoss(y_pred, y_true):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :return: loss value, a torch.Tensor
    """
    if cfg.listnet_power == 0:
        preds_smax = F.softmax(y_pred, dim=1)
    else:
        preds_smax = y_pred.pow(cfg.listnet_power)
        preds_smax = preds_smax / (preds_smax.sum(dim=-1, keepdim=True) + DEFAULT_EPS)

    preds_smax = preds_smax + DEFAULT_EPS
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(y_true * preds_log, dim=1))


def listMLELoss(y_pred, y_true):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    y_pred = y_pred / cfg.listmle_temp

    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))


def listMLELossEscort(y_pred, y_true):
    """
    ListMLE loss with escort transformation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    y_pred = torch.pow(y_pred, cfg.listnet_power)

    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true / (max_pred_values + DEFAULT_EPS)

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + DEFAULT_EPS) - torch.log(preds_sorted_by_true_minus_max + DEFAULT_EPS)

    return torch.mean(torch.sum(observation_loss, dim=1))


class RewardLearning(object):
    
    def __init__(self, fold, action_space, metric, args, reward_loss, test=False):
        self.reward_loss = reward_loss
        print(f"\n[RewardLearning] Use reward_loss: {self.reward_loss}; reward_learning_samples: {cfg.reward_learning_samples}; escort power: {cfg.listnet_power} \n", flush=True)
        if test:
            print(f"\n[RewardLearning] Load trained model from {args.model_path} \n", flush=True)
        self.reward_report_template = 'reward_report_{}_{}_.*.csv'
        self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        # use regression mode of Bart
        self.model = BartRewardModel.from_pretrained(
            args.model_path if test else args.pretrained_checkpoint,
            num_labels=1,
        )
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        self.model.to(args.device)
        self.vocab = Vocab(self.model, self.tokenizer)
        self.test = test

        self.train_val_fraction=0.8
        self.MAX_TIME_STEP=20
        self.MAX_GOAL_LEN=50
        self.MAX_STATE_LEN=50
        self.MAX_ACT_LEN=50

        self.fold = fold
        self.metric = metric
        self.TRAIN_ON = action_space

        self.root_path = './damd_multiwoz'
        self.dataset=json.loads(open(os.path.join(self.root_path, f'{cfg.data_folder}/multi-woz-processed/data_for_damd_reward_{self.fold}.json'),'r').read())

        self.reward_folder_path= os.path.join(self.root_path,f'{cfg.data_folder}/multi-woz-oppe/reward')
        self.data_for_damd = json.loads(open(os.path.join(self.root_path,f'{cfg.data_folder}/multi-woz-processed/data_for_damd.json'), 'r').read())
        
        self.processed_reward_rollouts = None
        self.embed_cache = {}

        self._train_step = 0

    def load_model(self, model_path=None, device=None):
        if model_path is None:
            model_path = self.args.model_path
        if device is None:
            device = self.args.device
        print(f"Load trained model from {model_path} to device: {device}", flush=True)
        self.model = BartRewardModel.from_pretrained(
            model_path,
            num_labels=1,  # regression mode
        )
        self.model.to(device)

    def metric_score(self, sucess,match,bleu):
        return sucess + match + 2 * bleu / 100
        
    def load_reward_rollouts(self):
        reward_record_file_prefix = self.reward_report_template.format(self.fold, self.metric)
        print('reward_record_file_prefix:',reward_record_file_prefix, flush=True)
        rollouts_processed = {}
        for file in os.listdir(self.reward_folder_path):
            if re.search(reward_record_file_prefix,file):
                print('file:',file, flush=True)
                reward_record_path = os.path.join(self.reward_folder_path,file)
                df = pd.read_csv(reward_record_path)
                for _,row in df.iterrows():
                    # each row is a synthetic dialogue
                    dial_id = row['dial_id']
                    rollout = json.loads(row['rollout'])
                    turn_nums = [int(z) for z in rollout.keys()]
                    turn_nums = sorted(turn_nums)
        
                    if dial_id not in rollouts_processed:
                        rollouts_processed[dial_id]={}
                        rollouts_processed[dial_id]['gen']=[]
                    
                    dia_rollout={}
                    rollouts_processed[dial_id]['gen'].append(dia_rollout)
                    dia_rollout['score'] = self.metric_score(row['success'],row['match'],row['bleu'])
                    
                    dia_rollout['rollout']=[]
                    for turn_num in turn_nums:
                        true_act_prob = [1.]
                        if 'aspn_prob' in rollout[str(turn_num)]:
                            true_act_prob = np.exp(rollout[str(turn_num)]['aspn_prob']).tolist()
                        dia_rollout['rollout'].append({
                            'tn':turn_num,
                            'act':rollout[str(turn_num)]['aspn_gen'],
                            'true_act':rollout[str(turn_num)]['aspn'],
                            'resp':rollout[str(turn_num)]['resp_gen'],
                            'true_act_prob':true_act_prob               # prob of the model getting true act
                        })
                    
                    if 'gt' not in rollouts_processed[dial_id]:
                        rollouts_processed[dial_id]['gt']={}
                        rollouts_processed[dial_id]['gt']['score']=4
                        rollouts_processed[dial_id]['gt']['rollout']=[]
                        for turn_num in turn_nums:
                            rollouts_processed[dial_id]['gt']['rollout'].append({
                                'tn':turn_num,
                                'act':rollout[str(turn_num)]['aspn'],
                                'resp':rollout[str(turn_num)]['resp'],
                                'true_act':rollout[str(turn_num)]['aspn'],
                                'true_act_prob':[1]
                            })
                            
        self.processed_reward_rollouts = rollouts_processed
        self.dial_ids = list(self.processed_reward_rollouts.keys())
        self.load_gt_dia_logs(self.dial_ids)
        return rollouts_processed

    def load_gt_dia_logs(self, dial_ids):
        gt_dia_logs={}
        for dial_id in dial_ids:
            goal = self.goal_as_st(self.dataset[dial_id]['goal'])
            gt_dia_log={
                'goal':goal
            }
            gt_dia_logs[dial_id]=gt_dia_log
            for turn in self.dataset[dial_id]['log']:
                gt_dia_log[turn['turn_num']]={}
                gt_dia_log[turn['turn_num']]['state'] = turn['cons_delex']
                # "cons_delex": https://github.com/TonyNemo/UBAR-MultiWOZ/blob/master/preprocess.py L337
                
        self.gt_dia_logs = gt_dia_logs

    def goal_as_st(self, goal):
        return str(goal).replace("'",' ')\
                        .replace(',',' , ').replace('{',' ')\
                        .replace('}',' ').replace('  ',' ')

    def padInput(self, sequences, max_len, use_dynamic_pad_len=False):
        pad_token = self.vocab.tokenizer.encode("<pad>")[0]
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        if use_dynamic_pad_len:
            max_len = min(max_len, max(lengths))
        input_ids = np.ones((num_samples, max_len), dtype=np.long) * pad_token
        masks = np.zeros((num_samples, max_len), dtype=np.long)

        for idx, s in enumerate(sequences):
            trunc = s[-max_len:]
            input_ids[idx, :lengths[idx]] = trunc
            masks[idx, :lengths[idx]] = 1
        return input_ids, masks

    def pad_time_step(self, sentence_embeds):
        max_seq_len = sentence_embeds.shape[-1]
        sentence_embeds = sentence_embeds[:self.MAX_TIME_STEP]
        time_padded_sentences = np.array(sentence_embeds)
        if self.MAX_TIME_STEP>len(sentence_embeds):
            pad = np.zeros((self.MAX_TIME_STEP-len(sentence_embeds),max_seq_len), dtype=np.long)
            time_padded_sentences = np.concatenate([sentence_embeds,pad])
        return time_padded_sentences

    def encode_resp(self, resp):
        # encode the system response, from utils.py L285
        # return a list of tokens
        return self.vocab.tokenizer.encode(resp) + self.vocab.tokenizer.encode('<eos_r>')

    def encode_state(self, state):
        # There is a bug if state is empty
        # modified source code based on https://github.com/huggingface/transformers/pull/4209
        # return a list of tokens
        return self.vocab.tokenizer.encode(state) + self.vocab.tokenizer.encode('<eos_b>')

    def sample_roll_out(self, dial_id):
        gen_rollouts_info = self.processed_reward_rollouts[dial_id]['gen']      # a list of all generated trajectories for the given dial_id (type: list[dict])
        gt_rollout_info = self.processed_reward_rollouts[dial_id]['gt']         # unique ground truth trajectory (type: dict)
        # only need to change size=2 to N (if num_of_candidates < N, use replace=True, o.w. use replace=False)
        candidates = gen_rollouts_info + [gt_rollout_info]
        rollout_infos = np.random.choice(candidates, size=cfg.reward_learning_samples, replace=(len(candidates) < cfg.reward_learning_samples))
        dia_log= self.gt_dia_logs[dial_id]
        goal = dia_log['goal']
        
        goal = [self.encode_state(goal)]
        goal_input_ids, goal_mask = self.padInput(goal, self.MAX_GOAL_LEN)      # ids, mask: np.array (1, MAX_GOAL_LEN)
        goal = (goal_input_ids, goal_mask)                                      # tuple: np.array (1, MAX_GOAL_LEN)
        for g in goal:
            assert isinstance(g, np.ndarray) and g.shape == (1, self.MAX_GOAL_LEN)

        
        rollout_pairs = []
        for rollout_info in rollout_infos:
            acts = []
            states = []
            for turn in rollout_info['rollout']:
                tn = turn['tn']
                act = turn[self.TRAIN_ON]   # turn['resp']
                
                if tn not in self.gt_dia_logs[dial_id]:
                    break
                
                state = self.gt_dia_logs[dial_id][tn]['state']

                state = self.encode_state(state)    # list(sentence_len)
                act = self.encode_resp(act)         # list(sentence_len)

                # act within acts does not have equal len at this moment
                acts.append(act)
                states.append(state)
            # pad sentences in states, acts and goal to get input_ids and masks
            states_input_ids, states_masks = self.padInput(states, self.MAX_STATE_LEN)      # np.array (dialog_len, MAX_STATE_LEN)
            acts_input_ids, acts_masks = self.padInput(acts, self.MAX_ACT_LEN)              # np.array (dialog_len, MAX_ACT_LEN)

            # pad time step for input_ids and masks for states and acts
            states = self.pad_time_step(states_input_ids), self.pad_time_step(states_masks)     # tuple: np.array (MAX_TIME_STEP, MAX_STATE_LEN)
            acts = self.pad_time_step(acts_input_ids), self.pad_time_step(acts_masks)           # tuple: np.array (MAX_TIME_STEP, MAX_ACT_LEN)

            # check dimensions
            for s in states:
                assert isinstance(s, np.ndarray) and s.shape == (self.MAX_TIME_STEP, self.MAX_STATE_LEN)
            for a in acts:
                assert isinstance(a, np.ndarray) and a.shape == (self.MAX_TIME_STEP, self.MAX_ACT_LEN)

            score=rollout_info['score']
            rollout_pairs.append([goal,states,acts,score])      # [tuple(np.array), tuple(np.array), tuple(np.array), float]
            # each element of rollout_pairs is a [goal,states,acts,score] list, len(rollout_pairs) = len(rollout_infos)
            # goal, states, acts are all tuples of input_ids and masks

        # prob = s1 / (s1 + s2)
        if self.reward_loss == "listNet":
            score_sum = sum([rollout_pair[-1] for rollout_pair in rollout_pairs]) + DEFAULT_EPS
            for rollout_pair in rollout_pairs:
                rollout_pair[-1] /= score_sum

        return rollout_pairs

    def get_data_gen(self, sample_roll_out):
        def data_gen(dial_ids,batch_size):
            try:
                all_s = [[] for _ in range(cfg.reward_learning_samples)]  # [s1s, s2s, s3s, ...]
                all_a = [[] for _ in range(cfg.reward_learning_samples)]  # [a1s, a2s, a3s, ...]
                all_g = [[] for _ in range(cfg.reward_learning_samples)]  # [g1s, g2s, g3s, ...]

                probs = []
                while True:
                    shuffle(dial_ids)
                    for dial_id in dial_ids:
                        rollout_pairs = sample_roll_out(dial_id)
                        probs.append([])
                        for idx, pair in enumerate(rollout_pairs):
                            goal, state, action, prob = pair
                            # all_s[idx], all_a[idx], all_g[idx]: list of tuple(np.array)
                            # probs: list[batch_size] of list[reward_learning_samples]
                            all_s[idx].append(state)
                            all_a[idx].append(action)
                            all_g[idx].append(goal)
                            probs[-1].append(prob)

                        if len(all_s[0]) >= batch_size:
                            probs = np.array(probs)     # np.array(batch_size, reward_learning_samples)
                            yield all_s, all_a, all_g, probs

                            all_s = [[] for _ in range(cfg.reward_learning_samples)]
                            all_a = [[] for _ in range(cfg.reward_learning_samples)]
                            all_g = [[] for _ in range(cfg.reward_learning_samples)]

                            probs = []

            except Exception as e:
                print(traceback.format_exc())
                raise e

        return data_gen

    def get_reward(self, input_seq):
        g = []
        s = []
        a = []
        for goal, state, aspn, resp in input_seq:
        
            state_token_embeds = self.encode_state(state)               # list(sentence_len)
            s.append(state_token_embeds)
    
            if self.TRAIN_ON=='act':
                action = aspn
            elif self.TRAIN_ON=='resp':
                action = resp
            else:
                raise Exception('Invalid TRAIN_ON selection')
            action_token_embeds = self.encode_resp(action)              # list(sentence_len)
            a.append(action_token_embeds)
    
            goal_token_embeds = self.encode_state(goal)                 # list(sentence_len)
            g.append(goal_token_embeds)

        states_input_ids, states_masks = self.padInput(s, self.MAX_STATE_LEN)       # np.array (dialog_len, MAX_STATE_LEN)
        acts_input_ids, acts_masks = self.padInput(a, self.MAX_ACT_LEN)             # np.array (dialog_len, MAX_ACT_LEN)
        goal_input_ids, goal_mask = self.padInput(g, self.MAX_GOAL_LEN)             # np.array (dialog_len, MAX_GOAL_LEN)

        states = (states_input_ids, states_masks)
        acts = (acts_input_ids, acts_masks)
        goals = (goal_input_ids, goal_mask)

        rewards = self.model_forward([numpyBinaryTuple2torch(states)], [numpyBinaryTuple2torch(acts)], [numpyBinaryTuple2torch(goals)], test_mode=True)         # (1, time_steps)
        rewards = rewards.view(-1).cpu().data.numpy()           # (time_steps,)

        return rewards

    def get_Gs(self,  gamma=0.9):
        self.model.eval()
        fn_Gs = {}
        num_fns = len(self.data_for_damd.keys())
        for ex_num,fn in enumerate(tqdm(reversed(list(self.data_for_damd.keys())),total=num_fns)):
            fn_Gs[fn] = {}
            goal = self.goal_as_st(self.data_for_damd[fn]['goal'])
            
            turn_num_inp_seq = {}
            for turn in self.data_for_damd[fn]['log']:
                turn_num = turn['turn_num']
                resp = turn['resp']
                state = turn['cons_delex']
                aspn = turn['sys_act']
                
                turn_num_inp_seq[turn_num]=[goal,state,aspn,resp]
                
            reverse_turn_nums = sorted(list(turn_num_inp_seq.keys()),reverse=True)
            inp_seq = []
            for turn_num in reverse_turn_nums:
                inp_seq.append(turn_num_inp_seq[turn_num])

            rewards = self.get_reward(inp_seq)
            assert len(rewards) == len(turn_num_inp_seq.keys()) == len(reverse_turn_nums)
            G = 0
            for turn_num,reward in zip(reverse_turn_nums,rewards):
                G = reward + gamma*G
                fn_Gs[fn][turn_num] = {
                    'G':G,
                    'gamma':gamma
                }
        self.model.train()
        return fn_Gs

    def _get_input_to_model(self, data_gen):
        all_s, all_a, all_g, probs = next(data_gen)
        # basically compare synthetic rollouts with the ground truth trajectory,
        # all_s, all_a, all_g: list[reward_learning_samples] of list[batch_size] of binary tuple(np.array)

        assert len(all_s) == len(all_a) == len(all_g) == probs.shape[1] == cfg.reward_learning_samples
        assert len(all_s[0]) == len(all_a[0]) == len(all_g[0]) == probs.shape[0] == cfg.batch_size

        probs = numpy2torch(probs).cuda()       # torch.tensor(batch_size, reward_learning_samples)

        return all_s, all_a, all_g, probs

    def model_forward(self, state_tuples_list, action_tuples_list, goal_tuples_list, test_mode=False):
        # wrapper of forward pass of the BART model
        # state_tuples_list, action_tuples_list, goal_tuples_list: list[batch_size] of binary tuples(torch.tensor).
        batch_size = len(state_tuples_list)
        # state_tuples_list[0]: tuple: torch.tensor(MAX_TIME_STEP, MAX_STATE_LEN)
        time_steps = state_tuples_list[0][0].shape[0] if test_mode else self.MAX_TIME_STEP
        if test_mode:
            assert batch_size == 1

        # shape: (batch_size, time_steps, MAX_STATE_LEN)
        state_input_ids, state_input_mask = binaryTupleList2Tensor(state_tuples_list)
        action_input_ids, action_input_mask = binaryTupleList2Tensor(action_tuples_list)
        goal_input_ids, goal_input_mask = binaryTupleList2Tensor(goal_tuples_list)

        assert state_input_ids.shape == state_input_mask.shape == (batch_size, time_steps, self.MAX_STATE_LEN)
        assert action_input_ids.shape == action_input_mask.shape == (batch_size, time_steps, self.MAX_ACT_LEN)
        assert goal_input_ids.shape == goal_input_mask.shape == (batch_size, time_steps, self.MAX_GOAL_LEN)

        # (batch_size x time_steps, MAX_STATE_LEN + MAX_ACT_LEN + MAX_GOAL_LEN)
        sag_len = self.MAX_STATE_LEN + self.MAX_ACT_LEN + self.MAX_GOAL_LEN
        input_ids = torch.cat([state_input_ids, action_input_ids, goal_input_ids], dim=-1).reshape(-1, sag_len)
        attention_mask = torch.cat([state_input_mask, action_input_mask, goal_input_mask], dim=-1).reshape(-1, sag_len)

        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # return the reward for each turn
        rewards = output[0]
        assert rewards.shape == (batch_size * time_steps, 1)

        return rewards.reshape(batch_size, -1)      # (batch_size, time_steps)

    def _get_rollout_returns(self, state_tuples_list, action_tuples_list, unrepeated_goal_tuples_list):
        # repeat the goal (g) for each turn in the dialogue
        goal_tuple_list = [(
            g_tuple[0].repeat_interleave(self.MAX_TIME_STEP, dim=0),
            g_tuple[1].repeat_interleave(self.MAX_TIME_STEP, dim=0)
        ) for g_tuple in unrepeated_goal_tuples_list]   # g_tuple: tuple: torch.tensor(1, MAX_GOAL_LEN)
        # goal_tuple_list: list[batch_size] of binary tuples of torch.tensor(MAX_TIME_STEP, MAX_GOAL_LEN)

        rews = self.model_forward(state_tuples_list, action_tuples_list, goal_tuple_list)     # (batch_size, self.MAX_TIME_STEP)

        return torch.sum(rews, dim=-1, keepdim=True)            # (batch_size, 1)

    def _get_reward_loss(self, all_s, all_a, all_g, probs):
        all_chi = []

        for state, action, goal in zip(all_s, all_a, all_g):
            # state, action, goal: list[batch_size] of binary tuple(np.array)
            # from numpy array to torch tensor on GPU
            state = [numpyBinaryTuple2torch(x) for x in state]
            action = [numpyBinaryTuple2torch(x) for x in action]
            goal = [numpyBinaryTuple2torch(x) for x in goal]

            all_chi.append(
                self._get_rollout_returns(state, action, goal)                  # (batch_size, 1)
            )

        chi = torch.cat(all_chi, dim=-1)                            # (batch_size, reward_learning_samples)

        if self.reward_loss == "listNet":
            loss = listNetLoss(y_pred=chi, y_true=probs)
        elif self.reward_loss == "listMLE":
            if cfg.listnet_power == 0:
                # use original listMLE loss
                loss = listMLELoss(y_pred=chi, y_true=probs)
            else:
                # Use ListMLELossEscort
                loss = listMLELossEscort(y_pred=chi, y_true=probs)
        else:
            raise NotImplementedError

        return loss

    def _train_one_epoch(self, data_gen, num_steps, epoch_count):
        self.model.train()
        log_train_loss = 0.
        start_time = time.time()
        for idx in range(num_steps):
            all_s, all_a, all_g, probs = self._get_input_to_model(data_gen)
            loss = self._get_reward_loss(all_s, all_a, all_g, probs) / self.args.gradient_accumulation_steps
            loss.backward()
            if self._train_step % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optim.step()
                self.optim.zero_grad()

            self._train_step += 1
            rew_loss = float(loss.item()) * self.args.gradient_accumulation_steps
            log_train_loss += rew_loss

            if self._train_step % cfg.report_interval == 0:
                print(
                    f"[{epoch_count}|{idx+1}/{num_steps}] total iter: {self._train_step}, minibatch loss: {rew_loss:.4f}, average train loss: {log_train_loss/(idx+1):.4f}, time: {(time.time()-start_time)/60:.1f} min",
                    flush=True
                )

        return log_train_loss / num_steps

    def _calculate_validation_loss(self, data_gen, num_steps):
        self.model.eval()
        log_valid_loss = 0.
        with torch.no_grad():
            for _ in range(num_steps):
                all_s, all_a, all_g, probs = self._get_input_to_model(data_gen)
                loss = self._get_reward_loss(all_s, all_a, all_g, probs)
                log_valid_loss += float(loss.item()) / num_steps

        self.model.train()
        return log_valid_loss

    def train_model(self):
        shuffle(self.dial_ids)
        train_dial_ids = self.dial_ids[:int(len(self.dial_ids) * self.train_val_fraction)]
        val_dial_ids = self.dial_ids[int(len(self.dial_ids) * self.train_val_fraction):]
        
        train_num_examples = len(train_dial_ids)
        valid_num_examples = len(val_dial_ids)

        print(f"train_val_fraction: {self.train_val_fraction}, batch_size: {cfg.batch_size}", flush=True)
        print('train_num_examples:',train_num_examples, flush=True)
        print('valid_num_examples:',valid_num_examples, flush=True)

        train_num_examples_per_epoch = max(3,int((train_num_examples/cfg.batch_size)/5))
        valid_num_examples_per_epoch = max(1,int((valid_num_examples/cfg.batch_size)/2))
        
        train_data_gen = self.get_data_gen(self.sample_roll_out)(train_dial_ids, cfg.batch_size)
        val_data_gen = self.get_data_gen(self.sample_roll_out)(val_dial_ids, cfg.batch_size)

        prev_min_loss = 1e10
        print(f"vocab_size: {self.model.config.vocab_size}", flush=True)
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        self.tokenizer.save_pretrained(self.args.model_path)
        # CONFIG_NAME: 'config.json'
        self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode="min",
            factor=args.lr_decay,
            patience=cfg.weight_decay_count,
            min_lr=0.000001,
            verbose=True
        )

        # maximum_training_steps = 10000
        start_time = time.time()
        for epoch in range(cfg.epoch_num):
            _ = self._train_one_epoch(train_data_gen, train_num_examples_per_epoch, epoch)
            valid_loss = self._calculate_validation_loss(val_data_gen, valid_num_examples_per_epoch)
            scheduler.step(valid_loss)
            print(
                f"[Epoch {epoch + 1}] total iter: {self._train_step}, valid loss: {valid_loss:.4f}, prev min loss: {prev_min_loss:.4f}, time: {(time.time() - start_time)/60:.1f} min",
                flush=True
            )

            if valid_loss < prev_min_loss * (1 - 1e-4):
                early_stop_count = cfg.early_stop_count
                prev_min_loss = valid_loss
                # WEIGHTS_NAME: 'pytorch_model.bin'
                save_loc = os.path.join(self.args.model_path, WEIGHTS_NAME)
                torch.save(self.model.state_dict(), save_loc)
                print(f'[Epoch {epoch + 1}] Model saved to {save_loc}', flush=True)
            else:
                # EarlyStopping and Learning Rate Scheduling
                early_stop_count -= 1
                print(f'[Epoch {epoch + 1}] early stop countdown {early_stop_count}/{cfg.early_stop_count}', flush=True)

                if early_stop_count == 0:
                    break

        # load the saved best model
        self.load_model()
        valid_loss = self._calculate_validation_loss(val_data_gen, valid_num_examples_per_epoch)
        print(
            f"[Loaded Model] total iter: {self._train_step}, valid loss: {valid_loss:.4f}, time: {(time.time()-start_time)/60:.1f} min",
            flush=True
        )
        return

    def save_returns(self, gamma=0.):
        num_fns = len(self.data_for_damd.keys())
        fn_Gs = self.get_Gs(gamma=gamma)
        fn_G_file_name = 'fn_Gs_{}_{}_{}_{}.json'.format(self.fold, gamma, self.TRAIN_ON, self.metric)
        
        print(fn_G_file_name, flush=True)
        fn_Gs_file_path = os.path.join(self.root_path,f'{cfg.data_folder}','multi-woz-oppe',fn_G_file_name)
        print('fn_Gs_file_path:',fn_Gs_file_path, flush=True)
        with open(fn_Gs_file_path,'w') as f:
            json.dump(fn_Gs,f)


def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", default=11, type=int, help="seed")
    parser.add_argument("-K", "--folds", dest="folds", default=10, type=int, help="Number of folds")
    parser.add_argument("-a", "--action_space", dest="action_space", choices={"resp"}, default='resp', help="action space. should be resp")
    parser.add_argument("-m", "--metric", dest="metric", choices={"hard", "soft"}, default='soft', help="metric used for pairwise reward candidate generation")
    parser.add_argument("-g", "--gamma", dest="gamma", default=0.0, type=float, help="The discount factor used in reward learning")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--exp_idx", type=str, default="0")
    parser.add_argument("--reward_learning_samples", type=int, default=2, help="Number of trajectories for reward learning")
    parser.add_argument("--reward_loss", type=str, default="listNet", help="Should be 'listNet' or 'listMLE'")
    parser.add_argument("--listmle_temp", type=float, default=1.)
    parser.add_argument("--listnet_power", type=int, default=1)

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default='facebook/bart-base')
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=2, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--back_bone", type=str, default="bart", help="choose t5 or bart")
    parser.add_argument("--policy_training_seed", type=str, default="111", help="random seed for training the policy")

    args = parser.parse_args()
    assert args.data_folder != ""
    assert args.reward_learning_samples > 1
    assert args.reward_loss in ("listNet", "listMLE")

    cfg.data_folder = args.data_folder
    cfg.exp_idx = args.exp_idx
    cfg.reward_learning_samples = args.reward_learning_samples
    cfg.listmle_temp = args.listmle_temp
    cfg.listnet_power = args.listnet_power

    assert isinstance(cfg.listmle_temp, float) and cfg.listmle_temp > 0.
    assert isinstance(cfg.listnet_power, int) and cfg.listnet_power >= 0

    if not os.path.exists(f'./experiments/Exp{cfg.exp_idx}'):
        os.makedirs(f'./experiments/Exp{cfg.exp_idx}')

    cfg.mode = args.mode
    if args.mode == 'test':
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(args.model_path, 'exp_cfg.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_per_domain', 'use_true_pv_resp',
                     'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                     'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                     'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                     'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                     'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                     'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                     'act_selection_scheme', 'beam_penalty_type', 'record_mode']:
                continue
            setattr(cfg, k, v)
            cfg.result_path = os.path.join(args.model_path, 'result.csv')
    else:
        parse_arg_cfg(args)
        if args.model_path == "":
            args.model_path = 'experiments/Exp{}/{}_sd{}/reward_model/'.format(
                cfg.exp_idx, '-'.join(cfg.exp_domains), args.policy_training_seed,
            )
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        cfg.result_path = os.path.join(args.model_path, 'result.csv')
        cfg.eval_load_path = args.model_path

    cfg._init_logging_handler(args.mode)

    assert cfg.seed == args.seed
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    with open(os.path.join(args.model_path, 'exp_cfg.json'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    print('\nargs:', args, "\n", flush=True)
    rewardLearning = RewardLearning(args.folds, args.action_space, args.metric, args, reward_loss=args.reward_loss)
    rewardLearning.load_reward_rollouts()
    rewardLearning.train_model()
    rewardLearning.save_returns(args.gamma)





    







