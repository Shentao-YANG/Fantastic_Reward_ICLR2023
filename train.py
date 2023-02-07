from __future__ import absolute_import
import torch.nn.functional as F
import copy
import sys
import os, random, argparse, time, logging, json, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import torch
from RewardTorch import RewardLearning, numpyBinaryTuple2torch
from transformers import (AdamW, T5Tokenizer, BartTokenizer, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)
from T5 import MiniT5
from BART import MiniBART
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), 'damd_multiwoz'))
from utils import Vocab, MultiWozReader
from damd_multiwoz.config import global_config as cfg
from damd_multiwoz.eval import MultiWozEvaluator


class BartTokenizer(BartTokenizer):
    def encode(self,text,add_special_tokens=False):
        encoded_inputs = self.encode_plus(text,add_special_tokens=False)
        return encoded_inputs["input_ids"]

RETURNS_PATH = './damd_multiwoz/data/multi-woz-oppe/'
DATASET_FILE_NAME = ''
ADOPT_CASPI = True
CASPI_WT = 1.0
VAL_FRACTION = 0.1

class Model(object):
    def __init__(self, args, test=False):
        print('[Model] RETURNS_PATH:', RETURNS_PATH, flush=True)
        print('[Model] ADOPT_CASPI:', ADOPT_CASPI, flush=True)
        print('[Model] CASPI_WT:', CASPI_WT, flush=True)
        print('[Model] VAL_FRACTION:', VAL_FRACTION, flush=True)
        self.neg_rew_weight = args.neg_rew_weight
        print('[Model] neg_rew_weight:', self.neg_rew_weight, flush=True)

        if args.back_bone=="t5":  
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = MiniT5.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        elif args.back_bone=="bart":
            self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = MiniBART.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        vocab = Vocab(self.model, self.tokenizer)
        self.reader = MultiWozReader(vocab,args)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        print('[Model] num_loorf_samples:', self.args.num_loorf_samples, flush=True)
        print('[Model] match_loss_val:', self.args.match_loss_val, flush=True)
        if len(cfg.cuda_device)==1:
            self.model.to(args.device)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.cuda_device)
            self.model = self.model.cuda()

        ###CASPI mod starts
        self.model.set_caspi_wt(float(CASPI_WT))
        self.fn_tn_return_dict = defaultdict(dict)
        self.reader.val_fraction = VAL_FRACTION
        with open(RETURNS_PATH, 'r') as f:
            return_json = json.load(f)
            for fn, info in return_json.items():
                for tn, G_info in info.items():
                    self.fn_tn_return_dict[fn][tn] = float(G_info['G'])
        ###CASPI mod end

        if self.neg_rew_weight > 0.:
            # load the reward learning object
            new_args = copy.deepcopy(args)
            new_args.model_path = os.path.join(args.model_path, "reward_model/")
            _, _, folds, gamma, action_space, metric = args.caspi_returns_file.split("_")
            # reward_loss don't matter since we do not train the reward model here
            self.reward_model = RewardLearning(int(folds), action_space, metric.split(".")[0], new_args, reward_loss="listMLE", test=True)
            self.reward_model_hidden_size = self.reward_model.model.model.shared.weight.shape[1]              # shared.weight (50333, 768)
            print(f"Reward model hidden size: {self.reward_model_hidden_size}", flush=True)             # should be 768
            if self.args.num_loorf_samples >= 1:
                self.nll_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def load_model(self):
        self.model = type(self.model).from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def get_neg_rew(self, states, actions, goals, lm_labels):
        states = self.reward_model.padInput(states, self.reward_model.MAX_STATE_LEN, use_dynamic_pad_len=True)          # states: (states_input_ids, states_masks)
        states = numpyBinaryTuple2torch(states)                                                                         # states[0], states[1]: torch.tensor(batch_size, max_state_len_in_batch)
        states_emb, states_mask = self.reward_model.model.get_embeddings(states[0]), states[1]
        # states_emb: torch.tensor(batch_size, max_state_len, 768);
        # states_mask: torch.tensor(batch_size, max_state_len)

        # no need to add '<eos_r>' to actions since lm_labels already have
        # actions: torch.tensor(batch_size, max_action_len, vocab_size)
        actions_emb = self.reward_model.model.get_embeddings(actions)
        # actions_emb: torch.tensor(batch_size, max_action_len, 768)
        actions_mask = (lm_labels > 1e-7).long()                  # torch.ones(actions_emb.shape[0], actions_emb.shape[1], dtype=torch.long).cuda()
        # actions_mask: torch.tensor(batch_size, max_action_len)

        goals = self.reward_model.padInput(goals, self.reward_model.MAX_GOAL_LEN, use_dynamic_pad_len=True)             # goals: (goal_input_ids, goal_mask)
        goals = numpyBinaryTuple2torch(goals)                                                                           # goals[0], goals[1]: torch.tensor(batch_size, max_goal_len_in_batch)
        goals_emb, goals_mask = self.reward_model.model.get_embeddings(goals[0]), goals[1]
        # goals_emb: torch.tensor(batch_size, max_goal_len, 768);
        # goals_mask: torch.tensor(batch_size, max_goal_len)

        assert len(states_emb.shape) == len(actions_emb.shape) == len(goals_emb.shape) == 3
        assert states_emb.shape[0] == actions_emb.shape[0] == goals_emb.shape[0]
        assert states_emb.shape[-1] == actions_emb.shape[-1] == goals_emb.shape[-1] == self.reward_model_hidden_size

        input_embs = torch.cat([states_emb, actions_emb, goals_emb], dim=-2)
        attention_mask = torch.cat([states_mask, actions_mask, goals_mask], dim=-1)

        assert input_embs.requires_grad

        rewards = self.reward_model.model.forward(
            input_ids=None,
            inputs_embeds=input_embs,
            attention_mask=attention_mask
        )[0]

        return (-1.) * rewards.mean()


    def get_neg_rew_loorf(self, states, action_logits, goals, lm_labels):
        # action_logits: (batch_size, max_action_len, vocab_size), prob vectors for each word on each batch
        num_samples = self.args.num_loorf_samples
        rew_each_sample = []
        nll_each_sample = []
        action_logits_flatten = action_logits.reshape(-1, action_logits.shape[-1])
        # action_logits_flatten (batch_size x max_action_len, vocab_size)
        action_probs_flatten = action_logits_flatten.detach()
        action_probs_flatten = (action_probs_flatten - action_probs_flatten.max(dim=-1, keepdim=True)[0]).exp()
        actions_mask = (lm_labels > 1e-7).long()               # torch.ones(action_logits.shape[0], action_logits.shape[1], dtype=torch.long).cuda()
        actions_mask_float = actions_mask.float()
        actions_mask_float_row_sum = actions_mask_float.sum(dim=-1, keepdim=True)
        # actions_mask, actions_mask_float: torch.tensor(batch_size, max_action_len)
        # actions_mask_float_row_sum: torch.tensor(batch_size, 1)

        states = self.reward_model.padInput(states, self.reward_model.MAX_STATE_LEN, use_dynamic_pad_len=True)          # states: (states_input_ids, states_masks)
        states_input_ids, states_masks = numpyBinaryTuple2torch(states)
        # states_input_ids, states_masks: torch.tensor(batch_size, max_state_len_in_batch)

        goals = self.reward_model.padInput(goals, self.reward_model.MAX_GOAL_LEN, use_dynamic_pad_len=True)  # goals: (goal_input_ids, goal_mask)
        goal_input_ids, goal_mask = numpyBinaryTuple2torch(goals)
        # goal_input_ids, goal_mask: torch.tensor(batch_size, max_goal_len_in_batch)

        for _ in range(num_samples):
            # batch_size: action_logits.shape[0]

            # sample action_ids from multinomial distribution (no gradient)
            action_sample_ids = torch.multinomial(action_probs_flatten, num_samples=1, replacement=True)    # torch.tensor(batch_size x max_action_len, 1)
            action_sample_ids = action_sample_ids.reshape(action_logits.shape[0], -1)                           # torch.tensor(batch_size, max_action_len)
            assert states_input_ids.shape[0] == action_sample_ids.shape[0] == goal_input_ids.shape[0] == action_logits.shape[0]
            assert action_sample_ids.shape[1] == action_logits.shape[1]

            # get the rewards
            input_ids = torch.cat([states_input_ids, action_sample_ids, goal_input_ids], dim=-1)
            attention_mask = torch.cat([states_masks, actions_mask, goal_mask], dim=-1)
            # input_ids, attention_mask: (batch_size, s_len + a_len + g_len)
            rewards = self.reward_model.model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach()        # (batch_size, 1)
            rew_each_sample.append(rewards)

            # get the NLLs
            NLL = self.nll_loss_fct(action_logits_flatten, action_sample_ids.view(-1)).reshape(action_logits.shape[0], -1)
            # NLL (batch_size, max_action_len)
            NLL = (NLL * actions_mask_float).sum(dim=-1, keepdim=True) / actions_mask_float_row_sum  # NLL (batch_size, 1)
            nll_each_sample.append(NLL)

        rew_each_sample = torch.cat(rew_each_sample, dim=-1)
        nll_each_sample = torch.cat(nll_each_sample, dim=-1)
        # rew_each_sample, nll_each_sample: (batch_size, num_samples)
        assert rew_each_sample.shape == nll_each_sample.shape == (action_logits.shape[0], num_samples)

        # de-mean rew across the samples
        rew_each_sample = rew_each_sample - rew_each_sample.mean(dim=-1, keepdim=True)
        loss = (rew_each_sample * nll_each_sample).mean() * (num_samples / (num_samples - 1))
        assert loss.requires_grad

        return loss

    def train(self):
        btm = time.time()
        step = 0
        prev_min_loss = 1000
        print(f"vocab_size:{self.model.config.vocab_size}")
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        self.tokenizer.save_pretrained(self.args.model_path)
        self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()
        # lr scheduler
        lr_lambda = lambda epoch: self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        for epoch in range(cfg.epoch_num):
            log_loss = 0
            log_dst = 0
            log_resp = 0
            log_cnt = 0
            log_neg_rew = 0.
            sw = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {'pv_bspn': None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                    for k in inputs:
                        if k!="turn_domain":
                            inputs[k] = inputs[k].to(self.args.device)

                    outputs = self.model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["state_input"],
                                        lm_labels=inputs["state_update"]
                                        )
                    dst_loss = outputs[0]

                    returns = []
                    if ADOPT_CASPI==True:
                        for fn,tn in zip(turn_batch['dial_id'],turn_batch['turn_num']):
                            tn = str(tn)
                            if fn in self.fn_tn_return_dict and tn in self.fn_tn_return_dict[fn]:
                                returns.append(self.fn_tn_return_dict[fn][tn])
                            else:
                                raise Exception('fn:{} has no return!?'.format(fn))

                    outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["response_input"],
                                        lm_labels=inputs["response"],
                                        returns = returns,
                                        adopt_caspi = ADOPT_CASPI,
                                        )
                    resp_loss = outputs[0]

                    py_prev['bspn'] = turn_batch['bspn']

                    if self.neg_rew_weight > 0.:
                        # states: turn_batch['bsdx'], list of encoded tokens (unpadded)
                        # goals: list of encoded tokens (unpadded)
                        goals = []
                        for fn in turn_batch['dial_id']:
                            if fn in self.reward_model.data_for_damd:
                                goals.append(self.reward_model.encode_state(self.reward_model.goal_as_st(self.reward_model.data_for_damd[fn]['goal'])))
                            else:
                                raise ValueError('fn:{} has no goal!?'.format(fn))

                        if self.args.num_loorf_samples < 2:
                            # use gumbel-softmax
                            # actions: (batch_size, max_action_len, vocab_size), one-hot vectors for each word on each batch
                            actions = F.gumbel_softmax(outputs[1], tau=1, hard=True)
                            neg_rew_loss = self.get_neg_rew(states=turn_batch['bsdx'], actions=actions,
                                                            goals=goals, lm_labels=inputs["response"])
                        else:
                            # use loorf estimator
                            neg_rew_loss = self.get_neg_rew_loorf(states=turn_batch['bsdx'], action_logits=outputs[1],
                                                                  goals=goals, lm_labels=inputs["response"])
                        dst_resp_loss = dst_loss + resp_loss
                        if self.args.match_loss_val:
                            # match the scale of dst_resp_loss to that of the neg_rew_loss
                            dst_resp_loss = dst_resp_loss / (dst_resp_loss.detach().abs() + 1e-10) * neg_rew_loss.detach().abs()
                        else:
                            # weight neg_rew_loss by self.neg_rew_weight
                            neg_rew_loss = neg_rew_loss * self.neg_rew_weight
                        total_loss = (dst_resp_loss + neg_rew_loss) / self.args.gradient_accumulation_steps
                    else:
                        total_loss = (dst_loss + resp_loss) / self.args.gradient_accumulation_steps

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step+=1
                    log_loss += float(total_loss.item())
                    log_dst +=float(dst_loss.item())
                    log_resp +=float(resp_loss.item())
                    if self.neg_rew_weight > 0.:
                        log_neg_rew += float(neg_rew_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|resp|neg_rew] loss: {:.2f} {:.2f} {:.2f} {:.4f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                           log_loss/(log_cnt+ 1e-8),
                                                                           log_dst/(log_cnt+ 1e-8),log_resp/(log_cnt+ 1e-8),
                                                                           log_neg_rew/(log_cnt+ 1e-8),
                                                                           time.time()-btm,
                                                                           turn_num+1))
            epoch_sup_loss = log_loss/(log_cnt+ 1e-8)
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))

            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, WEIGHTS_NAME))
                logging.info('Model saved')
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                scheduler.step()
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))

                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    self.eval()
                    return

        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        self.eval()

    def validate(self, data='dev', do_test=False):
        self.model.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, _ = self.reader.wrap_result(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
        self.model.train()
        if do_test:
            print('result preview...')
            self.eval()
        return valid_loss

    def eval(self, data='test'):
        self.model.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs
             
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, field = self.reader.wrap_result(result_collection)

        self.reader.save_result('w', results, field)

        metric_results = self.evaluator.run_metrics(results, eval_act=False)
        metric_field = list(metric_results[0].keys())
        req_slots_acc = metric_results[0]['req_slots_acc']
        info_slots_acc = metric_results[0]['info_slots_acc']

        self.reader.save_result('w', metric_results, metric_field,
                                            write_title='EVALUATION RESULTS:')
        self.reader.save_result('a', [info_slots_acc], list(info_slots_acc.keys()),
                                            write_title='INFORM ACCURACY OF EACH SLOTS:')
        self.reader.save_result('a', [req_slots_acc], list(req_slots_acc.keys()),
                                            write_title='REQUEST SUCCESS RESULTS:')
        self.reader.save_result('a', results, field+['wrong_domain', 'wrong_act', 'wrong_inform'],
                                            write_title='DECODED RESULTS:')
        self.reader.save_result_report(metric_results)

        self.model.train()
        return None

    def lexicalize(self, result_path,output_path):
        self.reader.relex(result_path,output_path)


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="t5-small", help="t5-small, t5-base, bart-large")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--caspi_returns_file", default='', type=str)
    parser.add_argument("--caspi", action='store_true', default=False)
    parser.add_argument("--caspi_wt", default=1.0, type=float)
    parser.add_argument("--caspi_val_fraction", default=0.1, type=float)
    parser.add_argument("--caspi_data_file", default='data_for_damd.json', type=str)
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--exp_idx", type=str, default="0")
    parser.add_argument("--neg_rew_weight", default=0.0, type=float)
    parser.add_argument("--num_loorf_samples", default=0, type=int)
    parser.add_argument("--match_loss_val", default=0, type=int, help='default (0): do not match loss values.')
    args = parser.parse_args()
    assert args.data_folder != ""
    assert args.num_loorf_samples >= 0
    assert isinstance(args.match_loss_val, int) and args.match_loss_val in (0, 1)
    args.match_loss_val = args.match_loss_val == 1

    cfg.data_folder = args.data_folder
    cfg.exp_idx = args.exp_idx

    if not os.path.exists(f'./experiments/Exp{cfg.exp_idx}'):
        os.makedirs(f'./experiments/Exp{cfg.exp_idx}')

    #CASPI param starts
    global RETURNS_PATH,ADOPT_CASPI,CASPI_WT,CASPI_DATA_FILE, VAL_FRACTION
    RETURNS_PATH = RETURNS_PATH.replace('data', cfg.data_folder)
    if '/' in args.caspi_returns_file:
        RETURNS_PATH = args.caspi_returns_file
    else:
        RETURNS_PATH = os.path.join(RETURNS_PATH, args.caspi_returns_file)
    CASPI_DATA_FILE = args.caspi_data_file
    ADOPT_CASPI = args.caspi
    CASPI_WT = args.caspi_wt
    caspi_uid = args.caspi_returns_file
    VAL_FRACTION = args.caspi_val_fraction
    #CASPI param ends

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'relex':
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
        print(args)
        if args.model_path=="":
            args.model_path = 'experiments/Exp{}/{}_sd{}/'.format(cfg.exp_idx, '-'.join(cfg.exp_domains), cfg.seed)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        cfg.result_path = os.path.join(args.model_path, 'result.csv')
        cfg.eval_load_path = args.model_path

    cfg._init_logging_handler(args.mode)
    
    if ADOPT_CASPI==True:
        cfg.data_file=CASPI_DATA_FILE

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.mode == 'train':
        with open(os.path.join(args.model_path, 'exp_cfg.json'), 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)
        m = Model(args)
        m.train()
    elif args.mode == 'test':
        m = Model(args,test=True)
        m.eval(data='test')
    elif args.mode == 'relex':
        m = Model(args,test=True)
        output_path = os.path.join(args.model_path, 'generation.csv')
        m.lexicalize(cfg.result_path,output_path)


if __name__ == '__main__':
    main()
