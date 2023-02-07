import os, random, argparse, time, logging, json, tqdm
import numpy as np

import torch
from torch.optim import Adam

import utils
from config import global_config as cfg
from reader import MultiWozReader
from damd_net import DAMD, cuda_, get_one_hot_input
from eval import MultiWozEvaluator
from otherconfig import *
import shutil
import base64
import hashlib
import re
import pandas as pd
import nvidia_smi


class Model(object):
    def __init__(self):
        self.reader = MultiWozReader()
        if len(cfg.cuda_device)==1:
            self.m =DAMD(self.reader)
        else:
            m = DAMD(self.reader)
            self.m=torch.nn.DataParallel(m, device_ids=cfg.cuda_device)
            # print(self.m.module)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        if cfg.cuda: self.m = self.m.cuda()  #cfg.cuda_device[0]
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

        if cfg.limit_bspn_vocab:
            self.reader.bspn_masks_tensor = {}
            for key, values in self.reader.bspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.bspn_masks_tensor[key] = v_
        if cfg.limit_aspn_vocab:
            self.reader.aspn_masks_tensor = {}
            for key, values in self.reader.aspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.aspn_masks_tensor[key] = v_
                
        self.epoch=0
        if other_config['gen_per_epoch_report']==True:
            self.df = pd.DataFrame(columns=['dial_id','success','match','bleu','rollout'])

    def add_torch_input(self, inputs, mode='train', first_turn=False):
        need_onehot = ['user', 'usdx', 'bspn', 'aspn', 'pv_resp', 'pv_bspn', 'pv_aspn',
                                   'dspn', 'pv_dspn', 'bsdx', 'pv_bsdx']
        inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())
        for item in ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
            if not cfg.enable_aspn and item == 'aspn':
                continue
            if not cfg.enable_bspn and item == 'bspn':
                continue
            if not cfg.enable_dspn and item == 'dspn':
                continue
            inputs[item] = cuda_(torch.from_numpy(inputs[item+'_unk_np']).long())
            if item in ['user', 'usdx', 'resp', 'bspn']:
                inputs[item+'_nounk'] = cuda_(torch.from_numpy(inputs[item+'_np']).long())
            else:
                inputs[item+'_nounk'] = inputs[item]
            # print(item, inputs[item].size())
            if item in ['resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
                if 'pv_'+item+'_unk_np' not in inputs:
                    continue
                inputs['pv_'+item] = cuda_(torch.from_numpy(inputs['pv_'+item+'_unk_np']).long())
                if item in ['user', 'usdx', 'bspn']:
                    inputs['pv_'+item+'_nounk'] = cuda_(torch.from_numpy(inputs['pv_'+item+'_np']).long())
                    inputs[item+'_4loss'] = self.index_for_loss(item, inputs)
                else:
                    inputs['pv_'+item+'_nounk'] = inputs['pv_'+item]
                    inputs[item+'_4loss'] = inputs[item]
                if 'pv_' + item in need_onehot:
                    inputs['pv_' + item + '_onehot'] = get_one_hot_input(inputs['pv_'+item+'_unk_np'])
            if item in need_onehot:
                inputs[item+'_onehot'] = get_one_hot_input(inputs[item+'_unk_np'])

        if cfg.multi_acts_training and 'aspn_aug_unk_np' in inputs:
            inputs['aspn_aug'] = cuda_(torch.from_numpy(inputs['aspn_aug_unk_np']).long())
            inputs['aspn_aug_4loss'] = inputs['aspn_aug']

        if 'G_unk_np' in inputs:
            inputs['G'] = cuda_(torch.from_numpy(inputs['G_unk_np']))
        if 'Q_unk_np' in inputs:
            inputs['Q'] = cuda_(torch.from_numpy(inputs['Q_unk_np']))
        if 'bhProb_unk_np' in inputs:
            inputs['bhProb'] = cuda_(torch.from_numpy(inputs['bhProb_unk_np']))
        return inputs

    def index_for_loss(self, item, inputs):
        raw_labels = inputs[item+'_np']
        if item == 'bspn':
            copy_sources = [inputs['user_np'], inputs['pv_resp_np'], inputs['pv_bspn_np']]
        elif item == 'bsdx':
            copy_sources = [inputs['usdx_np'], inputs['pv_resp_np'], inputs['pv_bsdx_np']]
        elif item == 'aspn':
            copy_sources = []
            if cfg.use_pvaspn:
                copy_sources.append(inputs['pv_aspn_np'])
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
        elif item == 'dspn':
            copy_sources = [inputs['pv_dspn_np']]
        elif item == 'resp':
            copy_sources = [inputs['usdx_np']]
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
            if cfg.enable_aspn:
                copy_sources.append(inputs['aspn_np'])
        else:
            return
        new_labels = np.copy(raw_labels)
        if copy_sources:
            bidx, tidx = np.where(raw_labels>=self.reader.vocab_size)
            copy_sources = np.concatenate(copy_sources, axis=1)
            for b in bidx:
                for t in tidx:
                    oov_idx = raw_labels[b, t]
                    if len(np.where(copy_sources[b, :] == oov_idx)[0])==0:
                        new_labels[b, t] = 2
        return cuda_(torch.from_numpy(new_labels).long())

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        weight_decay_count = cfg.weight_decay_count
        train_time = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            optim = self.optim
            # data_iterator generatation size: (batch num, turn num, batch size)
            btm = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                hidden_states = {}
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
                bgt = time.time()
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                    inputs = self.add_torch_input(inputs, first_turn=first_turn)
                    total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']

                    total_loss = total_loss.mean()
                    total_loss.backward(retain_graph=False)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()
                    sup_loss += float(total_loss)
                    sup_cnt += 1
                    torch.cuda.empty_cache()

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|aspn|resp] loss: {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                           float(total_loss),
                                                                           float(losses[cfg.bspn_mode]),float(losses['aspn']),float(losses['resp']),
                                                                           grad,
                                                                           time.time()-btm,
                                                                           turn_num+1))
                    if cfg.enable_dst and cfg.bspn_mode == 'bsdx':
                        logging.info('bspn-dst:{:.3f}'.format(float(losses['bspn'])))
                    if cfg.multi_acts_training:
                        logging.info('aspn-aug:{:.3f}'.format(float(losses['aspn_aug'])))

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))
            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                self.save_model(epoch)
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))
                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    # logging.info(str(cfg))
                    self.eval()
                    return
                if not weight_decay_count:
                    lr *= cfg.lr_decay
                    self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                  weight_decay=5e-5)
                    weight_decay_count = cfg.weight_decay_count
                    logging.info('learning rate decay, learning rate: %f' % (lr))
        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        # logging.info(str(cfg))
        self.eval()

    def validate(self, data='dev', do_test=False):
        self.m.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            hidden_states = {}
            py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
                    total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']

                    if cfg.valid_loss == 'total_loss':
                        valid_loss += float(total_loss)
                    elif cfg.valid_loss == 'bspn_loss':
                        valid_loss += float(losses[cfg.bspn_mode])
                    elif cfg.valid_loss == 'aspn_loss':
                        valid_loss += float(losses['aspn'])
                    elif cfg.valid_loss == 'resp_loss':
                        valid_loss += float(losses['reps'])
                    else:
                        raise ValueError('Invalid validation loss type!')
                else:
                    decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                    turn_batch['resp_gen'] = decoded['resp']
                    if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                        turn_batch['bspn_gen'] = decoded['bspn']
                    if cfg.enable_aspn:
                        turn_batch['aspn_gen'] = decoded['aspn']
                    py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                        py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                count += 1
                torch.cuda.empty_cache()

            if cfg.valid_loss in ['score', 'match', 'success', 'bleu']:
                result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
            valid_loss /= (count + 1e-8)
        else:
            results, _ = self.reader.wrap_result(result_collection)
            rollouts = {}
            for row in results:
                if row['dial_id'] not in rollouts:
                    rollouts[row['dial_id']]={}
                rollout = rollouts[row['dial_id']]
                rollout_step = {}
                rollout[row['turn_num']] = rollout_step
                
                rollout_step['resp'] = row['resp']
                rollout_step['resp_gen'] = row['resp_gen']
                rollout_step['aspn'] = row['aspn']
                
                if 'bspn' in row:
                    rollout_step['bspn'] = row['bspn']
                if 'bspn_gen' in row:
                    rollout_step['bspn_gen'] = row['bspn_gen']
                
                rollout_step['aspn_gen'] = row['aspn_gen']
                    
            if other_config['gen_per_epoch_report']==True:
                bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true, all_successes, all_matches, all_bleus ,dial_ids = self.evaluator.validation_metric(results, return_rich=True, return_per_dialog=True,soft_acc=other_config['soft_acc'])
                for i,dial_id in enumerate(dial_ids):
                    self.df.loc[len(self.df)] = [dial_id,all_successes[i],all_matches[i],all_bleus[i],json.dumps(rollouts[dial_id])]
                self.df.to_csv(other_config['per_epoch_report_path'])
            else:
                bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true = self.evaluator.validation_metric(results, return_rich=True)
            
            score = 0.5 * (success + match) + bleu
            valid_loss = 130 - score
            logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
        self.m.train()
        if do_test:
            print('result preview...')
            self.eval()
        return valid_loss

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            hidden_states = {}
            py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx':None}
            print('batch_size:', len(dial_batch[0]['resp']))
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                turn_batch['resp_gen'] = decoded['resp']
                if cfg.bspn_mode == 'bsdx':
                    turn_batch['bsdx_gen'] = decoded['bsdx'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                    turn_batch['bspn_gen'] = decoded['bspn'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                turn_batch['aspn_gen'] = decoded['aspn'] if cfg.enable_aspn else [[0]] * len(decoded['resp'])
                turn_batch['dspn_gen'] = decoded['dspn'] if cfg.enable_dspn else [[0]] * len(decoded['resp'])

                if self.reader.multi_acts_record is not None:
                    turn_batch['multi_act_gen'] = self.reader.multi_acts_record
                if cfg.record_mode:
                    turn_batch['multi_act'] = self.reader.aspn_collect
                    turn_batch['multi_resp'] = self.reader.resp_collect

                py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                if cfg.enable_bspn:
                    py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                    py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                if cfg.enable_aspn:
                    py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                if cfg.enable_dspn:
                    py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                torch.cuda.empty_cache()
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        if cfg.record_mode:
            self.reader.record_utterance(result_collection)
            quit()
        
        results, field = self.reader.wrap_result(result_collection)
        self.reader.save_result('w', results, field)

        metric_results = self.evaluator.run_metrics(results)
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
        self.m.train()
        return None

    def save_model(self, epoch, path=None, critical=False):
        if not cfg.save_log:
            return
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)
        logging.info('Model saved')

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)
        logging.info('Model loaded')

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        if not cfg.multi_gpu:
            initial_arr = self.m.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.embedding.weight.data.copy_(emb)
        else:
            initial_arr = self.m.module.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.module.embedding.weight.data.copy_(emb)


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt


def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            if k in other_config.keys():
                if isinstance(other_config[k],bool):
                    other_config[k] = eval(v)
                elif isinstance(other_config[k],int):
                    other_config[k] = int(v)
                elif isinstance(other_config[k],float):
                    other_config[k] = float(v)
                elif isinstance(other_config[k],str):
                    other_config[k] = str(v)
                elif other_config[k] is None:
                    other_config[k] = str(v)
                else:
                    raise Exception('Unkown config type:{}'.format(k))
                continue
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    if 'auto' in v:
                        v = [int(get_freer_device())]
                    else:
                        v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def get_freer_device():
    nvidia_smi.nvmlInit()
    gpu_free = []
    for gpu_id in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_free.append(mem.free)
    return np.argmax(gpu_free)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--exp_idx", type=str, default="0")
    args = parser.parse_args()
    assert args.data_folder != ""

    cfg.mode = args.mode
    cfg.data_folder = args.data_folder
    cfg.exp_idx = args.exp_idx

    if not os.path.exists(f'./experiments/Exp{cfg.exp_idx}'):
        os.makedirs(f'./experiments/Exp{cfg.exp_idx}')

    if args.mode == 'test' or args.mode=='adjust':
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(cfg.eval_load_path, 'config.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_load_path', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode' , 'data_file', 'use_true_db_pointer']:
                continue
            setattr(cfg, k, v)
            cfg.model_path = os.path.join(cfg.eval_load_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.eval_load_path, 'result.csv')
            
        other_config_path = os.path.join(cfg.eval_load_path, 'other_config.json')
        if os.path.isfile(other_config_path):
            other_cfg_load = json.loads(open(other_config_path, 'r').read())
            for k,v in other_cfg_load.items():
                other_config[k]=v
    else:
        parse_arg_cfg(args)
        
        print(other_config)
        hasher = hashlib.sha1(str(other_config).encode('utf-8'))
        hash_ = base64.urlsafe_b64encode(hasher.digest()[:10])
        hash_ = re.sub(r'[^A-Za-z]', '', str(hash_))
        
        if cfg.exp_path in ['' , 'to be generated']:
            cfg.exp_path = 'experiments/Exp{}/{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}_act{}_0.05_hash{}/'.format(cfg.exp_idx, '-'.join(cfg.exp_domains),
                                                                                            cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, cfg.weight_decay_count, cfg.enable_aspn,
                                                                                            hash_)
            if cfg.save_log:
                if os.path.exists(cfg.exp_path):
                    shutil.rmtree(cfg.exp_path)
                os.makedirs(cfg.exp_path)
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device)==1:
            cfg.multi_gpu = False
            torch.cuda.set_device(cfg.cuda_device[0])
        else:
            cfg.multi_gpu = True
            torch.cuda.set_device(cfg.cuda_device[0])
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    m = Model()
    cfg.model_parameters = m.count_params()

    if args.mode == 'train':
        if cfg.save_log:
            m.reader.vocab.save_vocab(cfg.vocab_path_eval)
            with open(os.path.join(cfg.exp_path, 'config.json'), 'w') as f:
                json.dump({**cfg.__dict__, **other_config}, f, indent=2)
                
            with open(os.path.join(cfg.exp_path, 'other_config.json'), 'w') as f:
                json.dump(other_config,f,indent=2)
        m.train()
    elif args.mode == 'adjust':
        m.load_model(cfg.model_path)
        m.train()
    elif args.mode == 'test':
        m.load_model(cfg.model_path)
        m.eval(data='test')


if __name__ == '__main__':
    main()
