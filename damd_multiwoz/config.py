import logging, time, os


class _Config:
    def __init__(self):
        self._data_folder = "data"
        self._exp_idx = 999999
        self._reward_learning_samples = 2
        self._listmle_temp = 1.
        self._listnet_power = 1
        self._multiwoz_damd_init()

    @property
    def data_folder(self):
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value):
        self._data_folder = value
        self.vocab_path_train = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/vocab')
        self.data_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/')
        self.dev_list = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz/valListFile.json')
        self.test_list = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz/testListFile.json')
        self.glove_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/glove/glove.6B.50d.txt')
        self.domain_file_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/domain_files.json')
        self.multi_acts_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/multi_act_mapping_train.json')

        if "TEST" in self.data_folder:
            self.report_interval = 50
            self.vocab_size = 514
            self.embed_size = 5
            self.hidden_size = 10
            self.epoch_num = 2
            self.early_stop_count = 1

    @property
    def exp_idx(self):
        return self._exp_idx

    @exp_idx.setter
    def exp_idx(self, value):
        self._exp_idx = value
        self.eval_load_path = f'experiments/Exp{self.exp_idx}/all_multi_acts_sample3_sd777_lr0.005_bs80_sp5_dc3'

    @property
    def reward_learning_samples(self):
        return self._reward_learning_samples

    @reward_learning_samples.setter
    def reward_learning_samples(self, value):
        self._reward_learning_samples = value
        print(f"\n SET: reward_learning_samples = {self.reward_learning_samples} \n", flush=True)

    @property
    def listmle_temp(self):
        return self._listmle_temp

    @listmle_temp.setter
    def listmle_temp(self, value):
        self._listmle_temp = value
        print(f"\n SET: listmle_temp = {self.listmle_temp} \n", flush=True)

    @property
    def listnet_power(self):
        return self._listnet_power

    @listnet_power.setter
    def listnet_power(self, value):
        self._listnet_power = value
        print(f"\n SET: listnet_power = {self.listnet_power} \n", flush=True)

    def _multiwoz_damd_init(self):
        self.test_mode = False

        self.vocab_path_train = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/vocab')
        self.vocab_path_eval = None
        self.data_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/')
        self.data_file = 'data_for_damd.json'
        self.dev_list = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz/valListFile.json')
        self.test_list = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz/testListFile.json')
        self.dbs = {
            'attraction': os.path.join(os.path.dirname(__file__),'db/attraction_db_processed.json'),
            'hospital': os.path.join(os.path.dirname(__file__),'db/hospital_db_processed.json'),
            'hotel': os.path.join(os.path.dirname(__file__),'db/hotel_db_processed.json'),
            'police': os.path.join(os.path.dirname(__file__),'db/police_db_processed.json'),
            'restaurant': os.path.join(os.path.dirname(__file__), 'db/restaurant_db_processed.json'),
            'taxi': os.path.join(os.path.dirname(__file__),'db/taxi_db_processed.json'),
            'train': os.path.join(os.path.dirname(__file__),'db/train_db_processed.json'),
        }
        self.glove_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/glove/glove.6B.50d.txt')
        self.domain_file_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/domain_files.json')
        self.slot_value_set_path = os.path.join(os.path.dirname(__file__),'db/value_set_processed.json')
        self.multi_acts_path = os.path.join(os.path.dirname(__file__), f'{self.data_folder}/multi-woz-processed/multi_act_mapping_train.json')
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # experiment settings
        self.mode = 'unknown'
        self.cuda = True
        self.cuda_device = [1]
        self.exp_no = ''
        self.seed = 11
        self.exp_domains = ['all']
        self.save_log = True
        self.report_interval = 100
        self.max_nl_length = 60
        self.max_span_length = 40
        self.truncated = False

        # model settings
        self.vocab_size = 3000
        self.embed_size = 50
        self.hidden_size = 100
        self.pointer_dim = 6 # fixed
        self.enc_layer_num = 1
        self.dec_layer_num = 1
        self.dropout = 0
        self.layer_norm = False
        self.skip_connect = False
        self.encoder_share = False
        self.attn_param_share = False
        self.copy_param_share = False
        self.enable_aspn = False
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = True

        # training settings
        self.lr = 0.005
        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.batch_size = 128
        self.epoch_num = 100
        self.early_stop_count = 6
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.multi_acts_training = False
        self.multi_act_sampling_num = 1
        self.valid_loss = 'score'

        # evaluation settings
        self.eval_load_path = f'experiments/Exp{self.exp_idx}/all_multi_acts_sample3_sd777_lr0.005_bs80_sp5_dc3'
        self.eval_per_domain = False
        self.use_true_pv_resp = True
        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_prev_dspn = False
        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = False
        self.use_true_db_pointer = False
        self.limit_bspn_vocab = False
        self.limit_aspn_vocab = False
        self.same_eval_as_cambridge = True
        self.same_eval_act_f1_as_hdsa = False
        self.aspn_decode_mode = 'greedy'  #beam, greedy, nucleur_sampling, topk_sampling
        self.beam_width = 5
        self.nbest = 5
        self.beam_diverse_param=0.2
        self.act_selection_scheme = 'high_test_act_f1'
        self.topk_num = 1
        self.nucleur_p = 0.
        self.record_mode = False

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists(f"./log/Exp{self.exp_idx}"):
            os.makedirs(f"./log/Exp{self.exp_idx}")
        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler('./log/Exp{}/log_{}_{}_{}_{}_sd{}.txt'.format(self.exp_idx, self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)


global_config = _Config()

