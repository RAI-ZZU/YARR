import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from multiprocessing import Lock
from typing import Optional, List
from typing import Union
import yaml

import numpy as np
import psutil
import torch
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator

NUM_WEIGHTS_TO_KEEP = 10


class PyTorchEvalRunner(TrainRunner):

    def __init__(self,
                 agent: Agent,
                 env_runner: EnvRunner,
                 wrapped_replay_buffer: Union[
                     PyTorchReplayBuffer, List[PyTorchReplayBuffer]],
                 train_device: torch.device,
                 replay_buffer_sample_rates: List[float] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,

                 iterations: int = int(1e6),
                 logdir: str = '/tmp/yarr/logs',
                 log_freq: int = 10,
                 transitions_before_train: int = 1000,
                 weightsdir: str = '/tmp/yarr/weights',
                 save_freq: int = 100,
                 replay_ratio: Optional[float] = None,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 buffers_per_batch: int = -1  # -1 = all
                 ):
        super(PyTorchEvalRunner, self).__init__(
            agent, env_runner, wrapped_replay_buffer,
            stat_accumulator,
            iterations, logdir, log_freq, transitions_before_train, weightsdir,
            save_freq)

        env_runner.log_freq = log_freq
        env_runner.target_replay_ratio = replay_ratio
        self._wrapped_buffer = wrapped_replay_buffer if isinstance(
            wrapped_replay_buffer, list) else [wrapped_replay_buffer]

        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging

        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir, tensorboard_logging, csv_logging)
        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)
        self._buffers_per_batch = buffers_per_batch if buffers_per_batch > 0 else len(wrapped_replay_buffer)


    def _signal_handler(self, sig, frame):
        if threading.current_thread().name != 'MainThread':
            return
        logging.info('SIGINT captured. Shutting down.'
                     'This may take a few seconds.')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]
        sys.exit(0)


    def generate_next_filename(self, logdir, base_name="eval_output"):

        yaml_files = [f for f in os.listdir(logdir) if f.endswith('.yaml')]
        

        existing_files = [f for f in yaml_files if f.startswith(base_name)]
        

        existing_numbers = []
        for f in existing_files:
            try:
                number = int(f[len(base_name) + 1:-5])  
                existing_numbers.append(number)
            except ValueError:
                continue  
        

        if existing_numbers:
            n = max(existing_numbers) + 1
        else:
            n = 1
        
        new_filename = f"{base_name}_{n}.yaml"
        return os.path.join(logdir, new_filename)

    def start(self):

        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock()

        # Kick off the environments

        self._env_runner.start(self._save_load_lock)

        while True:


            if self._env_runner.check():
                break
            #self._writer.end_iteration()
            time.sleep(1)

        env_summaries = self._env_runner.summaries()
        keys = env_summaries[0]
        values = env_summaries[1]
        print("*************")

        for key,value in zip(keys,values):
            if key != "valid_viewpoints":
                print(key,": ",value)
        print("*************")
        
        summaries_dict = dict(zip(keys, values))
        valid_viewpoints = summaries_dict.pop("valid_viewpoints", None)
        # valid_viewpoints_file = os.path.join(self._logdir, "valid_viewpoints.csv")
        # np.savetxt(valid_viewpoints_file, valid_viewpoints, delimiter=",")      
        
        # /data_nvme/wgk/train/open_drawer/ADAVM/active/seed0/eval/
        attributes = self._logdir.split("/")

        summaries_dict["task"] = attributes[-5]
        #
        summaries_dict["camera"] = attributes[-3]
        # logdir
        summaries_dict["log_dir"] = self._logdir

        summaries_dict["seed"] = int(attributes[-2][-1])

        summaries_dict["model"] = attributes[-4]

        new_yaml_file = self.generate_next_filename(self._logdir)
        

        with open(new_yaml_file, 'w') as yaml_file:
            yaml.dump(summaries_dict, yaml_file, default_flow_style=False, allow_unicode=True)
            
            
            
        # base_path  = "/home/ubuntu/code/ARM/tools/viewpoints/"
        # task_name =  attributes[-5]
        # learned_viewpoints_file = os.path.join(base_path,task_name,"all_learned_viewpoints.csv")
        # np.savetxt(learned_viewpoints_file, valid_viewpoints, delimiter=",")    


        if self._writer is not None:
            self._writer.close()

        logging.info('Stopping envs ...')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]





