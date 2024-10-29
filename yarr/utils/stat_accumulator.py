from multiprocessing import Lock
from typing import List
import copy

import numpy as np
from yarr.agents.agent import Summary, ScalarSummary
from yarr.utils.transition import ReplayTransition


class StatAccumulator(object):

    def step(self, transition: ReplayTransition, eval: bool):
        pass

    def pop(self) -> List[Summary]:
        pass

    def peak(self) -> List[Summary]:
        pass

    def reset(self) -> None:
        pass


class Metric(object):

    def __init__(self):
        self._previous = []
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        self._previous.clear()

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]


class _SimpleAccumulator(StatAccumulator):

    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()

        self._episode_returns = Metric()

        self._episode_lengths = Metric()
        self._summaries = []
        self._transitions = 0

    def _reset_data(self):
        with self._lock:

            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1

            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:

                self._episode_returns.next()
                self._episode_lengths.next()
            self._summaries.extend(list(transition.summaries))

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        names = ["return", "length"]
        metrics = [self._episode_returns, self._episode_lengths]
        for name, metric in zip(names, metrics):
            for stat_key in stat_keys:
                if self._mean_only:
                    assert stat_key == "mean"

                    sum_name = '%s/%s' % (self._prefix, name)
                else:
                    sum_name = '%s/%s/%s' % (self._prefix, name, stat_key)
                sums.append(
                    ScalarSummary(sum_name, getattr(metric, stat_key)()))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions))
        sums.extend(self._summaries)
        return sums

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 1:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()
    
    def reset(self):
        self._transitions = 0
        self._reset_data()


class SimpleAccumulator(StatAccumulator):

    def __init__(self, eval_video_fps: int = 30, mean_only: bool = True):
        self._train_acc = _SimpleAccumulator(
            'train_envs', eval_video_fps, mean_only=mean_only)
        self._eval_acc = _SimpleAccumulator(
            'eval_envs', eval_video_fps, mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition, eval)
        else:
            self._train_acc.step(transition, eval)

    def pop(self) -> List[Summary]:
        return self._train_acc.pop() + self._eval_acc.pop()

    def peak(self) -> List[Summary]:
        return self._train_acc.peak() + self._eval_acc.peak()
    
    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()







class MultiTaskAccumulator(StatAccumulator):

    def __init__(self, num_tasks,
                 eval_video_fps: int = 30, mean_only: bool = True,
                 train_prefix: str = 'train_task',
                 eval_prefix: str = 'eval_task'):
        self._train_accs = [_SimpleAccumulator(
            '%s%d/envs' % (train_prefix, i), eval_video_fps, mean_only=mean_only)
            for i in range(num_tasks)]
        self._eval_accs = [_SimpleAccumulator(
            '%s%d/envs' % (eval_prefix, i), eval_video_fps, mean_only=mean_only)
            for i in range(num_tasks)]
        self._train_accs_mean = _SimpleAccumulator(
            '%s_summary/envs' % train_prefix, eval_video_fps,
            mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        replay_index = transition.info["active_task_id"]
        if eval:
            self._eval_accs[replay_index].step(transition, eval)
        else:
            self._train_accs[replay_index].step(transition, eval)
            self._train_accs_mean.step(transition, eval)

    def pop(self) -> List[Summary]:
        combined = self._train_accs_mean.pop()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.pop())
        return combined

    def peak(self) -> List[Summary]:
        combined = self._train_accs_mean.peak()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.peak())
        return combined

    def reset(self) -> None:
        self._train_accs_mean.reset()
        [acc.reset() for acc in self._train_accs + self._eval_accs]




class _EvalAccumulator(StatAccumulator):

    def __init__(self, prefix,
                 episode_num,
                 worker_episode_num,
                 episode_length):
        self._prefix = prefix
        self._lock = Lock()

        self._summaries = []
        self._transitions = 0
        self._episode_num = episode_num

        self._worker_episode_num = worker_episode_num
        self._episode_length = episode_length

        self._success_flags = np.zeros([episode_num]) # [episode_num,]   
        self._execution_failed_flags = np.zeros([episode_num]) # [episode_num,]
        
        self._episode_entropys = np.zeros([episode_num,episode_length])  # [episode_num, length]
        self._episode_init_information_gain = np.zeros([episode_num,episode_length])  # [episode_num, length]
        self._episode_information_gain = np.zeros([episode_num,episode_length])  # [episode_num, length]
        self._episode_action_interactable = np.zeros([episode_num,episode_length])  # [episode_num, length] 

        self._episode_transitions = np.zeros([episode_num],dtype=np.int8)  # [episode_num]
        self._episode_ROI_reachable = np.zeros([episode_num,episode_length]) # [episode_num, length]
        
        self._episode_viewpoints = np.zeros([episode_num,episode_length,3]) # [episode_num, length, 3] 


    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            env_id = int(transition.info["env_name"][-1])
            episode_id = env_id*self._worker_episode_num + transition.info["episode_id"]
            self._episode_entropys[episode_id][self._episode_transitions[episode_id]] = transition.info["roi_entropy"] 
            if "roi_reachable" in transition.info.keys():
                self._episode_ROI_reachable[episode_id][self._episode_transitions[episode_id]] = transition.info["roi_reachable"]
            if "goal_position_interable" in transition.info.keys():
                self._episode_action_interactable[episode_id][self._episode_transitions[episode_id]] = transition.info["goal_position_interable"]
            if "information_gain" in transition.info.keys():
                self._episode_information_gain[episode_id][self._episode_transitions[episode_id]] = transition.info["information_gain"]
            if "init_information_gain" in transition.info.keys():
                self._episode_init_information_gain[episode_id][self._episode_transitions[episode_id]] = transition.info["init_information_gain"]
            if "viewpoint_tp0_1" in transition.info.keys():
                self._episode_viewpoints[episode_id][self._episode_transitions[episode_id]] = transition.info["viewpoint_tp0_1"]
                

            if transition.terminal:
                if transition.reward >= 100:
                    self._success_flags[episode_id] = 1

                elif self._episode_transitions[episode_id] < self._episode_length-1:
                    self._execution_failed_flags[episode_id] = 1
            self._episode_transitions[episode_id] += 1
            self._transitions += 1



    def _get(self) -> List[Summary]:
        sums = {}
        

        #execution_complate_episode_transitions = self._episode_transitions[np.where(self._execution_failed_flags==0)[0]]

        sums["episode_num"] = self._episode_num

        sums["overtime_rate"] = float(np.sum(self._episode_transitions==10)/self._episode_num)

        legal_episode_transitions = copy.deepcopy(self._episode_transitions)
        legal_episode_transitions[np.where(self._execution_failed_flags==1)[0]] = 10
        
        sums["episode_length_mean"] = float(np.round(legal_episode_transitions.mean(),4))
        sums["episode_length_std"] = float(np.round(legal_episode_transitions.std(),4))

        sums["success_rate"] = float(np.round(self._success_flags.mean(),4))

        sums["execution_failed_rate"] = float(np.round(self._execution_failed_flags.mean(),4))

        sums["execution_failed_rate1"] = float(np.round(self._execution_failed_flags.sum()/(self._episode_num-self._success_flags.sum()),4))

        max_step = self._episode_length 
        length_mask = np.arange(max_step) < self._episode_transitions[:, None]
        interaction_mask = self._episode_action_interactable.astype(bool) 
        mask = length_mask&interaction_mask 
        valid_entropys = np.where(mask, self._episode_entropys, np.nan)
        valid_information_gain =  np.where(length_mask, self._episode_information_gain, np.nan)
        valid_init_information_gain =  np.where(length_mask, self._episode_init_information_gain, np.nan)
        
        flat_valid_init_information_gain = valid_init_information_gain[~np.isnan(valid_init_information_gain)]
        flat_valid_information_gain = valid_information_gain[~np.isnan(valid_information_gain)]
        flat_valid_entropys = valid_entropys[~np.isnan(valid_entropys)]
        sums["roi_entropy_mean"] = float(np.round(flat_valid_entropys.mean(),4))
        sums["roi_entropy_std"] = float(np.round(flat_valid_entropys.std(),4))
        
        # ASSIG
        sums["ASSIG_mean"] = float(np.round(flat_valid_information_gain.mean(),4))
        sums["ASSIG_std"] = float(np.round(flat_valid_information_gain.std(),4))
        # AIIG
        sums["AIIG_mean"] = float(np.round(flat_valid_init_information_gain.mean(),4))
        sums["AIIG_std"] = float(np.round(flat_valid_init_information_gain.std(),4))

        first_interaction_steps =  np.where(np.any(mask == 1, axis=1), np.argmax(mask == 1, axis=1), self._episode_length-1) + 1
        sums["first_insteraction"] = float(np.round(first_interaction_steps.mean(), 4))
        

        episode_transition_num = copy.deepcopy(self._episode_transitions)
        episode_transition_num[np.where(self._execution_failed_flags==1)] = 10 
        interaction_step = mask.sum() 
        sums["non_insteraction"] = float((episode_transition_num.sum()-interaction_step)/self._episode_num)  
        

        #successful_episodes = self._episode_viewpoints[self._success_flags == 1]
        #valid_viewpoints = successful_episodes.reshape(-1, 3)
        valid_viewpoints = self._episode_viewpoints.reshape(-1,3)
        valid_viewpoints = valid_viewpoints[np.any(valid_viewpoints != (0, 0, 0), axis=1)]
        sums["valid_viewpoints"] = valid_viewpoints
        

        return [list(sums.keys()),list(sums.values())]

    def pop(self) -> List[Summary]:
        #data = []
        #if len(self._episode_returns) > 1:
        data = self._get()
        return data





class EvalAccumulator(StatAccumulator):

    def __init__(self,episode_num,
                 worker_episode_num,
                 episode_length):
        self._eval_acc = _EvalAccumulator(
            'eval_envs',episode_num,worker_episode_num,episode_length )

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition, eval)

    def pop(self) -> List:
        return self._eval_acc.pop()

