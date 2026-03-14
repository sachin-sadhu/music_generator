from collections import defaultdict
from Note import TrainingNote
import numpy as np

class RhythmMC:
    def __init__(self, training_notes: list[TrainingNote]) -> None:
        self.mc = self.calc_mc(training_notes)

    def calc_mc(self, training_notes):
        transition_count = defaultdict(lambda: defaultdict(int))   

        for i in range(len(training_notes)-1):
            curr_note_duration = training_notes[i].duration
            next_note_duration = training_notes[i+1].duration

            transition_count[curr_note_duration][next_note_duration] += 1

        transition_probs = defaultdict(lambda: defaultdict(float))
        for curr_duration in transition_count.keys():
            total_count = sum(transition_count[curr_duration].values())
            for next_duration, count in transition_count[curr_duration].items():
                transition_probs[curr_duration][next_duration] = count / total_count

        return transition_probs

    def sample_next_duration(self, curr_duration):
        default_duration = 0.5
        if curr_duration not in self.mc:
            return default_duration

        candidate_durations = list(self.mc[curr_duration].keys())
        candidate_durations_probs = list(self.mc[curr_duration].values())

        return candidate_durations[np.random.choice(len(candidate_durations), p=candidate_durations_probs)]