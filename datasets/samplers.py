"""Official datasets/samplers.py behavior, including weighted depletion and fixed16 queries."""
from copy import deepcopy
import numpy as np
from torch.utils.data import Sampler


def class_indices(dataset):
    grouped = {}
    for image_index, (_, class_id) in enumerate(dataset.samples):
        grouped.setdefault(class_id, []).append(image_index)
    return grouped


class TrainingEpisodeSampler(Sampler):
    """Support-first episodes; each source image used at most once per epoch.

    Class selection probability proportional to remaining image count, not uniform.
    No fixed epoch length, no drop/replacement of episodes beyond official depletion.
    """
    def __init__(self, dataset, way, shots):
        self.way = way
        self.shots = tuple(shots)
        self.class_to_indices = class_indices(dataset)
        if way > len(self.class_to_indices) or way <= 0:
            raise ValueError("Episode way exceeds available classes")
        if any(len(indices) < sum(shots) for indices in self.class_to_indices.values()):
            raise ValueError("A class lacks enough support+query images; no automatic resampling")

    def __iter__(self):
        remaining = deepcopy(self.class_to_indices)
        for indices in remaining.values():
            np.random.shuffle(indices)
        while len(remaining) >= self.way:
            classes = list(remaining)
            counts = np.array([len(remaining[class_id]) for class_id in classes])
            chosen_classes = np.random.choice(classes, size=self.way, replace=False,
                                               p=counts/counts.sum())
            episode_indices = []
            for count in self.shots:
                for class_id in chosen_classes:
                    for _ in range(count):
                        episode_indices.append(remaining[class_id].pop())
            for class_id in chosen_classes:
                if len(remaining[class_id]) < sum(self.shots):
                    remaining.pop(class_id)
            yield episode_indices


class EvaluationEpisodeSampler(Sampler):
    """Original query argument is ignored; fixed16 is an explicit compatibility rule."""
    OFFICIAL_QUERY = 16

    def __init__(self, dataset, way, shot, episodes, query):
        self.class_to_indices = class_indices(dataset)
        self.way, self.shot, self.episodes = way, shot, episodes
        self.query = self.OFFICIAL_QUERY
        self.requested_query = query
        if way > len(self.class_to_indices) or way <= 0:
            raise ValueError("Episode way exceeds available classes")
        if any(len(indices) < shot+self.query for indices in self.class_to_indices.values()):
            raise ValueError("Evaluation class lacks enough disjoint support/query images")

    def __iter__(self):
        grouped = deepcopy(self.class_to_indices)
        classes = list(grouped)
        self.episode_indices = []
        for _ in range(self.episodes):
            np.random.shuffle(classes)
            selected = classes[:self.way]
            for class_id in selected:
                np.random.shuffle(grouped[class_id])
            support = [index for class_id in selected for index in grouped[class_id][:self.shot]]
            query = [index for class_id in selected
                     for index in grouped[class_id][self.shot:self.shot+self.query]]
            episode = support+query
            self.episode_indices.append(episode)
            yield episode

    def __len__(self):
        return self.episodes
