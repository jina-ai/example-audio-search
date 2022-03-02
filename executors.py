import librosa as lr
import numpy as np
from collections import defaultdict

from jina import Executor, requests
from docarray import Document, DocumentArray

class AudioSegmenter(Executor):
    def __init__(self, window_size: float = 1, stride: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size  # seconds
        self.stride = stride

    @requests(on=['/index', '/search'])
    def segment(self, docs: DocumentArray, **kwargs):
        for idx, doc in enumerate(docs):
            try:
                doc.tensor, sample_rate = lr.load(doc.uri, sr=16000)
            except RuntimeError as e:
                print(f'failed to load {doc.uri}, {e}')
                continue
            doc.tags['sample_rate'] = sample_rate
            chunk_size = int(self.window_size * sample_rate)
            stride_size = int(self.stride * sample_rate)
            num_chunks = max(1, int((doc.tensor.shape[0] - chunk_size) / stride_size))
            for chunk_id in range(num_chunks):
                beg = chunk_id * stride_size
                end = beg + chunk_size
                if beg > doc.tensor.shape[0]:
                    break
                c = Document(
                    tensor=doc.tensor[beg:end],
                    offset=idx,
                    location=[beg, end],
                    tags=doc.tags,
                    uri=doc.uri
                )
                c.tags['beg_in_ms'] = beg / sample_rate * 1000
                c.tags['end_in_ms'] = end / sample_rate * 1000
                doc.chunks.append(c)


class MyRanker(Executor):
    @requests(on='/search')
    def rank(self, docs: DocumentArray = None, **kwargs):
        for doc in docs['@r']:
            parents_scores = defaultdict(list)
            parents_match = defaultdict(list)
            for m in DocumentArray([doc])['@c,m']:
                parents_scores[m.parent_id].append(m.scores['cosine'].value)
                parents_match[m.parent_id].append(m)
            # Aggregate match scores for parent document and
            # create doc's match based on parent document of matched chunks
            new_matches = []
            for match_parent_id, scores in parents_scores.items():
                score_id = np.argmin(scores)
                score = scores[score_id]
                match = parents_match[match_parent_id][score_id]
                new_match = Document(
                    uri=match.uri,
                    id=match_parent_id,
                    scores={'cosine': score})
                new_match.tags['beg_in_ms'] = match.tags['beg_in_ms']
                new_match.tags['end_in_ms'] = match.tags['end_in_ms']
                new_matches.append(new_match)
            # Sort the matches
            new_matches.sort(key=lambda d: d.scores['cosine'].value)
            doc.matches = new_matches