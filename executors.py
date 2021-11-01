import librosa as lr
import numpy as np
from collections import defaultdict
from typing import Dict, Iterable, Optional

from audioread.exceptions import NoBackendError

from jina import Document, DocumentArray, Executor, requests


_ALLOWED_METRICS = ['min', 'max']


class TimeSegmenter(Executor):
    def __init__(self, chunk_duration: float = 1, chunk_stride: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_duration = chunk_duration  # seconds
        self.stride = chunk_stride

    @requests(on=['/index', '/search'])
    def segment(
            self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        if not docs:
            return
        for idx, doc in enumerate(docs):
            try:
                doc.blob, sample_rate = lr.load(doc.uri, sr=16000)
            except NoBackendError as e:
                print(f'failed to load {doc.uri}, {e}')
                continue
            doc.tags['sample_rate'] = sample_rate
            chunk_size = int(self.chunk_duration * sample_rate)
            strip = parameters.get('chunk_strip', self.stride)
            strip_size = int(strip * sample_rate)
            num_chunks = max(1, int((doc.blob.shape[0] - chunk_size) / strip_size))
            for chunk_id in range(num_chunks):
                beg = chunk_id * strip_size
                end = beg + chunk_size
                if beg > doc.blob.shape[0]:
                    break
                c = Document(
                    blob=doc.blob[beg:end],
                    offset=idx,
                    location=[beg, end],
                    tags=doc.tags,
                    uri=doc.uri
                )
                c.tags['beg_in_ms'] = beg / sample_rate * 1000
                c.tags['end_in_ms'] = end / sample_rate * 1000
                doc.chunks.append(c)


class SimpleRanker(Executor):
    """
    SimpleRanker aggregates the score of matches of chunks, where these matches are just
    chunks of some larger document as well, to the score of the parent document of the
    matches. The Document's matches are then replaced by matches based on the parent
    documents of the matches of chunks - they contain an `id` and the aggregated score only.
    This ranker is used to "bubble-up" the scores of matches of chunks to the scores
    of the matches' parent document.
    """

    def __init__(
        self,
        metric: str = 'cosine',
        ranking: str = 'min',
        traversal_paths: Iterable[str] = ('r',),
        *args,
        **kwargs,
    ):
        """
        :param metric: the distance metric used in `scores`
        :param ranking: The sort and aggregation function that the executor uses.
            The allowed options are:
            - `min`: Set the (parent) match's score to the minimum score of its chunks,
                sort matches in an ascending order.
            - `max`: Set the (parent) match's score to the maximum score of its chunks,
                sort matches in a descending order.
        :param traversal_paths: The traversal paths, used to obtain the documents we
            want the ranker to work on - these are the "query" documents, for which
            we wish to create aggregated matches.
        """
        super().__init__(*args, **kwargs)

        if ranking not in _ALLOWED_METRICS:
            raise ValueError(
                f'ranking should be one of {_ALLOWED_METRICS}, got "{ranking}"',
            )

        self.metric = metric
        self.ranking = ranking
        self.traversal_paths = traversal_paths

    @requests(on='/search')
    def rank(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Aggregate the score of matches of chunks to the score of their parent
        document.
        The matches of query documents that are passed in `docs` are replaced (if they
        exist) by documents based on the aggregated score of the parent documents of
        matches of chunks - that is, by documents containing only parent id of matches
        of chunks, and the aggregated score corresponding to that id.
        :param docs: The documents for which to create aggregated matches (specifically,
            the aggregated matches will be created for documents that are on the
            traversal paths of documents passed in this argument).
        :param parameters: Extra parameters that can be used to override the parameters
            set at creation of the Executor. Valid values are `traversal_paths`
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        for doc in docs.traverse_flat(traversal_paths):
            parents_scores = defaultdict(list)
            parents_match = defaultdict(list)
            for m in DocumentArray([doc]).traverse_flat(['cm']):
                parents_scores[m.parent_id].append(m.scores[self.metric].value)
                parents_match[m.parent_id].append(m)

            # Aggregate match scores for parent document and
            # create doc's match based on parent document of matched chunks
            new_matches = []
            for match_parent_id, scores in parents_scores.items():
                if self.ranking == 'min':
                    score_id = np.argmin(scores)
                elif self.ranking == 'max':
                    score_id = np.argmax(scores)
                score = scores[score_id]
                match = parents_match[match_parent_id][score_id]
                uri = match.uri
                new_match = Document(uri=uri, id=match_parent_id, scores={self.metric: score})
                new_match.tags['beg_in_ms'] = match.tags['beg_in_ms']
                new_match.tags['end_in_ms'] = match.tags['end_in_ms']
                new_matches.append(new_match)

            # Sort the matches
            doc.matches = new_matches
            if self.ranking == 'min':
                doc.matches.sort(key=lambda d: d.scores[self.metric].value)
            elif self.ranking == 'max':
                doc.matches.sort(key=lambda d: -d.scores[self.metric].value)