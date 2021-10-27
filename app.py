from typing import Optional, Tuple
import os
import librosa as lr
import numpy as np
import click

from jina import Document, DocumentArray, Executor, Flow, requests


class TimeSegmenter(Executor):
    def __init__(self, chunk_duration: float = 10, chunk_strip: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_duration = chunk_duration  # seconds
        self.strip = chunk_strip

    @requests(on='/index')
    def segment(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        if not docs:
            return
        for idx, doc in enumerate(docs):
            doc.blob, sample_rate = self._load_raw_audio(doc)
            doc.tags['sample_rate'] = sample_rate
            chunk_size = int(self.chunk_duration * sample_rate)
            strip = parameters.get('chunk_strip', self.strip)
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
                    )
                c.tags['beg_in_ms'] = beg / sample_rate * 1000
                c.tags['end_in_ms'] = end / sample_rate * 1000
                doc.chunks.append(c)
            print(f'process {doc.id}, {len(doc.chunks)}')

    def _load_raw_audio(self, doc: Document) -> Tuple[np.ndarray, int]:
        if doc.blob is not None and doc.tags.get('sample_rate', None) is None:
            raise NotImplementedError('data is blob but sample rate is not provided')
        elif doc.blob is not None:
            return doc.blob, int(doc.tags['sample_rate'])
        elif doc.uri is not None and doc.uri.endswith('.mp3'):
            return self._read_mp3(doc.uri)
        else:
            raise NotImplementedError('doc needs to have either a blob or a wav/mp3 uri')

    def _read_mp3(self, file_path: str) -> Tuple[np.ndarray, int]:
        return lr.load(file_path)


def check_index(resp):
    for d in resp.docs:
        print(f'{d.id}: {len(d.chunks)}')


def check_query(resp):
    for d in resp.docs:
        for m in d.matches:
            print(f'{m.id[:10]}: {m.scores["cosine"].value:.4f},'
                  f' begin: {m.tags["beg_in_ms"]}, end: {m.tags["end_in_ms"]}')

def index():
    f = (Flow()
         .add(name='segmenter',
              uses=TimeSegmenter,
              uses_with={'chunk_duration': 0.5, 'chunk_strip': 0.1})  # split into chunks of 0.5s with 0.4s overlaps
         .add(name='encoder',
              uses='jinahub+docker://AudioCLIPEncoder/v0.4',
              uses_with={'traversal_paths': ['c', ]},
              volumes=f'{str(model_dir)}:/workdir/assets')
         .add(name='indexer', uses='jinahub://SimpleIndexer/v0.7')
         )

    with f:
        f.post(on='/index', inputs=Document(uri='toy-data/zvXkQkqd2I8_30.mp3'), on_done=check_index)


def query():
    f = (Flow()
         .add(name='loader', uses='jinahub://AudioLoader/v0.1', uses_with={'audio_types': ['mp3',]})
         .add(name='encoder',
              uses='jinahub+docker://AudioCLIPEncoder/v0.4',
              volumes=f'{str(model_dir)}:/workdir/assets')
         .add(name='indexer',
              uses='jinahub://SimpleIndexer/v0.7',
              uses_with={'match_args': {'limit': 10, 'traversal_rdarray': ['c', ]}})
         )

    with f:
        f.post(on='/search',
               inputs=Document(uri='toy-data/query.mp3'),
               parameters={'traversal_paths': ['r', ]},
               on_done=check_query)


@click.command()
@click.option('--task', '-t')
def main(task):
    if task == 'index':
        index()
    elif task == 'query':
        query()
    else:
        return


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(cur_dir, "models")
    main()