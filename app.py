from jina import DocumentArray, Flow
from jina.types.document.generators import from_files
import os
import sys
import logging

def check_query(resp):
    for d in resp.docs:
        print(f'{d.uri}, {len(d.chunks)}')
        for m in d.matches:
            print(f'+- {m.uri}: {m.scores["cosine"].value:.6f}, {m.tags}')

def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.join(cur_dir, 'workspace')
    logger = logging.getLogger('audio-search')
    docs = DocumentArray(from_files('toy-data/*.mp3'))
    if os.path.exists(workspace):
            logger.error(
                f'\n +------------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                           | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again.  | \
                    \n |                                   🤖🤖🤖                                           | \
                    \n +------------------------------------------------------------------------------------+'
            )
            sys.exit(1)
    f = Flow.load_config('flow.yml')
    with f:
        f.post(on='/index', inputs=docs, show_progress=True)
        f.post(on='/search', inputs=docs, on_done=check_query)
        f.protocol = 'http'
        f.cors = True
        f.block()


if __name__ == '__main__':
    main()