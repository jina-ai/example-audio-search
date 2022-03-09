from jina import Flow
from docarray import DocumentArray
from pathlib import path
import sys

def check_query(resp):
    for d in resp.docs:
        print(f'{d.uri}, {len(d.chunks)}')
        for m in d.matches:
            print(f'+- {m.uri}: {m.scores["cosine"].value:.6f}, {m.tags}')

def main():
    workspace = Path(__file__).parent.absolute() / 'workspace'
    docs = DocumentArray.from_files('toy-data/*.mp3')
    if workspace.exists():
            print(
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
        f.cors = True
        f.block()


if __name__ == '__main__':
    main()