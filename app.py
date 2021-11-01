import glob
import os
import click

from jina import Document, Flow


def check_index(resp):
    for d in resp.docs:
        print(f'{d.uri}: {len(d.chunks)}')
        for c in d.chunks:
            print(f'+- {c.embedding.shape}')


def check_query(resp):
    for d in resp.docs:
        print(f'{d.uri}, {len(d.chunks)}')
        for m in d.matches:
            print(f'+- {m.id}: {m.scores["cosine"].value:.4f}, {m.tags}')


def get_index_doc():
    for fn in glob.glob('toy-data/*.mp3'):
        yield Document(id=fn, uri=fn)


def index():
    f = Flow.load_config('flow.yml')
    with f:
        f.post(on='/index', inputs=get_index_doc, on_done=check_index)


def query():
    f = Flow.load_config('flow.yml')
    with f:
        f.post(on='/search', inputs=get_index_doc, on_done=check_query)


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