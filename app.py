from jina import DocumentArray, Flow
from jina.types.document.generators import from_files

def check_query(resp):
    for d in resp.docs:
        print(f'{d.uri}, {len(d.chunks)}')
        for m in d.matches:
            print(f'+- {m.uri}: {m.scores["cosine"].value:.6f}, {m.tags}')

def main():
    docs = DocumentArray(from_files('toy-data/*.mp3'))

    f = Flow.load_config('flow.yml')
    with f:
        f.post(on='/index', inputs=docs)
        f.post(on='/search', inputs=docs, on_done=check_query)
        f.protocol = 'http'
        f.cors = True
        f.block()


if __name__ == '__main__':
    main()