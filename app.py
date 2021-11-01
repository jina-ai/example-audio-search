from jina import DocumentArray, Flow
from jina.types.document.generators import from_files


def main():
    docs = DocumentArray(from_files('toy-data/*.mp3'))

    f = Flow.load_config('flow.yml')
    with f:
        f.post(on='/index', inputs=docs)
        f.protocol = 'http'
        f.cors = True
        f.block()


if __name__ == '__main__':
    main()