# Search Similar Audios

Searching for similar audios has a wide range of application including finding similar songs, replacing curse words and detecting the speakers. In this tutorial, we will build an example of searching similar audios using the [AudioSet](https://research.google.com/audioset/) dataset and the [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model.

## Usage

Download the model
```yaml
bash scripts/download_models.sh
```

Start the flow and index the data at `./toy-data`
```python
python app.py
```

With the Flow running as a http service, we can use the Jina swagger UI tool to query. 
Open the browser at `localhost:45678/docs`, send query via the Swagger UI,

```json
{
  "data": [
    {
      "uri": "toy-data/6pO06krKrf8_30000_airplane.mp3"
    }
  ]
}
```