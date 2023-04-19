import json

ndjson_path = "gutenberg-poetry-v001.ndjson"

textfile_path = "poetry.txt"

with open(ndjson_path, "r") as f:
    for line in f:
        data = json.loads(line)
        data.encode().decode()
        with open(textfile_path, "a") as textfile:
            textfile.write(data["s"] + "\n")