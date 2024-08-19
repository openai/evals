Additional requirements (in addition to the base reqs of this repo) for generating this dataset are in `requirements.txt`.

To generate datasets, run in order:
```bash
python dataset.py # Generates dataset in CSV format
python csv2jsonl.py # Converts CSV dataset to JSONL as expected by evals framework
```

## Troubleshooting
* For some versions of Python (tested with Python 3.10.12), you may encounter the error described [here](https://github.com/huggingface/datasets/issues/5613#issuecomment-1703169594) when running `python dataset.py`. If so, you can fix it by additionally running `pip install multiprocess==0.70.15` _after_ installing `requirements.txt`.