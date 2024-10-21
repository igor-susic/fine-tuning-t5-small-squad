# Distributed training of T5

If you want to finetune T5 on multiple GPU devices run the `distributed.py` script.

````shell
python distributed.py 20 5 --batch_size 32
````

Where parameters are:
1. First parameter is number of epochs
2. Second is each N epochs we will checkpoint the model weigths
3. `--batch_size` - self descriptive


Results from approx 10 epochs:

````shell
python eval_script.py dev-v2.0.json finetuned-results-distributed-training.json
````

````json
{
  "exact": 67.64086583003453,
  "f1": 71.62779540992244,
  "total": 11873,
  "HasAns_exact": 65.7557354925776,
  "HasAns_f1": 73.74102815485946,
  "HasAns_total": 5928,
  "NoAns_exact": 69.52060555088309,
  "NoAns_f1": 69.52060555088309,
  "NoAns_total": 5945
}
````