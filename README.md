# Author Profiling: Bot and Gender Prediction using a Multi-Aspect Ensemble Approach
This project contains our paper's codes in python, used to indetifying bots and humans and in case of human used to identifying gender in [PAN 2019](https://pan.webis.de/clef19/pan19-web/author-profiling.html) competition.


# Usage
please follow the instructions to profiling authors.

### 1. running `prepare_dataset.py`
Parameters:
```
usage: prepare_dataset.py [-h] [-i INPUT] [-o OUTPUT]

optional arguments:
  -i INPUT      path to input dataset
  -o OUTPUT     path to output directory(default = 'prepared_dataset')
```
running the script usage:
```
python prepare_dataset.py -i path_to_dataset_root_dir
```
### 2. running `training_ngram.py`
Parameters:
```
usage: training_ngram.py [-h] [-i INPUT] [-o OUTPUT] [-ft FT] [-n N]

optional arguments:
  -i INPUT            path to prepared dataset
  -o OUTPUT           path to output directory(default='pre-trained_models')
  -ft FT              frequency threshold (default=5)
  -n N                n-gram order (default=4)
```
running the script usage:
```
python training_ngram.py -i path_to_prepared_dataset
```
### 3. running `training_tfidf.py`
Parameters:
```
usage: training_tfidf.py [-h] [-i INPUT] [-o OUTPUT] [-ft FT]

optional arguments:
  -i INPUT      path to prepared dataset
  -o OUTPUT     path to output directory(default='pre-trained_models')
  -ft FT        frequency threshold (default=5)
```
running the script usage:
```
python training_tfidf.py -i path_to_prepared_dataset
```
### 4. running `training_doc2vec.py` 
Parameters:
```
usage: training_doc2vec.py [-h] [-i INPUT] [-o OUTPUT]

optional arguments:
  -i INPUT      path to prepared dataset
  -o OUTPUT     path to output directory(default='pre-trained_models')

```
running the script usage:
```
python training_doc2vec.py -i path_to_prepared_dataset
```
### 5. running `bot_gender_profiling.py`
Parameters:
```
usage: bot_gender_profiling.py [-h] [-i INPUT] [-o OUTPUT] [-t TRAIN_DIR] [-m MODELS] [-n N]

optional arguments:
  -i INPUT        path to dataset directory
  -o OUTPUT       path to output directory
  -t TRAIN_DIR    path to train dataset directory
  -m MODELS       path to models directory
  -n N            n-gram order (default=4)

```
running the script usage:
```
python bot_gender_profiling.py -i path_to_dataset_dir -o paht_to_output_dir -t path_to_train_dataset_dir -m paht_to_modles_dir
```

# Citation
Please cite us as:

*HB Giglou, M Rahgouy, T Rahgouy, MK Sheykhlan, E Mohammadzadeh. Author Profiling: Bot and Gender Prediction using a Multi-Aspect Ensemble Approach - Notebook for PAN at CLEF 2019. In CLEF 2019 Evaluation Labs and Workshop–Working Notes Papers. CEUR-WS. org.*
