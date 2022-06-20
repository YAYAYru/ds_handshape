## Tutorial
- MLOps и production подход к ML исследованиям [ods.ai](https://ods.ai/tracks/ml-in-production-spring-22)|[gitlab](https://gitlab.com/m1f/mlops-course/-/blob/main/src/models/train.py)
- Data version control:
    - Course:Iterative Tools for Data Scientists & Analysts [dvc.org](https://learn.iterative.ai/path-player?courseid=data-scientist-path&unit=61d379a63ce70b73600f7913Unit)
    - course-ds-base from the course [github](https://github.com/iterative/course-ds-base)
    - DVC Get Started [github](https://github.com/iterative/example-get-started)
    - Machine Learning Experiments with DVC (Hands-On Tutorial!) [youtub](https://youtu.be/iduHPtBncBk)|Maybe [github](https://github.com/elleobrien/2_dvc)
    - kaggle-titanic-dvc [dagshub](https://dagshub.com/kingabzpro/kaggle-titanic-dvc)

- Cпособ веток для экспериментов [Udemy](https://www.udemy.com/course/draft/3194020/learn/lecture/23313562#overview)
- Сравнение результатов экспериментов [Udemy](https://www.udemy.com/course/draft/3194020/learn/lecture/23313576#overview)

## Пользование
### Run
- Список файлов в указанной папке в raw записан в csv c двумя метками "righthand" и "lefthand":
```bash
python3 src/data/videofolder2filelistcsv.py 
```
### Запуск преобразования видео на скелет `video2skelet.py`
Пример [структуры датасет видеофайлов](data/raw/sl_hand) для запуска преобразования видео на скелетную модель в формате json:
```bash
python3 src/stages/video2skelet.py params.yaml 
```
------------
    data
     └─raw
     │  └─datasetname
     │    ├─anyname1_signername1
     │    │  ├anyname1.mp4
     │    │  ├anyname2.mp4
     │    │  └ ...
     │    ├─anyname1_signername2
     │    │  ├anyname1.mp4
     │    │  ├anyname2.mp4
     │    │  └ ...
     │    └ ...
     └─interim
        └─datasetname
          ├─anyname1_signername1
          │  ├anyname1.mp4
          │  ├anyname2.mp4
          │  └ ...
          ├─anyname1_signername2
          │  ├anyname1.mp4
          │  ├anyname2.mp4
          │  └ ...
          └ ...
В папке data/raw/datasetname такая структура должна быть, а в папке data/interim/datasetname автоматически генеруются файлы и папки. Должен быть параметры в `params.yaml`:
```YAML
video2skelet:
  deps:
    path_video_folders: data/raw/datasetname
  outs:
    path_json_xyz_folders: data/interim/datasetname
```

### Запуск объединения скелет и метки в csv `video2skelet.py`
Пример [структуры датасет видеофайлов](data/raw/sl_hand_label) для Запуск объединения скелет и метки в csv:
------------
    data
     └─raw
     │  └─datasetname_label
     │    ├─anyname1_signername1.csv
     │    ├─anyname1_signername2.csv
     │    └ ...
     └─interim
        └─datasetname_label.csv
В папке data/raw/datasetname_label такая структура должна быть, а в папке data/interim/datasetname_label.csv автоматически генеруются строки. Еще нужна важная последняя фраза(пример signername1) на имени файла signername1, чтобы автоматически будет вставить signername1 в строку Должен быть параметры в `params.yaml`:
```YAML
json2csv:
  deps:
    path_folder_csv: data/raw/sl_hand_label
  outs:
    path_xyz_63f_csv: data/interim/sl_hand_label.csv
```