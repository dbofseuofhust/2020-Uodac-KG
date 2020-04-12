cd mmdetection/undersonar/
python src/submit.py json/r50_gc.bbox.json all
python src/submit.py json/r50_speckle.bbox.json all
python src/submit.py json/r50_db.bbox.json all
python src/submit.py json/r101.bbox.json all

python src/csv2pkljson.py

python src/ensemble.py

python src/submit_db.py --jsonpath json/testB_ensemble.pkl.bbox.json --testpath coco/annotations/instances_testB_all.json --savepath submit.csv 
