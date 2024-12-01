python main_sacl.py --method isogclr --batch_size 128 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
python main_sacl.py --method sogclr --temp 0.1 --batch_size 128 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
python main_sacl.py --method simclr --temp 0.3 --batch_size 128 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
