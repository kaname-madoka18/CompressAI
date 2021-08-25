import os

qualities = [5e-3, 1e-3]
# qualities = [5e-3, 0.01, 0.05, 0.1]
for lam in qualities:
    os.system(f"python3 examples/myTrain.py -d /backup/home/zetian/DS --batch-size 16 -lr 1e-4 --lambda {lam} --save "
              f"--cuda -e 1")
    os.mkdir(f"lam{lam}")
    os.system(f"mv checkpoint* lam{lam}/")
