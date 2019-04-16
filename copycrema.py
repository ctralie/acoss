import deepdish as dd
import glob

for i, f in enumerate(glob.glob("features_benchmark/*.h5")):
    print(i, f)
    f1 = dd.io.load(f)
    f2 = dd.io.load(f.replace("features_benchmark", "crema_benchmark"))
    f1['crema'] = f2['crema']
    dd.io.save(f, f1)
