import lmdb
env = lmdb.open("/media/zj/新加卷/ubuntu/projects/aster.pytorch/data/true_data/test/")
txn = env.begin(write=True)
for key, value in txn.cursor():
    if key[:5]==b'label':
        print (key, value.decode('gbk'))