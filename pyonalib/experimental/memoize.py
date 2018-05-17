import os
import pickle
import inspect
import hashlib
import gzip
import functools


class Store(object):
    def __init__(self, path):
        self.path = os.path.join(path, '.memoize')
        os.makedirs(self.path, exist_ok=True)

    def _get_path_for_key(self, key):
        return os.path.join(self.path, f'{key}.pkl.gz')

    def put(self, key, value):
        with gzip.open(self._get_path_for_key(key), 'wb') as f:
            pickle.dump(value, f)

    def get(self, key):
        try:
            with gzip.open(self._get_path_for_key(key), 'rb') as f:
                return pickle.load(f)
        except:
            os.remove(self._get_path_for_key(key))
            raise "Error occured"

    def __contains__(self, key):
        return os.path.isfile(self._get_path_for_key(key))

class memoize():
    def __init__(self, func, store_dir=os.getcwd()):
        self.func = func
        self.store = Store(store_dir)
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        key = self._get_key(*args, **kwargs)
        if key not in self.store:
            val = self.func(*args, **kwargs)
            self.store.put(key, val)
        return self.store.get(key)

    def _arg_hash(self, *args, **kwargs):
        _str = pickle.dumps(args, 2) + pickle.dumps(kwargs, 2)
        return hashlib.md5(_str).hexdigest()

    def _src_hash(self):
        _src = inspect.getsource(self.func)
        return hashlib.md5(_src.encode()).hexdigest()

    def _get_key(self, *args, **kwargs):
        arg = self._arg_hash(*args, **kwargs)
        src = self._src_hash()
        return src + '_' + arg
