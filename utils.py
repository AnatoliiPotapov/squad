# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from os import path
from keras.utils.data_utils import get_file

#def custom_objects():
#    from layers import *
#    from model import *
#    return locals()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def CoreNLP_path():
    SERVER = 'http://nlp.stanford.edu/software/'
    VERSION = 'stanford-corenlp-full-2017-06-09'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    lib_dir = path.join(path.abspath(path.dirname(__file__)), 'lib')

    get_file('/tmp/stanford-corenlp.zip',
             origin=origin,
             cache_dir=lib_dir,
             cache_subdir='',
             extract=True)

    return path.join(lib_dir, VERSION)
