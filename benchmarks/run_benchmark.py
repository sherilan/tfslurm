'''
Module for benchmarking performance improvements
'''
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import argparse
import pickle 
import time
import glob 
import sys 
import os 

# Fix sys-path so that we can import tf-slurm
sys.path.append( '..' )
from tfslurm import TFSlurm 

parser = argparse.ArgumentParser( 'Benchmark TFSlurm on a slowly training VGG16 model' )
parser.add_argument( '-j', '--jobdir',
  help = 'Directory where job output should be stored',
)
parser.add_argument( '--batch-size',
  help     = 'Batch size to train with (not distributed over tasks)',
  type     = int,
  required = True
)
parser.add_argument( '--num-ps',
  help     = 'Number of parameter servers to use. Must be set if not running locally',
  type     = int
)
parser.add_argument( '--local',
  help     = 'Whether to run locally (not distributed)',
  action   = 'store_true'
)
args = parser.parse_args()
if args.local:
  tfslurm = TFSlurm.create_local( dict( logdir=args.jobdir, shlevel='INFO' ))
  tfslurm.logger.info( 'Created local TFSlurm')
else:
  assert args.num_ps is not None, 'Number of parameter servers must be set when running distributed'
  tfslurm = TFSlurm.from_slurm_env( 
    num_ps=args.num_ps, 
    base_port=2222, 
    logger_opts=dict( logdir=args.jobdir, shlevel='INFO' )  
  )
  tfslurm.logger.info( 'Created distributed TFSlurm' )


tfslurm.logger.info( f'cluster    : { tfslurm.cluster }' )
tfslurm.logger.info( f'job name   : { tfslurm.my_job_name }' )
tfslurm.logger.info( f'job index  : { tfslurm.my_job_index }' )
tfslurm.logger.info( f'gpu frac   : { tfslurm.gpu_frac }' )
tfslurm.logger.info( f'num_ps     : { tfslurm.num_ps }' )
tfslurm.logger.info( f'num_workers: { tfslurm.num_workers }' )
tfslurm.logger.info( f'is ps      : { tfslurm.is_ps }' )
tfslurm.logger.info( f'is worker  : { tfslurm.is_worker }' )
tfslurm.logger.info( f'is chief   : { tfslurm.is_chief }' )
tfslurm.logger.info( f'dev setter : { tfslurm.device_setter }' )
tfslurm.logger.info( f'worker dev : { tfslurm.worker_device }' )
tfslurm.logger.info( f'ps devs    : { tfslurm.ps_devices }' )
tfslurm.logger.info( f'server     : { tfslurm.server }' )
sys.exit(0)

# Cat vs dogs dataset is assumed
PATH    = 'data/train/*.jpg'
CACHE_X = 'data/cacheX.npy'
CACHE_Y = 'data/cacheY.npy'
WIDTH  = 224
HEIGHT = 224
MAX_D  = 5000
DOG    = [ 0, 1 ]
CAT    = [ 1, 0 ]
EPOCHS     = 10
BATCH_SIZE = args.batch_size

if CACHE_X is None or not os.path.exists( CACHE_X ):

  tfslurm.logger.info( f'Proprocessing data from: {PATH}' )
  load_img = tf.keras.preprocessing.image.load_img
  preprocess_input = tf.keras.applications.vgg16.preprocess_input 
  filepaths = glob.glob( PATH )
  tfslurm.logger.info( f'Found {len(filepaths)} images' )

  images = []
  labels = []

  n_dogs = 0
  n_cats = 0

  for filepath in filepaths:
    
    if 'dog' in os.path.basename( filepath ):
      if n_dogs >= MAX_D: continue
      n_dogs += 1
      labels.append( DOG )
    elif 'cat' in os.path.basename( filepath ):
      if n_cats >= MAX_D: continue
      n_cats += 1
      labels.append( CAT )
    else:
      raise Exception( f'Bad filepath: {filepath}' )

    image = load_img( filepath, target_size=(224,224) )
    image = np.array( image, dtype=np.float32 )
    image = preprocess_input( image )
    images.append( image )


  X = np.array( images, dtype=np.float32 )
  Y = np.array( labels, dtype=np.float32 )

  tfslurm.logger.info( f'Processed {n_dogs} dog images' )
  tfslurm.logger.info( f'Processed {n_cats} cat imagse' )

  if not CACHE_X is None:
    np.save( CACHE_X, X )
    tfslurm.logger.info( f'Cached dataset X: {CACHE_X}' )
    np.save( CACHE_Y, Y )
    tfslurm.logger.info( f'Cached dataset Y: {CACHE_Y}' )

else:
  X = np.load( CACHE_X )
  Y = np.load( CACHE_Y )
  tfslurm.logger.info( f'Read data from cache: {CACHE_X}' )


tfslurm.logger.info( f'Dataset X: shape={X.shape} dtype={X.dtype}' )
tfslurm.logger.info( f'Dataset Y: shape={Y.shape} dtype={Y.dtype}' )


tfslurm.logger.info( 'Creating resnet50 model' )
vgg16 = tf.keras.applications.vgg16.VGG16(
  include_top=False
)
model = tf.keras.models.Model(
  vgg16.input,
  tf.keras.layers.Activation( 'softmax' )(
    tf.keras.layers.GlobalAveragePooling2D()(
      tf.keras.layers.Conv2D( 2, (3,3), padding='same' )( vgg16.output )
    )
  )
)

def generator( X, Y, batch_size ):
  assert len( X ) == len( Y ), 'X and Y must be the same length' 
  n = len( X )
  i = n
  while True:
    if i >= n:
      indices = np.random.permutation( n )
      i = 0
    j = min( n, i+batch_size )
    batch = indices[i:j]
    yield X[batch], Y[batch]
    i = j

class MonitoringCallback( tf.keras.callbacks.Callback ):
  
  def __init__( self ):
    self.prev_t = None 
    self.times = []
    self.losses = []
    self.accuracies = []

  def on_batch_end( self, batch, logs={} ):
    t = time.time()
    if not self.prev_t is None:
      self.times.append( t - self.prev_t )
      self.losses.append( logs['loss'] )
      self.accuracies.append( logs['acc'] )
    self.prev_t = t 

  def finalize( self ):
    data = pd.DataFrame( dict(
      times = self.times,
      losses = self.losses,
      accuracies = self.accuracies
    ))
    data.to_csv( os.path.join( args.jobdir, f'data.{tfslurm.my_job_name}_{my_job_index}.csv'))

model.compile(
  loss='categorical_crossentropy',
  optimizer=tf.keras.optimizers.SGD( lr=5e-6 ),
  metrics=['accuracy']
)

tfslurm.logger.info( f'Begin training {EPOCHS} steps with batch size {BATCH_SIZE}' )
callback = MonitoringCallback()
model.fit_generator(
  generator=generator( X, Y, BATCH_SIZE ),
  steps_per_epoch=len(X)//(BATCH_SIZE*4),
  epochs=EPOCHS,
  callbacks=[callback]
)
callback.finalize()

