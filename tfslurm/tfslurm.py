from __future__ import division 
from __future__ import print_function 

import tensorflow as tf 
import logging 
import sys 
import os 
import re 

class TFSlurm( object ):
  '''
  Convenience class for running tf on slurm.
  '''
  PARAMETER_SERVER = 'ps'
  WORKER = 'worker'


  def __init__( self, cluster, my_job_name, my_job_index, gpu_frac=None, logger=None ): 
    self.cluster = cluster 
    self.my_job_name = my_job_name
    self.my_job_index = my_job_index
    self.gpu_frac = gpu_frac or 1.0
    self.logger = logger or logging.getLogger( 'tfslurm' )
    self._server = None

  @property
  def num_ps( self ):
    ''' Get the number of parameter servers in the configured cluster '''
    return len( self.cluster['ps'] )

  @property
  def num_workers( self ):
    ''' Get the number of workers in the configured cluster '''
    return len( self.cluster['worker'] )

  @property
  def is_ps( self ):
    ''' Returns whether this process/task is a parameter server'''
    return self.my_job_name == 'ps'

  @property
  def is_worker( self ):
    ''' Returns whether this process/task is a worker '''
    return not self.is_ps 

  @property
  def is_chief( self ):
    ''' Returns whether this process/task is the chief worker'''
    return self.my_job_index == 0 

  @property
  def device_setter( self ):
    ''' Get a device setter that can be used in "with tf.device" context '''
    return tf.train.replica_device_setter(
      worker_device=self.worker_device,
      cluster=self.cluster,
    )

  @property
  def worker_device( self ):
    ''' Generate device string for this process/task's worker '''
    return '/job:worker/task:{}'.format( self.my_job_index )

  @property
  def ps_devices( self ):
    ''' Get a list of device strings for all parameter servers '''
    return ['/job:ps/task:%s'%i for i in range( self.num_ps ) ]

  @property
  def server(self):
    ''' Get (and possibly create) the server used to communicate between tasks '''
    if self._server is None:
      self.logger.info( 'Creating server' )
      self._server = tf.train.Server( 
        server_or_cluster_def = tf.train.ClusterSpec( self.cluster ),
        job_name = self.my_job_name,
        task_index = self.my_job_index
      )
    return self._server

  def start_server( self ):
    ''' 
    Start the server, but only if this process/task 
    is supposed to start one. Then wait for training to
    complete and exit.

    NOTE: No code will be executed beyond this method if
    the current process/task is configured to be PS
    '''
    self.server
    if self.is_ps:
      self.logger.info( 'Running server as PS' )
      self.server.join()
      sys.exit( 0 )

    return self


  def synched_optimizer( self, opt ):
    ''' 
    Wrap a normal tensorflow optimizer in a SynchReplicasOptimizer
    to prevent stale gradients.
    See: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
    
    Args:
    - opt : Optimizer
      Any tensorflow optimizer 

    Returns:
      A tf.train.SyncReplicasOptimizer that wraps
      the original optimizer
    '''
    return tf.train.SyncReplicasOptimizer( opt, 
      replicas_to_aggregate=self.num_workers,
      total_num_replicas=self.num_workers
    )


  def synced_training_session( self, opt, **kwargs ):
    '''
    Create a distributed training session, with a hook for
    handling initialization and queues

    Args:
    - opt : tf.Train.SyncReplicasOptimizer
      A synched replicas optimizer. Use (TFSlurm.synched_optimizer to get this)
    - **kwargs
      Extra keyword args that should be passed to the session constructor

    Returns:
      A tf.train.MonitoredTrainingSession object
    '''
    sync_replicas_hook = opt.make_session_run_hook( self.is_chief )
    return tf.train.MonitoredTrainingSession(
      master = self.server.target,
      is_chief = self.is_chief,
      hooks = kwargs.pop( 'hooks', [] ) + [ sync_replicas_hook ],
      **kwargs
    )


  @classmethod
  def create_local( cls, logger_opts=None ):
    '''
    Create a TFSlurm dummy object that contains the same methods
    and properties, except intended for local use.
    '''
    logger = cls.create_logger( 'local', **(logger_opts or {}) )
    return TFLocal( logger )


  @classmethod
  def from_slurm_env( cls, num_ps, base_port=2222, logger_opts=None ):
    '''
    Create a TFSlurm object from the environment variables
    passed by slurm.

    Args:
    - num_ps : int
      Number of parameter servers to use. The first 3
      will be allocated
    - base_port : int
      Port number used by first task on a node, subsequent
      tasks on the same node will get incrementing port numbers
    - logger_opts : dict
      Optional parameters sent to the logger creation function
      See "create_logger" for more details

    Returns 
      An initialized TFSlurm object
    '''
    # Get the total number of nodes allocated for the job
    num_nodes = int( os.environ['SLURM_JOB_NUM_NODES'] )
    # Parse the list of nodes the job is configured to run over
    nodelist = cls.parse_nodelist( os.environ['SLURM_NODELIST'] )
    # Parse the list of n_tasks/num_procs for each node
    tasklist = cls.parse_tasklist( os.environ['SLURM_TASKS_PER_NODE'] )
    # Get the name of the node this task is running on
    my_node = os.environ['SLURMD_NODENAME']
    # Get the total number of tasks/procs running across all nodes
    num_procs = int( os.environ['SLURM_NPROCS'] )
    # Get the global index of tasks/procs for the job
    my_proc  = int( os.environ['SLURM_PROCID'] )
    # Get the gpus available on each node
    gpulist  = os.environ['SLURM_JOB_GPUS'].split(',')

    # Create a logger 
    logger = cls.create_logger( '%s_%s'%(my_proc,my_node), **(logger_opts or {}) )

    # Make sure the numbers adds up
    errors = []
    if not num_nodes == len( nodelist ):
      error = 'Inconsistent nodes: SLURM_JOB_NUM_NODES=%s, nodelist=%s (%s)'%(num_nodes, nodelist, os.environ['SLURM_NODELIST'] )
      logger.critical( error )
      errors.append( error )
    if not num_nodes == len( tasklist ):
      error = 'Inconsistent nodes: SLURM_JOB_NUM_NODES=%s, tasklist=%s (%s)'%(num_nodes, tasklist, os.environ['SLURM_TASKS_PER_NODE'] )
      logger.critical( error )
      errors.append( error )
    if not num_procs == sum( tasklist ):
      error = 'Inconsistent task allocation: SLURM_NPROCS=%s, tasklist=%s (%s)'%(num_procs, tasklist, os.environ['SLURM_TASKS_PER_NODE'] )
      logger.critical( error )
      errors.append( error )
    if num_ps > num_procs:
      error = 'More parameter servers (num_ps=%s) are requested than there are tasks (num_procs=%s)'%(num_ps, num_procs)
      logger.critical( error )
      errors.append( error )
    if errors:
      raise Exception( '%s errors occurred when processing slurm environment'%len(errors))

    # If everything checks out; set up table of task, nodes, and gpu allocations
    logger.info( 'Setting up task allocations' )
    logger.info( 'My node: %s'%my_node )
    logger.info( 'My proc: %s'%my_proc )
    # It is important that this allocation table is deterministic across all tasks
    tasktable = cls.create_task_table( num_ps, base_port, nodelist, tasklist, gpulist, my_node, my_proc )
    # Make sure a single instance task/proc exist in tasktable
    if not sum( t['me'] for t in tasktable ) == 1:
      error = 'Unique instance of task not found in task table (#me != 1): %s'%tasktable
      logger.critical( error )
      raise Exception( error )

    # Get the parameters for this process/task
    my_task = next( t for t in tasktable if t['me'] )
    logger.info( 'My Task: %s'%my_task )
    # Mask out non-assigned GPUs
    os.putenv('CUDA_VISIBLE_DEVICES', my_task['gpu'] )
    # Define cluster
    ps_tasks     = [ t for t in tasktable if t['role'] == cls.PARAMETER_SERVER ] 
    worker_tasks = [ t for t in tasktable if t['role'] == cls.WORKER ]
    cluster = { cls.PARAMETER_SERVER: [ '%s:%s'%(t['node'],t['port'] ) for t in ps_tasks ],
                cls.WORKER:           [ '%s:%s'%(t['node'],t['port'] ) for t in worker_tasks ]}
    
    # Create TFSlurm object
    return TFSlurm(
      cluster = cluster,
      my_job_name = my_task['role'],
      my_job_index = my_task['role_index'],
      gpu_frac = my_task['gpu_frac'],
      logger = logger
    )


  @classmethod
  def create_logger( cls, taskname, logdir=None, logfmt=None, loglevel=None, shlevel=None, fhlevel=None ):
    '''
    Set up a logger for the task. It's just a normal python logger
    configured such that ERROR and worse will be sent to stdout,
    which is shared between slurm tasks, whereas everything down
    to DEBUG will in addition be written to a dedicated file.
    
    Args:
    - taskname : string
      Name of task, used in the stream handler logging prefix and as
      filename for the filehandler
    - logidr : string 
      Optional directory for log files. If set, then the directory will be 
      created and a file handler will be attached to the logger
    - logfmt : string
      Optional format for log entries
    - loglevel : string | logging.<LEVEL>
      Optional overal level for logging. Defaults to logging.DEBUG
    - shlevel : string | logging.<LEVEL>
      Optional level for the stream handler
    - fhlevel : string | logging.<level>
      Optional level for the file handler

    Returns:
      A python logging logger
    '''
    logger = logging.getLogger( 'tfslurm' )
    logger.setLevel( loglevel or logging.DEBUG )
    # Create stream handler for ERRORS and above
    sh = logging.StreamHandler()
    sh.setLevel( shlevel or logging.ERROR )
    sh.setFormatter( logging.Formatter( logfmt or '%(levelname)-8s %(asctime)s '+taskname+' :: %(message)s' ))
    logger.addHandler( sh )
    # Create a filehandler for everything down to debug if a logdir is provided
    if logdir:
      if not os.path.exists( logdir ):
        os.makedirs( logdir )
      fh = logging.FileHandler( os.path.join( logdir, taskname+'.log' ))
      fh.setLevel( fhlevel or logging.INFO )
      fh.setFormatter( logging.Formatter( logfmt or '%(levelname)-8s %(asctime)s :: %(message)s' ))
      logger.addHandler( fh )

    return logger


  @classmethod
  def create_task_table( cls, num_ps, base_port, nodelist, tasklist, gpulist, my_node, my_proc ):
    '''
    Deterministically create a table with task/node/gpu allocations
    '''
    tasktable = []
    for node, num_tasks in zip( nodelist, tasklist ):
      # Generate list of even gpu allocations over the current node
      gpu_alloc = [int(min(num_tasks,len(gpulist))*task/num_tasks) for task in range( num_tasks )]
      # Create a config dictionary for each task on the node
      for task in range( num_tasks ):
        tasktable.append( dict(
          # Node the task is being run on
          node=node,
          # Local task index on node
          node_task=task,
          # Task type 
          role = cls.PARAMETER_SERVER if len(tasktable) < num_ps else cls.WORKER,
          # Global task index across all nodes
          global_task_index=len( tasktable ),
          # Task within role, e.g. ps:0,1,... or worker:0,1,2..
          role_index=len(tasktable) if len(tasktable) < num_ps else len(tasktable)-num_ps,
          # Port for parameter server communication
          port=base_port + task,
          # Allocated gpu for this task
          gpu=gpulist[gpu_alloc[task]],
          # Fraction of assigned gpu available to this task
          gpu_frac=1/sum( gpui==gpu_alloc[task] for gpui in gpu_alloc ),
          # Whether this task is the current process' task
          me=(node==my_node and len(tasktable)==int(my_proc))
        ))

    return tasktable 


  @classmethod
  def parse_nodelist( cls, nodelist ):
    '''
    Expand a nodelist, which might come in a compact form
    like

    Example:
    - "compute-3-0-[1-3,7]"
        -> ['compute-3-0-1','compute-3-0-2','compute-3-0-3',compute-3-0-7']

    Args:
    - nodelist : string
      List of nodes, raw from SLURM_JOB_NODELIST as shown in example

    Returns:
      A list of expanded node ids, where each is a valid identifier
      of a node on the cluster
    '''
    prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
    ids = cls.parse_idlist( ids )
    result = [prefix + str(id) for id in ids]
    return result


  @classmethod
  def parse_idlist( cls, idlist ):
    '''
    Exapand one or more nodeids. 
    '''
    idlist = idlist.split(',')
    result = []
    for id in idlist:
      if '-' in id:
        tokens = id.split('-')
        token = tokens[0]
        begin, end = [ int(token) for token in tokens ]
        result.extend( (str(t).rjust(len(token), '0') for t in range(begin, end+1)))
      else:
        result.append(id)
    return result


  @classmethod
  def parse_tasklist( cls, tasklist ):
    '''
    Expand the list saying how many tasks that are run on
    each node
    '''
    parts = tasklist.split(',')
    result = []
    for part in parts:
      m = re.match('(\d+)(\(x(\d+)\))?', part )
      # Check if there is an "x" multiplier, if so add all
      # otherwise, just add a single
      result.extend([ int(m.group(1)) for _ in range( int(m.group(3) or 1)) ])
    return result 



class TFLocal( TFSlurm ):

  num_ps        = 1 
  num_workers   = 1
  is_ps         = True
  is_worker     = True 
  is_chief      = True 
  device_setter = None 
  worker_device = None 
  ps_devices    = [ None ]
  server        = None 

  def __init__( self, logger=None, gpu_frac=None ):
    self.logger = logger or logging.getLogger( 'tfslurm' )
    self.cluster = {}
    self.my_job_name = 'local'
    self.my_job_index = 0
    self.gpu_frac = gpu_frac or 1.0
  
  def start_server( self ):
    return self

  def synched_optimizer( self, opt ):
    return opt 

  def synced_training_session( self, opt, **kwargs ):
    return tf.Session( **kwargs )