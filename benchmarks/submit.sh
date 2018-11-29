# Generate a slurm jobfile from template and submit it 

# CD to the directory of this file, so that we submit from /benchmarks/
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Make sure email to is set, it should not be set from env file due to privacy reasons
if [ -z $EMAIL_TO ]; then echo 'Error: $EMAIL_TO env variable must be set'; exit 1; fi 

# Get job directory, either from env or as first positional argument
JOBDIR=`basename ${JOBNAME:-$1}`
# Make sure it is set
if [ -z $JOBDIR ]; then echo 'Error: $JOBNAME env variable must be set or passed as first positional argument'; exit 1; fi
echo "Using jobdir: $JOBDIR"
# And that it exists
if [ ! -d $JOBDIR ]; then echo "Error: job directory does not exist"; exit 1; fi


# Get environment file
ENVFILE=`python -c "import os; print( os.path.join('$JOBDIR', 'env.sh' ))"`
echo "Using envfile: $ENVFILE"
# Make sure it exists
if [ ! -f $ENVFILE ]; then echo "Error: env file does not exist"; exit 1; fi
# And source it
source $ENVFILE

# Relevant $ENVFILE variables
# LOCAL - whether to run locally, tensorflow will not be distributed if this is the case
# GPUS  - which gpus to book for the job, e.g. "gpu:P100:2"
# TASKS - how many tasks to submit
# NODES - how many nodes to book
# TASKS_PER_NODE - max tasks allocated to each node
# NUM_PS - how many parameter servers to use (only relevant when local != true)
# BATCH_SIZE - batch size to use in each task


# Make sure all environment variables are set 
if [ "$LOCAL" != "true" ] && [ "$LOCAL" != "false" ]; then echo 'Error: $LOCAL env variable must be "true" or "false"'; fi
if [ -z $GPUS ];           then echo 'Error: $GPUS env variable is not set'; exit 1; fi
if [ -z $TASKS ];          then echo 'Error: $TASKS env variable is not set'; exit 1; fi
if [ -z $NODES ];          then echo 'Error: $NODES env variable is not set'; exit 1; fi
if [ -z $TASKS_PER_NODE ]; then echo 'Error: $TASKS_PER_NODE env variable is not set'; exit 1; fi
if [ -z $NUM_PS ];         then echo 'Error: $NUM_PS env variable is not set'; exit 1; fi
if [ -z $BATCH_SIZE ];     then echo 'Error: $BATCH_SIZE env variable is not set'; exit 1; fi


# Grep uses different switches based on os
if [[ "$OSTYPE" == "darwin"* ]]; then
    GREP_SWITCH="-E" # Mac (for testing generation)
else
    GREP_SWITCH="-P" # Linux
fi
# Check for existing versoins
VERSIONS=`ls $JOBDIR | grep "$GREP_SWITCH" '^v\d+$'`
echo "Found versions: `echo $VERSIONS | sed`"
# Set next version to be the highest current version, incremented by one
NEXT_VERSION=`python -c "print( max( map( int, ( x[1:] for x in \"\"\"v-1\n$VERSIONS\"\"\".split('\n') if x.strip() ))) + 1 )"`
VERSIONDIR=`python -c "import os; print( os.path.join( '$JOBDIR', 'v$NEXT_VERSION' ))"`
echo "Next version: $VERSIONDIR"
if [ -d $VERSIONDIR ]; then echo 'Error: version dir already exists'; exit 1; fi 
# Prompt user before continuing
read -p "Submit job in $VERSIONDIR? [Yy]: "
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Ok. Exiting..."
    exit 1
fi
# Create new dir for version
mkdir $VERSIONDIR


# Generate job file
JOBFILE="$VERSIONDIR/job.slurm"
echo "Generating jobfile: $JOBFILE"
# Add extra arg for running locally if configured
if [ $LOCAL == 'true' ]; then
    EXTRA_ARGS="--local"
else
    EXTRA_ARGS=""
fi 
# Escape filepaths, as they will otherwise mess with sed expressions
VERSIONDIR_ESCAPED=$(echo $VERSIONDIR | sed -e 's/\//\\\//')
# Parse template
cat job.template.slurm | sed \
    -e 's/{email_to}/'$EMAIL_TO'/' \
    -e 's/{jobdir}/'$VERSIONDIR_ESCAPED'/' \
    -e 's/{tasks}/'$TASKS'/' \
    -e 's/{nodes}/'$NODES'/' \
    -e 's/{tasks_per_node}/'$TASKS_PER_NODE'/' \
    -e 's/{gpus}/'$GPUS'/' \
    -e 's/{batch_size}/'$BATCH_SIZE'/' \
    -e 's/{num_ps}/'$NUM_PS'/' \
    -e 's/{extra_args}/'$EXTRA_ARGS'/' > $JOBFILE


exit 0
sbatch $JOB_FILE
echo "Submitted: $JOB_FILE"