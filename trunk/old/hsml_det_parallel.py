from mpi4py import MPI
from scipy.spatial import cKDTree
import numpy as np
import gadget_type1 as gd
import getopt
import sys

def usage():
    print '\nUsage: -i --input=  --> input_file'
    print '       -o --output= --> output file'
    print '       -v --verbose --> verbose mode'
    print '       -n --neighbor --> distance to the n-th neighbor\n'
    exit()

def opt_reader():
    #default values:
    verbose = False        
    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           'i:o:v',
                                           ['input=',
                                            'output=',
                                            'verbose',
                                            'nb='])
    except getopt.GetoptError, err:
        print str(err)
        usage()
    output_file = input_file = ''
    nb = 16
    for opt,arg in options:
        if opt in ('-o', '--output'):
            output_file = arg
            continue
        if opt in ('-i', '--input'):
            input_file = arg
            continue
        if opt in ('-v', '--verbose'):
            verbose = True
            continue
        if opt in ('--nb'):
            nb = arg
            continue

    opt = {'verbose': verbose,
           'input': input_file,
           'output': output_file,
           'neighbor': nb}
    return opt


opt = opt_reader()
verbose     = opt['verbose']
input_file  = opt['input']
output_file = opt['output']
nb          = opt['neighbor']

nb = np.int(nb)

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

if(rank == 0):
	if verbose: 
		print 'Opening file: ', input_file
		print 'Number of neighbors: ', nb
	snap = gd.snapshot(input_file)
	pos  = snap.Pos('drk')
else:
	pos  = None

pos = comm.bcast(pos, root=0)
if verbose: print 'Processor', rank, ' has received the particles array.'

N     = np.shape(pos)[1]
npart = np.int(N/size)
rest  = np.mod(N,size)

pos = np.reshape(pos,(3,N))

if verbose: print 'Processor ', rank, ' is Building 3DTree...'
tree = cKDTree(pos.T)
if verbose: print 'Processor ', rank, ' is Searching the ', nb, 'neighbors of each particle...'

if(rank <= rest):
	offset = rank*(npart+1)
else:
	offset = rank*npart+rest
	
if(rank < rest):
	npart = npart+1

d, idx = tree.query(pos[:,offset:offset+npart].T, k=nb)
hsml_rank = d[:,nb-1]

if verbose: print 'Processor ', rank, 'is gathering the smoothing lenght.'
recvhsml = comm.gather(hsml_rank, root=0)

if(rank == 0):
	hsml = np.ndarray([0])
	for i in xrange(size):
		hsml = np.append(hsml,recvhsml[i][:])
	if verbose: print 'Writing the output file ', output_file	
	fo = open(output_file, 'wb')
	(hsml.astype(np.float32)).tofile(fo)
	print 'Done'
