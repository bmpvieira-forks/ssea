'''
Created on Nov 6, 2013

@author: mkiyer
'''
import os
import logging
import collections
import numpy as np

FLOAT_DTYPE = np.float32

class CountMatrix(object):
    def __init__(self):
        pass

class BigCountMatrix(CountMatrix):
    MEMMAP_FILE = 'counts.memmap'
    MEMMAP_T_FILE = 'counts.transpose.memmap'
    ISOFORM_FRAC_FILE = 'isoform_frac.memmap'
    ROWNAMES_FILE = 'rownames.txt'
    COLNAMES_FILE = 'colnames.txt'
    SIZE_FACTORS_FILE = 'size_factors.txt'

    def __init__(self):
        self.rownames = []
        self.colnames = []
        self.counts = None
        self.counts_t = None
        self.size_factors = None
        self.isoform_fracs = None
    
    @property
    def shape(self):
        return (len(self.rownames),len(self.colnames))
    
    @staticmethod
    def open(input_dir):
        counts_file = os.path.join(input_dir, BigCountMatrix.MEMMAP_FILE)
        counts_t_file = os.path.join(input_dir, BigCountMatrix.MEMMAP_T_FILE)
        rownames_file = os.path.join(input_dir, BigCountMatrix.ROWNAMES_FILE)
        colnames_file = os.path.join(input_dir, BigCountMatrix.COLNAMES_FILE)
        size_factors_file = os.path.join(input_dir, BigCountMatrix.SIZE_FACTORS_FILE)
        isoform_frac_file = os.path.join(input_dir, BigCountMatrix.ISOFORM_FRAC_FILE)
        self = BigCountMatrix()
        self.matrix_dir = input_dir
        with open(rownames_file, 'r') as fileh:
            self.rownames = [line.strip() for line in fileh]
        with open(colnames_file, 'r') as fileh:
            self.colnames = [line.strip() for line in fileh]
        self.counts = np.memmap(counts_file, dtype=FLOAT_DTYPE, mode='r', 
                                shape=(len(self.rownames), len(self.colnames)))
        self.counts_t = np.memmap(counts_t_file, dtype=FLOAT_DTYPE, mode='r', 
                                  shape=(len(self.colnames), len(self.rownames)))
        if os.path.exists(size_factors_file):
            with open(size_factors_file, 'r') as fileh:
                self.size_factors = np.array([float(line.strip()) for line in fileh])
        if os.path.exists(isoform_frac_file):
            self.isoform_fracs = np.memmap(isoform_frac_file, 
                                           dtype=FLOAT_DTYPE, 
                                           mode='r',
                                           shape=(len(self.rownames), 
                                                  len(self.colnames)))
        return self

    def close(self):
        del self.counts
        del self.counts_t
        del self.isoform_fracs
        
    def copy(self, output_dir, rowsubset=None, colsubset=None):
        if (rowsubset is None) or (len(rowsubset) == 0):
            rowsubset = set(self.rownames)
        else:
            rowsubset = set(rowsubset)
        if (colsubset is None) or (len(colsubset) == 0):
            colsubset = set(self.colnames)
        else:
            colsubset = set(colsubset)  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        counts_file = os.path.join(output_dir, BigCountMatrix.MEMMAP_FILE)
        counts_t_file = os.path.join(output_dir, BigCountMatrix.MEMMAP_T_FILE)
        rownames_file = os.path.join(output_dir, BigCountMatrix.ROWNAMES_FILE)
        colnames_file = os.path.join(output_dir, BigCountMatrix.COLNAMES_FILE)        
        size_factors_file = os.path.join(output_dir, BigCountMatrix.SIZE_FACTORS_FILE)
        isoform_frac_file = os.path.join(output_dir, BigCountMatrix.ISOFORM_FRAC_FILE)
        # write rownames and colnames
        with open(rownames_file, 'w') as fileh:
            row_inds = []
            ind = 0
            for rowname in self.rownames:
                if rowname in rowsubset:
                    print >>fileh, rowname
                    row_inds.append(ind)
                ind += 1
        with open(colnames_file, 'w') as fileh:
            col_inds = []
            ind = 0
            for colname in self.colnames:
                if colname in colsubset:
                    print >>fileh, colname
                    col_inds.append(ind)
                ind += 1
        # write rows
        fp = np.memmap(counts_file, dtype=FLOAT_DTYPE, mode='w+', 
                       shape=(len(row_inds),len(col_inds)))
        for i,ind in enumerate(row_inds):
            fp[i,:] = np.array(self.counts[ind,col_inds], dtype=np.float)
        del fp
        # write cols
        fp = np.memmap(counts_t_file, dtype=FLOAT_DTYPE, mode='w+', 
                       shape=(len(col_inds),len(row_inds)))
        for j,ind in enumerate(col_inds):
            fp[j,:] = np.array(self.counts_t[ind,row_inds], dtype=np.float)
        del fp
        # size factors
        if self.size_factors is not None:
            with open(size_factors_file, 'w') as fileh:
                for ind in col_inds:
                    print >>fileh, self.size_factors[ind]
        # isoform fractions
        if self.isoform_fracs is not None:
            fp = np.memmap(isoform_frac_file, dtype=FLOAT_DTYPE, mode='w+',
                           shape=(len(row_inds),len(col_inds)))
            for i,ind in enumerate(row_inds):
                fp[i,:] = np.array(self.isoform_fracs[ind,col_inds], dtype=np.float)
            del fp

    @staticmethod
    def from_tsv(input_file, output_dir, na_values=None):
        '''
        convert/copy a tab-delimited file containing numeric weight data
        to a BigCountMatrix (memmap files)    
        
        input_file: string path to tab-delimited matrix file
        output_dir: output directory
        na_val: set of values corresponding to missing data 
        '''
        if na_values is None:
            na_values = set(['NA'])
        else:
            na_values = set(na_values)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        counts_file = os.path.join(output_dir, BigCountMatrix.MEMMAP_FILE)
        counts_t_file = os.path.join(output_dir, BigCountMatrix.MEMMAP_T_FILE)
        rownames_file = os.path.join(output_dir, BigCountMatrix.ROWNAMES_FILE)
        colnames_file = os.path.join(output_dir, BigCountMatrix.COLNAMES_FILE)        
        self = BigCountMatrix()
        self.matrix_dir = output_dir
        # get rownames and colnames        
        with open(input_file, 'r') as fileh:
            header_fields = fileh.next().strip().split('\t')
            self.colnames = list(header_fields[1:])
            for line in fileh:
                rowname = line.strip().split('\t',1)[0]
                self.rownames.append(rowname)
        
        # write rownames and colnames
        with open(rownames_file, 'w') as fileh:
            for rowname in self.rownames:
                print >>fileh, rowname
        with open(colnames_file, 'w') as fileh:
            for colname in self.colnames:
                print >>fileh, colname

        # create memmap files
        self.counts = np.memmap(counts_file, dtype=FLOAT_DTYPE, mode='w+', 
                                shape=(len(self.rownames), len(self.colnames)))
        with open(input_file, 'r') as fileh:
            fileh.next() # skip header
            for i,line in enumerate(fileh):
                fields = line.strip().split('\t')
                # convert to floats and store
                counts = [(np.nan if x in na_values else float(x))
                          for x in fields[1:]]
                self.counts[i,:] = counts
        # write transpose
        self.counts_t = np.memmap(counts_t_file, dtype=FLOAT_DTYPE, mode='w+', 
                                  shape=(len(self.colnames), len(self.rownames)))
        self.counts_t[:] = self.counts.T[:]
        return self

    def _estimate_size_factors_deseq(self):
        '''
        Implements the procedure as described in DESeq:
        
        Each column is divided by the geometric means of the rows.
        the median of these ratios (skipping genes with zero values)
        is used as the size factor for each column.
        '''
        nrows = len(self.rownames)
        ncols = len(self.colnames)
        # compute row geometric means
        geomeans = np.empty(nrows, dtype=np.float)
        for i in xrange(nrows):
            a = np.around(self.counts[i,:])
            valid = np.logical_and((a > 0), np.isfinite(a))
            if np.all(valid):
                geomeans[i] = np.exp(np.log(a[valid]).mean())
            else:
                geomeans[i] = np.nan
            if i % 10000 == 0:
                logging.debug("%d %d" % (i, np.isfinite(geomeans[:i]).sum()))
        #print 'found', np.isfinite(geomeans).sum()
        # ignore rows of zero or nan
        valid_geomeans = np.logical_and((geomeans > 0), np.isfinite(geomeans))
        size_factors = np.empty(ncols, dtype=np.float)
        for j in xrange(ncols):
            a = np.around(self.counts_t[j,:])
            lib_valid = np.logical_and((a > 0), np.isfinite(a))
            valid = np.logical_and(lib_valid, valid_geomeans)
            if np.any(valid):
                size_factors[j] = np.median(a[valid] / geomeans[valid])
            else:
                size_factors[j] = np.nan
            print j, self.colnames[j], size_factors[j], valid.sum(), a[valid].sum()
        return size_factors

    def estimate_size_factors(self, method='deseq'):
        '''
        estimate size factors (normalization factors) for libraries
        '''
        ncols = len(self.colnames)
        if method == 'deseq':
            size_factors = self._estimate_size_factors_deseq()
        elif method == 'median':
            size_factors = np.empty(ncols, dtype=np.float)
            for j in xrange(ncols):
                a = self.counts_t[j,:]
                valid = np.logical_and((a > 0),np.isfinite(a))
                size_factors[j] = np.median(a[valid])
            size_factors = size_factors / np.median(size_factors)
        elif method == 'total':
            size_factors = np.empty(ncols, dtype=np.float)
            for j in xrange(ncols):
                a = self.counts_t[j,:]
                a = a[np.isfinite(a)]
                size_factors[j] = a.sum()
            size_factors = size_factors / np.median(size_factors)
        self.size_factors = size_factors
        size_factors_file = os.path.join(self.matrix_dir, BigCountMatrix.SIZE_FACTORS_FILE)
        with open(size_factors_file, 'w') as fileh:
            for x in self.size_factors:
                print >>fileh, x
        return size_factors
    
    def calculate_isoform_fractions(self, isoforms_file):
        # determine gene-transcript associations (one-to-many)
        transcript_row_map = dict((x,i) for i,x in enumerate(self.rownames))
        gene_row_map = collections.defaultdict(lambda: [])
        with open(isoforms_file) as fileh:
            for i,line in enumerate(fileh):
                fields = line.strip().split('\t')
                transcript_id = fields[0]
                if transcript_id not in transcript_row_map:
                    continue
                gene_id = fields[1]
                gene_row_map[gene_id].append(transcript_row_map[transcript_id])
        # create memmap file
        isoform_frac_file = os.path.join(self.matrix_dir, BigCountMatrix.ISOFORM_FRAC_FILE)
        self.isoform_fracs = np.memmap(isoform_frac_file, dtype=FLOAT_DTYPE, mode='w+', 
                                       shape=(len(self.rownames), len(self.colnames)))
        for gene_id in sorted(gene_row_map):
            indexes = np.array(gene_row_map[gene_id])
            counts = np.array(self.counts[indexes,:])
            # ignore 'nan' values by setting them to zero
            nancounts = np.isnan(counts)
            counts[nancounts] = 0
            # convert from raw counts to fractions of total for gene
            countsums = counts.sum(axis=0)
            nzsums = countsums.nonzero()[0]
            fracs = np.zeros_like(counts)
            fracs[:,nzsums] = counts[:,nzsums] / countsums[nzsums]
            fracs[nancounts] = np.nan
            self.isoform_fracs[indexes,:] = fracs
