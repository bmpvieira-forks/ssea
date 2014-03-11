import sys
import os
import collections
import operator
import logging
import argparse
import glob

# third-party imports
import numpy as np

# local imports
from ssea.lib.config import Config 
from ssea.lib.countdata import BigCountMatrix
from ssea.lib.base import Result, JobStatus

def check_path(path):
    if not os.path.exists(path):
        logging.error('Directory not found at path "%s"' % (path))
        return False
    elif not os.path.isdir(path):
        logging.error('Directory not valid at path "%s"' % (path))
        return False
    elif JobStatus.get(path) != JobStatus.DONE:
        logging.info('Result status not set to "DONE" at path "%s"' % (path))
        return False
    return True

def find_paths(input_dirs):
    paths = set()
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            logging.error("input dir '%s' not found" % (input_dir))
        if not os.path.isdir(input_dir):
            logging.error("input dir '%s' not a valid directory" % (input_dir))
        input_dir = os.path.abspath(input_dir)
        for path in glob.iglob(os.path.join(input_dir, "*")):
            if os.path.exists(path) and os.path.isdir(path):
                if check_path(path):
                    paths.add(path)
    return paths

def parse_results(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            result = Result.from_json(line.strip())
            yield result


def get_sample_set_id(db, sample_set_name):
    coll = db[SAMPLE_SETS]
    spec = {'name': sample_set_name}
    ss_id = coll.find_one(spec, ['_id'])['_id']
    return ss_id

def get_transcript_metadata(db, t_id, fields=None):
    coll = db[ROW_META]
    spec = {'_id': t_id}
    r = coll.find_one(spec, fields)
    return r

def get_top_transcripts(db, ss_id, fdr_threshold, topn, sortorder):
    # get results for this sample set id
    coll = db[SORTCOLLECTION]
    spec = { 'ss_id': ss_id }
    proj = ['t_id', 'nes', 'ss_fdr_q_value']
    # get top transcripts up/down
    cur = coll.find(spec, proj)
    cur.sort(SORTBY, sortorder)
    cur.limit(topn)
    top_transcripts = []
    for res in cur:
        if res['ss_fdr_q_value'] > fdr_threshold:
            continue
        top_transcripts.append((res['t_id'], res['nes']))
    return top_transcripts

def get_heatmap_data(db, t_ids, ss_ids):
    coll = db[RESULTS]
    data = np.zeros((len(t_ids),len(ss_ids)), dtype=np.float)
    ss_ind_map = dict((ss_id,i) for i,ss_id in enumerate(ss_ids))
    for i,t_id in enumerate(t_ids):
        logging.debug("Row %d / %d" % (i, len(t_ids)))
        for r in coll.find({'t_id': t_id, 'ss_id': {'$in': ss_ids}},
                           {'ss_id': 1, 'nes': 1}):
            ss_id = r['ss_id']
            j = ss_ind_map[ss_id]
            data[i,j] = r['nes']
    return data
#    t_ind_map = dict((t_id,i) for i,t_id in enumerate(t_ids))
#     for j,ss_id in enumerate(ss_ids):
#         cur = coll.find({'t_id': {'$in': t_ids}, 'ss_id': ss_id},
#                         {'t_id': 1, 'nes': 1})
#         for r in cur:
#             t_id = r['t_id']
#             i = t_ind_map[t_id]
#             data[i,j] = r['nes']
#     return data

def get_heatmap(input_paths, bm,
                fdr_threshold, 
                frac_threshold, 
                prec_threshold):

    t_id_dict = {}
    ss_ids = []
    
    for input_path in input_paths:
        logging.debug('Parsing path %s' % (input_path))
        results_file = os.path.join(input_path, Config.RESULTS_JSON_FILE)
        # extract data
        ss_compname = os.path.basename(input_path)
        i = 0
        sig = 0
        for res in parse_results(results_file):
            # logging
            i += 1
            if (i % 10000) == 0:
                logging.debug('Parsed %d results' % (i))
            transcript_id = bm.rownames[res.t_id]
            # calc precision
            core_size = res.core_hits + res.core_misses
            if core_size == 0:
                prec = 0.0
            else:
                prec = res.core_hits / float(res.core_hits + res.core_misses)
            if ((res.ss_fdr_q_value <= fdr_threshold) and 
                (abs(res.ss_frac) >= frac_threshold) and
                (prec >= prec_threshold)):
                if transcript_id in t_id_dict:
                    cur_es = t_id_dict[transcript_id][1]
                else:
                    cur_es = 0.0
                if abs(cur_es)
                

        # merge sample set t_ids
        for t_id,nes in top_transcripts:
            if t_id in t_id_dict:
                cur_nes = t_id_dict[t_id][1]
            else:
                cur_nes = 0.0 
            if abs(nes) > abs(cur_nes):
                logging.debug("\tReplaced transcript %d NES=%f with NES=%f" % (t_id, cur_nes, nes))
                t_id_dict[t_id] = (ss_id, nes)
                
                fields.extend([ss_compname,
                               res.es,
                               res.nes,
                               res.ss_fdr_q_value,
                               res.ss_frac,
                               prec])
                print '\t'.join(map(str, fields))
                sig += 1
        logging.debug('Found %d results for path %s' % (sig, input_path))

    pass


def get_heatmap_table(database, study, sample_set_names, fdr_threshold, topn, topfrac):
    client = pymongo.MongoClient(database)
    db = client[study]
    # count number of transcripts in the study
    coll = db[ROW_META]
    nrows = coll.count()
    # find number of transcripts to report per sample set
    if topn is not None:
        topn = min(nrows, topn)
    else:
        topfrac = max(0.0, topfrac)
        topfrac = min(1.0, topfrac)
        topn = int(round(nrows * topfrac))
    # get list of top transcripts
    logging.debug("Querying %d sample sets for top %d transcripts up/down "
                  "with FDR threshold <= %f" % 
                  (len(sample_set_names), topn, fdr_threshold))
    t_id_dict = {}
    ss_ids = []
    for sample_set_name in sample_set_names:
        logging.debug("\tSample Set: %s" % (sample_set_name))
        # convert sample set names to id
        ss_id = get_sample_set_id(db, sample_set_name)
        ss_ids.append(ss_id)
        top_transcripts = get_top_transcripts(db, ss_id, fdr_threshold, topn, sortorder=-1)
        logging.debug("\tFound %d transcripts" % (len(top_transcripts)))
        # merge sample set t_ids
        for t_id,nes in top_transcripts:
            if t_id in t_id_dict:
                cur_nes = t_id_dict[t_id][1]
            else:
                cur_nes = 0.0 
            if abs(nes) > abs(cur_nes):
                logging.debug("\tReplaced transcript %d NES=%f with NES=%f" % (t_id, cur_nes, nes))
                t_id_dict[t_id] = (ss_id, nes)
    # sort t_ids by ss_id
    logging.info("Found %d total transcripts" % (len(t_id_dict)))
    logging.debug("Ordering transcripts by sample set and NES")
    ss_id_dict = collections.defaultdict(lambda: [])
    for t_id in t_id_dict:
        ss_id, nes = t_id_dict[t_id]
        ss_id_dict[ss_id].append((nes, t_id))
    # order transcripts by sample set and sort by nes
    t_ids = []
    for ss_id in ss_ids:
        ss_t_ids = [tup[1] for tup in sorted(ss_id_dict[ss_id], 
                                             key=operator.itemgetter(0), 
                                             reverse=True)]
        t_ids.extend(ss_t_ids)
        #logging.debug('ss id ' + str(ss_id) + ' t ids ' + str(len(t_ids)))
        #t_ids = [tup[1] for tup in sorted(ss_id_dict_down[ss_id], key=operator.itemgetter(0))]
        #t_ids_down.extend(t_ids)
    # get nes data for entire heatmap
    logging.debug("Querying for heatmap data")
    data_up = get_heatmap_data(db, t_ids, ss_ids)
    #data_down = get_heatmap_data(db, t_ids_down, ss_ids)
    # output data
    logging.debug("Writing output")
    header_fields = ['name', 'enrichment', 'gene_id', 'locus', 'category', 'nearest_gene_names']
    header_fields.extend(sample_set_names)
    print '\t'.join(header_fields)
    for i,t_id in enumerate(t_ids):
        r = get_transcript_metadata(db, t_id)
        fields = [r['name'], 
                  'up',
                  r['gene_id'],
                  '%s[%s]' % (r['locus'], r['strand']),
                  '%s' % (r['category']),
                  '%s' % (r['nearest_gene_names'])]
        fields.extend(map(str,data_up[i,:]))
        print '\t'.join(fields)
#    for i,t_id in enumerate(t_ids_down):
#        r = get_transcript_metadata(db, t_id)
#        fields = [r['name'], 
#                  'down',
#                  r['gene_id'],
#                  '%s[%s]' % (r['locus'], r['strand']),
#                  '%s' % (r['category']),
#                  '%s' % (r['nearest_gene_names'])]
#        fields.extend(map(str,data_down[i,:]))
#        print '\t'.join(fields)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdr', dest='fdr', type=float, default=1.0)
    parser.add_argument('--frac', dest='frac', type=float, default=0.0)
    parser.add_argument('--meta', dest='metadata_file', default=None)
    parser.add_argument("-i", dest="input_paths_file", default=None)
    parser.add_argument('matrix_dir')
    args = parser.parse_args()
    # get args
    matrix_dir = args.matrix_dir
    fdr_threshold = args.fdr
    frac_threshold = args.frac
    input_paths_file = args.input_paths_file
    metadata_file = args.metadata_file
    # check args
    bm = BigCountMatrix.open(matrix_dir)
    fdr_threshold = max(0.0, min(fdr_threshold, 1.0))
    frac_threshold = max(0.0, min(frac_threshold, 1.0))
    input_paths = []
    if input_paths_file is None:
        logging.error('No input directories specified (use -i).. Exiting.')
        return 1
    if not os.path.exists(input_paths_file):
        logging.error('Input paths file "%s" not found' % (input_paths_file))
    else:
        with open(input_paths_file) as fileh:
            for line in fileh:
                path = line.strip()
                if path in input_paths:
                    continue
                if check_path(path):
                    input_paths.append(path)
    if len(input_paths) == 0:
        logging.error('No valid SSEA results directories found.. Exiting.')
        return 1
    meta = None
    if metadata_file is not None:
        logging.debug('Parsing transcript metadata')
        meta = {}
        with open(metadata_file) as f:
            meta_header_fields = f.next().strip().split()
            for line in f:
                fields = line.strip().split('\t')
                meta[fields[0]] = fields
        logging.debug('Found metadata for %d transcripts' % (len(meta)))
    else:
        meta = None
        meta_header_fields = ['transcript_id']
    # parse results
    logging.debug('SSEA results: %d' % (len(input_paths)))
    logging.debug('FDR threshold: %f' % (fdr_threshold))
    logging.debug('Frac threshold: %f' % (frac_threshold))
    header_fields = meta_header_fields + ['ss_compname', 'es', 'nes', 'fdr', 'frac', 'prec']
    print '\t'.join(header_fields)
    for input_path in input_paths:
        logging.debug('Parsing path %s' % (input_path))
        results_file = os.path.join(input_path, Config.RESULTS_JSON_FILE)
        # extract data
        ss_compname = os.path.basename(input_path)
        i = 0
        sig = 0
        for res in parse_results(results_file):
            # logging
            i += 1
            if (i % 10000) == 0:
                logging.debug('Parsed %d results' % (i))
            transcript_id = bm.rownames[res.t_id]
            if meta is not None:
                if transcript_id not in meta:
                    continue
            if ((res.ss_fdr_q_value <= fdr_threshold) and 
                (abs(res.ss_frac) >= frac_threshold)):
                if meta is None:
                    fields = [bm.rownames[res.t_id]]
                else:
                    fields = list(meta[transcript_id])
                core_size = res.core_hits + res.core_misses
                if core_size == 0:
                    prec = 0.0
                else:
                    prec = res.core_hits / float(res.core_hits + res.core_misses)
                fields.extend([ss_compname,
                               res.es,
                               res.nes,
                               res.ss_fdr_q_value,
                               res.ss_frac,
                               prec])
                print '\t'.join(map(str, fields))
                sig += 1
        logging.debug('Found %d results for path %s' % (sig, input_path))
    bm.close()

if __name__ == '__main__':
    sys.exit(main())

