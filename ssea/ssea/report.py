'''
Created on Oct 29, 2013

@author: mkiyer
'''
import os
import json
import logging
import shutil
import matplotlib.pyplot as plt

# setup path to web files
import ssea
SRC_WEB_PATH = os.path.join(ssea.__path__[0], 'web')

# setup html template environment
from jinja2 import Environment, PackageLoader
env = Environment(loader=PackageLoader("ssea", "templates"),
                  extensions=["jinja2.ext.loopcontrols"])

# matplotlib static figure for plotting
global_fig = plt.figure(0)

class Report(object):
    # header fields for report
    FIELDS = ['name', 
              'desc',
              'sample_set_name',
              'sample_set_desc',
              'sample_set_size',
              'es',
              'nes',
              'nominal_p_value',
              'fdr_q_value',
              'fwer_p_value',
              'global_nes',
              'global_fdr_q_value',
              'rank_at_max',
              'leading_edge_num_hits',
              'leading_edge_frac_hits',
              'sample_set_frac_in_leading_edge',
              'null_set_frac_in_leading_edge',
              'details']
    FIELD_MAP = dict((v,k) for k,v in enumerate(FIELDS))

    @staticmethod
    def parse(filename):
        '''
        parses lines of the report file produced by SSEA and 
        generates dictionaries using the first line of the file
        containing the header fields
        '''
        fileh = open(filename, 'r')
        header_fields = fileh.next().strip().split('\t')
        details_ind = header_fields.index('details')
        for line in fileh:
            fields = line.strip().split('\t')
            fields[details_ind] = json.loads(fields[details_ind])
            yield dict(zip(header_fields, fields))
        fileh.close()

def create_detailed_report(name, desc, res, details_dir, config):
    '''
    Generate detailed report files including enrichment plots and 
    a tab-delimited text file showing the running ES score and rank
    of each member in the sample set

    name: string
    desc: string
    res: base.Result object
    details_dir: path to write files
    config: config.Config object
    
    returns dict containing files written
    '''
    d = {}
    if config.create_plots:
        # create enrichment plot
        res.plot(plot_conf_int=config.plot_conf_int,
                 conf_int=config.conf_int, fig=global_fig)    
        # save plots
        eplot_png = '%s.%s.eplot.png' % (name, res.sample_set.name)
        eplot_pdf = '%s.%s.eplot.pdf' % (name, res.sample_set.name)
        global_fig.savefig(os.path.join(details_dir, eplot_png))
        global_fig.savefig(os.path.join(details_dir, eplot_pdf))
        # create null distribution plot
        res.plot_null_distribution(fig=global_fig)
        nplot_png = '%s.%s.null.png' % (name, res.sample_set.name)
        nplot_pdf = '%s.%s.null.pdf' % (name, res.sample_set.name)
        global_fig.savefig(os.path.join(details_dir, nplot_png))        
        global_fig.savefig(os.path.join(details_dir, nplot_pdf))
        d.update({'eplot_png': eplot_png,
                  'nplot_png': nplot_png})
        d.update({'eplot_pdf': eplot_pdf,
                  'nplot_pdf': nplot_pdf})
    # write detailed report
    details_rows = res.get_details_table()
    details_tsv = '%s.%s.tsv' % (name, res.sample_set.name)
    fp = open(os.path.join(details_dir, details_tsv), 'w')
    for fields in details_rows:
        print >>fp, '\t'.join(map(str,fields))
    fp.close()
    d['tsv'] = details_tsv
    # render to html
    if config.create_html:
        fields = res.get_report_fields(name, desc)
        result_dict = dict(zip(Report.FIELDS, fields))
        details_html = '%s.%s.html' % (name, res.sample_set.name)
        t = env.get_template('details.html')
        fp = open(os.path.join(details_dir, details_html), 'w')
        print >>fp, t.render(res=result_dict, 
                             files=d,
                             details=details_rows)
        fp.close()
        d['html'] = details_html
    return d


def create_html_report(filename, rel_details_dir, config):
    def parse_report_as_dicts(filename):
        '''
        parses lines of the out.txt report file produced by SSEA and 
        generates dictionaries using the first line of the file
        containing the header fields
        '''
        fileh = open(filename, 'r')
        header_fields = fileh.next().strip().split('\t')
        details_ind = header_fields.index('details')
        for line in fileh:
            fields = line.strip().split('\t')
            fields[details_ind] = json.loads(fields[details_ind])
            yield dict(zip(header_fields, fields))
        fileh.close()

    # create directory for static web files (CSS, javascript, etc)
    web_dir = os.path.join(config.output_dir, 'web')
    if not os.path.exists(web_dir):
        logging.info("\tInstalling web files")
        shutil.copytree(SRC_WEB_PATH, web_dir)
    
    report_html = 'out.html'
    t = env.get_template('report.html')
    fp = open(os.path.join(config.output_dir, report_html), 'w')
    print >>fp, t.render(name=config.name,
                         details_dir=rel_details_dir,
                         results=parse_report_as_dicts(filename))
    fp.close()
