'''
Created on Oct 21, 2013

@author: mkiyer
'''
import os
import matplotlib.pyplot as plt



# def report_single_test(wobj, res, config):
#     # get report without details
#     reportdict = res.get_report_json()
#     reportdict['wname'] = wobj.name
#     reportdict['wdesc'] = wobj.desc
#     reportdict['details_json'] = None
#     reportdict['details_html'] = None
#     # only create detailed report if pvalue less than threshold
#     if res.qval <= config.fdr_qval_threshold:
#         eplot_png = '%s.%s.eplot.png' % (wobj.name, res.sample_set.name)
#         eplot_pdf = '%s.%s.eplot.pdf' % (wobj.name, res.sample_set.name)
#         null_png = '%s.%s.null.png' % (wobj.name, res.sample_set.name)
#         null_pdf = '%s.%s.null.pdf' % (wobj.name, res.sample_set.name)            
#         details_json = '%s.%s.json' % (wobj.name, res.sample_set.name)
#         details_html = '%s.%s.html' % (wobj.name, res.sample_set.name)
#         # create enrichment plot
#         fig = res.plot(plot_conf_int=config.plot_conf_int,
#                        conf_int=config.conf_int)
#         fig.savefig(os.path.join(config.output_dir, eplot_png))
#         fig.savefig(os.path.join(config.output_dir, eplot_pdf))
#         plt.close()
#         # create null distribution plot
#         fig = res.plot_null_distribution()
#         fig.savefig(os.path.join(config.output_dir, null_png))
#         fig.savefig(os.path.join(config.output_dir, null_pdf))
#         plt.close()
#         # create detailed report
#         detailsdict = res.get_details_json()
#         detailsdict.update({'eplot_png': eplot_png,
#                             'eplot_pdf': eplot_pdf,
#                             'null_png': null_png,
#                             'null_pdf': null_pdf})
#         fp = open(os.path.join(config.output_dir, details_json), 'w')
#         json.dump(detailsdict, fp)
#         fp.close()
#         # render to html
#         t = env.get_template('details.html')
#         fp = open(os.path.join(config.output_dir, details_html), 'w')
#         print >>fp, t.render(name=wobj.name, 
#                              desc=wobj.desc,
#                              report=reportdict,
#                              details=detailsdict)
#         fp.close()
#         # update report dict
#         reportdict['details_json'] = details_json
#         reportdict['details_html'] = details_html
#     return reportdict
#
# 
# def report_meta(reportdicts, config):
#     # write configuration
#     meta_report_html = '%s.html' % (config.name)
#     t = env.get_template('metareport.html')
#     fp = open(os.path.join(config.output_dir, meta_report_html), 'w')
#     print >>fp, t.render(name=config.name, reportdicts=reportdicts, 
#                          filter=None)
#     fp.close()
#     return meta_report_html
#
# def report_single_observation(wobj, results, config):
#     # create detailed reports
#     result_json_dicts = []
#     for res in results:
#         result_json_dicts.append(report_single_test(wobj, res, config))
#     # create observation report
#     report_json = '%s.json' % (wobj.name)
#     report_html = '%s.html' % (wobj.name)
#     jsondict = {'wname': wobj.name,
#                 'wdesc': wobj.desc,
#                 'report_json': report_json,
#                 'report_html': report_html,                
#                 'results': result_json_dicts}
#     fp = open(os.path.join(config.output_dir, report_json), 'w')              
#     json.dump(jsondict, fp, indent=2, sort_keys=True)
#     fp.close()
#     # render to html
#     t = env.get_template('report.html')
#     fp = open(os.path.join(config.output_dir, report_html), 'w')
#     print >>fp, t.render(data=jsondict)
#     fp.close()
#     return jsondict
# 
# def report_meta_sample_set(reportdicts, sample_sets, config):
#     d = {}
#     for sample_set in sample_sets:
#         sample_set_html = '%s.html' % (sample_set.name)
#         t = env.get_template('metareport.html')
#         fp = open(os.path.join(config.output_dir, sample_set_html), 'w')
#         print >>fp, t.render(name=config.name, 
#                              reportdicts=reportdicts,
#                              filter=sample_set.name)
#         fp.close()
#         d[sample_set.name] = sample_set_html
#     return d
# 
# 
# def report_config(config):
#     # config json file
#     config_json = 'config.json'
#     config_html = 'config.html'
#     jsondict = config.get_json()
#     fp = open(os.path.join(config.output_dir, config_json), 'w')
#     json.dump(jsondict, fp)
#     fp.close()
#     # render to html
#     t = env.get_template('config.html')
#     fp = open(os.path.join(config.output_dir, config_html), 'w')
#     print >>fp, t.render(config=jsondict)
#     fp.close()
#     jsondict['config_json'] = config_json
#     jsondict['config_html'] = config_html
#     return jsondict
# 
# def report_index(meta_report_html,
#                  sample_set_html_dict, 
#                  configdict,
#                  config):
#     t = env.get_template('index.html')
#     fp = open(os.path.join(config.output_dir, 'index.html'), 'w')
#     print >>fp, t.render(meta_report_html=meta_report_html,
#                          sample_set_html_dict=sample_set_html_dict,                         
#                          configdict=configdict)
#     fp.close()
#     
#     
# def report_main(config):
#     web_dir = os.path.join(config.output_dir, 'web')
#     if not os.path.exists(web_dir):
#         logging.info("\tInstalling web files")
#         shutil.copytree(src_web_path, web_dir)
#     logging.info("Running SSEA")
#     reportdicts = []
#     for wobj in WeightMatrix.parse_wmt(config.weight_matrix_file):
#         logging.info("\tName: %s (%s)" % (wobj.name, wobj.desc))
#         results = ssea_run(wobj.samples, wobj.weights, sample_sets, 
#                            weight_method_miss=config.weight_miss,
#                            weight_method_hit=config.weight_hit,
#                            perms=config.perms)
#         logging.debug("\t\twriting reports")
#         reportdicts.append(report_single_observation(wobj, results, config))
#     # write configuration report
#     configdict = report_config(config)
#     # write meta report
#     sample_set_html_dict = report_meta_sample_set(reportdicts, sample_sets, 
#                                                   config)
#     meta_report_html = report_meta(reportdicts, config)
#     # write index file
#     report_index(meta_report_html,
#                  sample_set_html_dict, 
#                  configdict,
#                  config)

