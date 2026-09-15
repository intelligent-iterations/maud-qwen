#!/usr/bin/env python3
"""Build the standalone MAUD article and figures from frozen result artifacts.

Requires matplotlib==3.9.4. No model, API call or inference is involved.
"""
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'writing/maud-blog'
OUTPUT = ROOT / 'reports/maud-fine-tuning.html'
ASSETS = ROOT / 'reports/maud-blog-assets'
TEST_PATH = ROOT / 'results/published/final-test-verification.json'
VALIDATION_PATH = ROOT / 'results/published/all-checkpoints-validation-summary.json'
SELECTION_SHA = '0b170b851921e905a6578b6b0a7e268e5633e419f1d47607be1807b20d6d328c'
WORD_LIMIT = 3000


class ReaderText(HTMLParser):
    """Count body copy, including closed details and image alt text, not assets."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'body':
            self.in_body = True
        if tag in ('script', 'style'):
            self.skip += 1
        if self.in_body and not self.skip and tag == 'img':
            self.parts.append(dict(attrs).get('alt', ''))

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip -= 1
        if tag == 'body':
            self.in_body = False

    def handle_data(self, data):
        if self.in_body and not self.skip:
            self.parts.append(data)

    @property
    def word_count(self):
        return len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", ' '.join(self.parts)))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_url(content, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(content).decode()


def validation_plot(rows, destination, mobile=False):
    paper, ink, muted, accent, line = '#f1ede5', '#282523', '#71695f', '#b83e31', '#d6cec1'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12 if mobile else 13,
                         'svg.fonttype':'path', 'svg.hashsalt':'maud-frozen-validation-v1',
                         'axes.edgecolor':line, 'text.color':ink, 'axes.labelcolor':muted,
                         'xtick.color':muted, 'ytick.color':muted})
    normal = [next(r for r in rows if r['checkpoint'] == f'Epoch {i}') for i in range(1,7)]
    values = [r['question_macro_f1'] for r in normal]
    weighted = next(r for r in rows if r['checkpoint'] == 'Weighted epoch 3')['question_macro_f1']
    fig, ax = plt.subplots(figsize=(5.2,4.8) if mobile else (11.2,5.5), facecolor=paper)
    fig.subplots_adjust(left=.14 if mobile else .09, right=.94, top=.72 if mobile else .74,
                        bottom=.18)
    ax.set_facecolor(paper)
    ax.set_xlim(.7,6.4);ax.set_ylim(.600,.668)
    ax.set_xticks(range(1,7));ax.set_xlabel('Normal training epoch',labelpad=12)
    ax.set_yticks([.60,.62,.64,.66]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=0,pad=9);ax.grid(axis='y',color=line,linewidth=.8)
    ax.plot(range(1,7),values,color=accent,linewidth=2.3,marker='o',markersize=6,zorder=3)
    ax.plot([2,3],[values[1],weighted],color=muted,linewidth=1.2,linestyle=(0,(4,3)),zorder=2)
    ax.scatter([3],[weighted],marker='D',s=45,facecolor=paper,edgecolor=muted,linewidth=1.5,zorder=4)
    ax.scatter([5],[values[4]],s=210,facecolor='none',edgecolor=accent,linewidth=1.1,zorder=4)
    for i,value in enumerate(values,1):
        if mobile and i not in (1,5,6):continue
        dy = -24 if i in (3,4,6) else 14
        ax.annotate(f'{value:.4f}',(i,value),xytext=(0,dy),textcoords='offset points',
                    ha='center',fontsize=10 if mobile else 12,color=ink)
    ax.annotate('Weighted E3\n0.6456' if mobile else 'Weighted third epoch · 0.6456',
                (3,weighted),xytext=(-4,30 if mobile else 32),textcoords='offset points',
                fontsize=10 if mobile else 12,ha='center',color=muted,
                arrowprops={'arrowstyle':'-','color':muted,'lw':.8,'shrinkA':4,'shrinkB':6})
    fig.text(.14 if mobile else .09,.925,'03 / CHECKPOINT SELECTION',fontsize=9 if mobile else 10,
             color=accent,weight='bold')
    fig.text(.14 if mobile else .09,.855,'Epoch 5 selected on validation.',fontsize=15 if mobile else 22,
             color=ink,weight='bold')
    fig.text(.14 if mobile else .09,.788,'Primary macro-F1 · 119 eligible tasks',fontsize=10 if mobile else 12,color=muted)
    fig.text(.14 if mobile else .09,.025,'The final test evaluated the selected checkpoint.',fontsize=9 if mobile else 11,color=muted)
    fig.savefig(destination,format='svg',metadata={'Date':None,'Creator':'MAUD experiment / Matplotlib 3.9.4'})
    plt.close(fig)
    # Matplotlib leaves trailing spaces in glyph paths; normalize generated text.
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')


def main():
    test, validation = read(TEST_PATH), read(VALIDATION_PATH)
    assert test['selection_ledger']['selection_sha256'] == SELECTION_SHA
    assert sha(ROOT/'results/published/final_selection.json') == SELECTION_SHA
    assert test['test_examples'] == 5283 and test['test_agreements'] == 23
    assert not test['test_used_for_model_selection'] and not test['additional_training_launched']
    systems = test['systems'];base=systems['base'];tuned=systems['epoch5']
    # Editorial counts in the fixed prose must continue to match the source artifacts.
    assert (base['rare']['correct'],tuned['rare']['correct'],base['rare']['n'],tuned['rare']['n']) == (66,57,154,154)
    assert (base['rare']['predictions'],tuned['rare']['predictions']) == (1366,104)
    assert (base['rare']['incorrect_predictions'],tuned['rare']['incorrect_predictions']) == (1300,47)
    assert all(s['truncated_rows'] == 0 for s in systems.values())
    ASSETS.mkdir(parents=True,exist_ok=True)
    for mobile in (False,True):
        validation_plot(validation['rows'],ASSETS/('validation-mobile.svg' if mobile else 'validation.svg'),mobile)
    data = {'split':'frozen agreement-held-out test','examples':5283,'agreements':23,
            'systems':{k:{'accuracy':v['scores']['accuracy'],'question_macro_f1':v['scores']['question_macro_f1'],
                          'rare_recall':v['rare']['recall']} for k,v in systems.items()}}
    evidence = {'title':'I fine-tuned a local AI on merger law','test':{
        k:test[k] for k in ('verified_utc','split','test_examples','test_agreements','primary_difference',
                            'primary_bootstrap','secondary_intervals','paired_errors','rare_training_support',
                            'test_used_for_model_selection','additional_training_launched')},
        'systems':{k:{'scores':{m:v['scores'][m] for m in ('n','question_macro_f1','accuracy','eligible_tasks_scored','missing_eligible_tasks')},
                      **{m:v[m] for m in ('rare','invalid_outputs','truncated_rows','prediction_sha256')}} for k,v in systems.items()},
        'validation':{'examples':validation['validation_examples'],'agreements':validation['validation_agreements'],
                      'eligible_tasks':validation['eligible_tasks'],'rows':validation['rows']},
        'selection':read(ROOT/'results/published/final_selection.json'),
        'mlflow':read(ROOT/'results/published/final-test-mlflow.json'),
        'source_sha256':{'final-test-verification.json':sha(TEST_PATH),'all-checkpoints-validation-summary.json':sha(VALIDATION_PATH),
                         'final_selection.json':SELECTION_SHA}}
    evidence_bytes = (json.dumps(evidence,indent=2,ensure_ascii=False)+'\n').encode()
    (ASSETS/'experiment-evidence.json').write_bytes(evidence_bytes)
    labels={'majority':'Training majority','base':'Untuned Qwen3-8B','epoch5':'QLoRA · epoch 5'}
    bars=[];table=[]
    for name in ('majority','base','epoch5'):
        s=systems[name];m=s['scores'];rare=s['rare'];css='tuned' if name=='epoch5' else name
        description={'majority':'per-question training counts','base':'same frozen NF4 base','epoch5':'selected on validation'}[name]
        bars.append(f'<div class="bar-row {css}" data-system="{name}"><div class="bar-label">{labels[name]}<small>{description}</small></div><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="--value:{m["question_macro_f1"]*100:.8f}%"></div></div><div class="bar-number">{m["question_macro_f1"]:.4f}</div></div>')
        precision='n/a' if rare['precision'] is None else f'{rare["precision"]:.2%}'
        table.append(f'<tr><th scope="row">{labels[name]}</th><td>{m["question_macro_f1"]:.4f}</td><td>{m["accuracy"]:.2%}</td><td>{rare["correct"]}/{rare["n"]}</td><td>{precision}</td><td>{s["invalid_outputs"]}</td></tr>')
    provenance={'Model revision':'b968826d9c46dd6066d109eabc6255188de91218',
                'Official data commit':'4640316078dcd370debb877f350c39a28b181ffe',
                'Model/evaluator code':'874c27a33e4f3982ed2241bbaf05aa612d5749b4',
                'Final operations code':'749f6b772cefab4de9e16f5d85af55bc8ff5635c',
                'Final selection SHA-256':SELECTION_SHA,
                'Adapter SHA-256':'a544b41c61871f08d86f07819531fd8dbf195a245c3b233c7adcfd56ada328b7',
                'Comparison MLflow run':evidence['mlflow']['comparison_mlflow_run_id']}
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td style="overflow-wrap:anywhere"><code>{html.escape(v)}</code></td></tr>' for k,v in provenance.items())
    weighted_rows = []
    for checkpoint, label in (('Epoch 3', 'Normal epoch 3'), ('Weighted epoch 3', 'Weighted epoch 3 · 3×')):
        record = next(r for r in validation['rows'] if r['checkpoint'] == checkpoint)
        weighted_rows.append(f'<tr><th scope="row">{label}</th><td>{record["question_macro_f1"]:.4f}</td><td>{record["rare_correct"]}/{record["rare_n"]}</td></tr>')
    pct=lambda name,key:f'{systems[name]["scores"][key]:.2%}'
    rare_pct=lambda name,key:f'{systems[name]["rare"][key]:.2%}'
    template=(SOURCE/'article.template.html').read_text()
    variables={'CSS':(SOURCE/'article.css').read_text(),'JS':(SOURCE/'article.js').read_text(),
               'DATA_JSON':json.dumps(data).replace('<','\\u003c'),'READING_MINUTES':'1',
               'BASE_ACCURACY':pct('base','accuracy'),'TUNED_ACCURACY':pct('epoch5','accuracy'),
               'BASE_F1':f'{base["scores"]["question_macro_f1"]:.4f}',
               'TUNED_F1':f'{tuned["scores"]["question_macro_f1"]:.4f}',
               'WEIGHTED_ROWS':''.join(weighted_rows),
               'F1_DELTA':f'{test["primary_difference"]:.3f}',
               'F1_CI':', '.join(f'{v:+.3f}' for v in test['primary_bootstrap']['difference_right_minus_left_ci95']),
               'RESULT_BARS':''.join(bars),'RESULT_ROWS':''.join(table),'PROVENANCE_ROWS':rows,
               'VALIDATION_CHART':data_url((ASSETS/'validation.svg').read_bytes(),'image/svg+xml'),
               'VALIDATION_MOBILE':data_url((ASSETS/'validation-mobile.svg').read_bytes(),'image/svg+xml'),
               'CHART_DOWNLOAD':data_url((ASSETS/'validation.svg').read_bytes(),'image/svg+xml'),
               'EVIDENCE_DOWNLOAD':data_url(evidence_bytes,'application/json'),
               'REPORT_DOWNLOAD':data_url((ROOT/'results/published/final-test-report.md').read_bytes(),'text/markdown'),
               'BASE_RARE_RECALL':rare_pct('base','recall'),'TUNED_RARE_RECALL':rare_pct('epoch5','recall'),
               'BASE_RARE_PRECISION':rare_pct('base','precision'),'TUNED_RARE_PRECISION':rare_pct('epoch5','precision')}
    rendered=re.sub(r'\{\{([A-Z0-9_]+)\}\}',lambda m:variables[m.group(1)],template)
    assert not re.search(r'\{\{[A-Z0-9_]+\}\}',rendered)
    reader = ReaderText()
    reader.feed(rendered)
    word_count = reader.word_count
    if word_count > WORD_LIMIT:
        raise ValueError(f'Article has {word_count} words; limit is {WORD_LIMIT}')
    variables['READING_MINUTES'] = str(math.ceil(word_count / 230))
    rendered=re.sub(r'\{\{([A-Z0-9_]+)\}\}',lambda m:variables[m.group(1)],template)
    OUTPUT.write_text(rendered)
    build={'article':str(OUTPUT.relative_to(ROOT)),'article_sha256':sha(OUTPUT),'figure_generator':'matplotlib '+matplotlib.__version__,
           'word_count':word_count,'word_limit':WORD_LIMIT,
           'word_count_scope':'All HTML body text, including closed details, tables and image alt text; excluding scripts/styles',
           'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),SOURCE/'article.template.html',SOURCE/'article.css',SOURCE/'article.js',TEST_PATH,VALIDATION_PATH)},
           'design_reference':'Original project engineering article; visual attribution retained in the article',
           'design_reference_sha256':'ad96f1e5bf87a50bd74b461fafc563807ade7be1ab7ec1ccdf39ad7bc92eb37c',
           'selected_epoch':5,'selection_sha256':SELECTION_SHA,'external_runtime_dependencies':[]}
    (ASSETS/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'reports/maud-blog.zip','w',zipfile.ZIP_DEFLATED) as archive:
        files={OUTPUT:'index.html',ROOT/'results/published/final-test-report.md':'evidence/final-test-report.md',
               **{p:'assets/'+p.name for p in sorted(ASSETS.iterdir()) if p.is_file()}}
        for path,name in files.items():
            item=zipfile.ZipInfo(name,date_time=(2026,9,12,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(item,path.read_bytes())
    print(json.dumps({'html':str(OUTPUT),'bytes':OUTPUT.stat().st_size,'word_count':word_count,'reading_minutes':variables['READING_MINUTES'],'zip':str(ROOT/'reports/maud-blog.zip')},indent=2))


if __name__=='__main__':
    main()
