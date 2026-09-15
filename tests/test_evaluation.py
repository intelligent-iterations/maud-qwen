import copy
import numpy as np
import pytest
from sklearn.metrics import f1_score

from maud_qwen.metrics import score, validate_predictions, bootstrap
from maud_qwen.prompt import parse_answer, encode, AnswerCollator
from maud_qwen.data import build_schema


def fixture():
    schema = {'q1':{'answers':['no','yes','unsupported'],'category':'one','question':'parent1','subquestion':'<NONE>'},
              'q2':{'answers':['no','yes'],'category':'two','question':'parent2','subquestion':'<NONE>'}}
    coverage = {q:{'eligible':True,'rare_labels':[1]} for q in schema}
    predictions=[]
    for i,(q,y,p,a) in enumerate([('q1',0,0,'a'),('q1',1,-1,'b'),('q1',1,0,'a'),('q2',1,1,'b')]):
        predictions.append({'example_id':str(i),'task_id':q,'label':y,'prediction':p,'agreement_id':a,'split':'validation','data_type':'main'})
    return schema,coverage,predictions


def test_invalid_is_false_negative_fixed_labels_and_equal_questions():
    schema,coverage,rows=fixture()
    result=score(rows,schema,coverage)
    q1=f1_score([0,1,1],[0,-1,0],labels=[0,1,2],average='macro',zero_division=0)
    q2=f1_score([1],[1],labels=[0,1],average='macro',zero_division=0)
    assert result['question_macro_f1']==pytest.approx((q1+q2)/2)
    assert result['accuracy']==0.5
    assert result['invalid_output_rate']==0.25
    assert result['rare_answer_accuracy']==pytest.approx(1/3)
    assert result['per_question']['q1']['per_label'][2]['f1']==0


@pytest.mark.parametrize('text,label',[(' A\n',0),('B',1),('C',-1),('a',-1),('A.',-1),('Answer: A',-1),('',-1),('AB',-1),('<think></think>A',-1)])
def test_strict_parser(text,label):
    assert parse_answer(text,2)==label


def test_prediction_integrity():
    schema,_,rows=fixture()
    validate_predictions(rows,rows,schema)
    for bad in [rows[:-1],rows+[rows[0]]]:
        with pytest.raises(ValueError):validate_predictions(rows,bad,schema)
    bad=copy.deepcopy(rows);bad[0]['label']=1
    with pytest.raises(ValueError):validate_predictions(rows,bad,schema)


def test_paired_cluster_bootstrap_deterministic_and_zero_for_identical_models():
    schema,coverage,rows=fixture()
    one=bootstrap(rows,rows,schema,coverage,replicates=50)
    two=bootstrap(rows,rows,schema,coverage,replicates=50)
    assert one==two
    assert one['difference_right_minus_left_ci95']==[0,0]
    better=copy.deepcopy(rows)
    for r in better:r['prediction']=r['label']
    result=bootstrap(rows,better,schema,coverage,replicates=200)
    assert result['difference_right_minus_left_ci95'][0]>=0


def test_multilabel_schema_mapping_is_not_collapsed():
    rows=[]
    for tid,sub in [('1.0','foo'),('1.1','bar')]:
        for label,answer in enumerate(['baz',sub+', baz']):
            rows.append({'id':tid,'subquestion':sub,'label':str(label),'answer':answer,'question':'Parent','text_type':'Type','category':'Cat'})
    result=build_schema(rows)
    assert len(result)==2 and result['1.0']['answers']==['<OTHER>','foo']
    bad=copy.deepcopy(rows);bad.append({**bad[0],'answer':'foo, baz'})
    with pytest.raises(AssertionError):build_schema(bad)


class TinyTokenizer:
    eos_token_id=99
    def apply_chat_template(self,*args,**kwargs):
        assert kwargs['enable_thinking'] is False
        return [10,20,30]
    def encode(self,text,**kwargs):return [ord(text)]


def test_prompt_and_padding_loss_mask_and_overflow():
    schema={'q':{'question':'Q','subquestion':'<NONE>','text_type':'T','answers':['x','y']}}
    row={'task_id':'q','label':0,'text':'passage','example_id':'id'}
    out=encode(TinyTokenizer(),row,schema,5)
    assert out['labels']==[-100,-100,-100,65,99]
    short={'input_ids':[5,6],'attention_mask':[1,1],'labels':[-100,6]}
    batch=AnswerCollator(0)([out,short])
    assert batch['labels'][1].tolist()==[-100,6,-100,-100,-100]
    assert batch['attention_mask'][1].tolist()==[1,1,0,0,0]
    with pytest.raises(ValueError):encode(TinyTokenizer(),row,schema,4)


def test_answer_position_loss_matches_full_masked_causal_loss():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from maud_qwen.train import answer_loss
    torch.manual_seed(42)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=100,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16))
    model.eval()
    batch={'input_ids':torch.tensor([[1,2,3,4,5]]),'attention_mask':torch.ones(1,5,dtype=torch.long),
           'labels':torch.tensor([[-100,-100,-100,4,5]])}
    full=model(**batch,use_cache=False).loss
    efficient=answer_loss(model,batch)
    assert torch.allclose(full,efficient,atol=1e-6)


def test_prepared_agreements_and_derived_examples_are_disjoint():
    from pathlib import Path
    from maud_qwen.common import read_json,read_jsonl,verify_freeze
    if not Path('manifests/freeze.json').exists():pytest.skip('Run prepare for integration audit')
    verify_freeze()
    sets=[]
    assignments=read_json('manifests/agreements.json')['assignments']
    for split in ['train','validation','test']:
        rows=read_jsonl(f'data/heldout/{split}.jsonl')
        sets.append({r['agreement_id'] for r in rows})
        assert all(assignments[r['agreement_id']]==split for r in rows)
        assert all(r['data_type'] in ['main','abridged'] for r in rows)
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert len(set.union(*sets))==152
