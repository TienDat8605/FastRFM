"""Budgeted, sequence-disjoint exact-pullback screening on a frozen prior."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.acceleration.adjoint import integrate
from src.acceleration.colab_benchmark import BodyField, poses
from src.acceleration.distillation import endpoint_error, exact_pullback, normalized_pullback_error
from src.study.geometry import exp, log, sample
from src.study.data import balanced, sha256
from src.study.runtime import OfficialBody, completion_metrics
from src.study.acceleration import optimize_with_rollout

MASKS = {'legs':[0,1,3,4,6,7,9,10], 'torso':[2,5,8,15,16]}

class BudgetExpired(RuntimeError): pass

class Screen:
    def __init__(self,args):
        self.args=args
        self.out=Path(args.output_dir)
        self.out.mkdir(parents=True,exist_ok=True)
        if (self.out/'manifest.json').exists(): raise ValueError('Fresh output directory required')
        self.start=time.monotonic(); self.deadline=self.start+args.budget_hours*3600
        self.rows=[]; self.status={'phase':'setup','confirmation_complete':False}

    def save(self,name,value):
        path=self.out/name
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value,indent=2)); temp.replace(path)

    def guard(self, forecast=0):
        self.status['elapsed_seconds']=time.monotonic()-self.start
        self.save('status.json',self.status)
        if time.monotonic()+forecast>self.deadline: raise BudgetExpired('Insufficient budget for next complete work unit')

    def evaluate(self,model,name,ids,phase,iterations=20,repeats=1,weight=True):
        gt=self.validation[ids]; n=100 if name=='teacher100' else 50 if name=='teacher50' else 4
        solver='euler' if n>4 else 'midpoint'
        new=[]
        for mask,missing in MASKS.items():
            visible=torch.ones(21,device='cuda',dtype=torch.bool); visible[missing]=False
            obs=gt.repeat_interleave(2,0)
            objective=lambda r: log(r.transpose(-1,-2)@obs)[:,visible].norm(dim=-1).sum(-1).mean()
            for seed in (0,1):
                self.guard()
                gen=torch.Generator(device='cuda').manual_seed(seed)
                initial=sample((len(obs),21),'cuda',generator=gen)
                fn=lambda x: integrate(model,x,n,solver)
                optimize_with_rollout(initial,objective,fn,1,.1,.001*100/n if weight else 0)
                times=[]; peaks=[]; increments=[]; reserved=[]
                for rep in range(repeats):
                    self.guard(self.last_eval_seconds*1.2 if hasattr(self,'last_eval_seconds') else 0)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    base=torch.cuda.memory_allocated(); started=time.monotonic()
                    pred,history=optimize_with_rollout(initial,objective,fn,iterations,.1,.001*100/n if weight else 0)
                    torch.cuda.synchronize(); elapsed=time.monotonic()-started; self.last_eval_seconds=elapsed
                    times.append(elapsed); peaks.append(torch.cuda.max_memory_allocated()); increments.append(peaks[-1]-base); reserved.append(torch.cuda.max_memory_reserved())
                metrics,replaced=completion_metrics(self.body,pred,gt,visible,2)
                row={'phase':phase,'method':name,'mask':mask,'seed':seed,'iterations':iterations,'trajectory':weight,'seconds':times,'peak_allocated':peaks,'incremental_allocated':increments,'peak_reserved':reserved,**metrics}
                self.rows.append(row);new.append(row);self.save('measurements.json',self.rows)
                torch.save({'prediction':replaced.cpu(),'raw_prediction':pred.detach().cpu(),'ids':ids,'history':history},self.out/f'{phase}_{name}_{mask}_{seed}_{weight}.pt')
        return new

    def score(self,rows):
        return max(np.mean([v for r in rows if r['mask']==m for v in r['mpvpe_mm']])/self.baseline[m] for m in MASKS)

    def train(self,recipe,coefficient,seed):
        torch.manual_seed(seed); model=copy.deepcopy(self.teacher).requires_grad_(True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)
        name=f'{recipe}_{coefficient}_seed{seed}'; best=float('inf'); best_state=None; losses=[]
        train_started=time.monotonic()
        for update in range(1501):
            self.guard()
            if update%250==0:
                if update:
                    forecast=(time.monotonic()-train_started)/update*(1500-update)
                    self.status['remaining_recipe_forecast_seconds']=forecast
                    self.guard(forecast)
                rows=self.evaluate(model,name,self.dev,'development')
                score=self.score(rows)
                torch.save({'state_dict':model.state_dict(),'update':update,'score':score},self.out/f'{name}_update{update}.pt')
                if score<best:
                    best=score;best_state=copy.deepcopy(model.state_dict());torch.save({'state_dict':best_state,'update':update,'score':score},self.out/f'{name}_selected.pt')
                self.save(f'{name}_loss.json',losses)
            if update==1500: break
            idx=torch.randint(4096,(32,),device='cuda'); x=self.cache['source'][idx]
            flow=lambda z:integrate(model,z,4,'midpoint',False)[0]
            endpoint=flow(x);loss=endpoint_error(endpoint,self.cache['endpoint'][idx]);pull=loss.new_zeros(())
            if recipe!='endpoint':
                offset=0 if recipe=='random' else 2
                for k in range(offset,offset+2):
                    _,g=exact_pullback(flow,x,self.cache['covectors'][idx,k],self.cache['endpoint'][idx],True)
                    pull=pull+normalized_pullback_error(g,self.cache['pullbacks'][idx,k])/2
                loss=loss+coefficient*min(1,(update+1)/100)*pull
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite student loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append({'update':update+1,'loss':float(loss.detach()),'pullback':float(pull.detach())})
        model.load_state_dict(best_state)
        self.audit_model(model,name)
        return model,name,best

    def report(self,promoted=False,gates=None):
        lines=['# T4 exact-pullback screening', '', 'Promotion passed.' if promoted else 'No promoted candidate. Confirmation may be incomplete; see status.json.', '', '| Phase | Method | Mask | MPVPE mm | Visible rad | APD cm | Seconds/batch |', '|---|---|---|---:|---:|---:|---:|']
        keys=sorted(set((r['phase'],r['method'],r['mask']) for r in self.rows))
        for phase,method,mask in keys:
            rows=[r for r in self.rows if (r['phase'],r['method'],r['mask'])==(phase,method,mask)]
            values=[np.mean([v for r in rows for v in r[k]]) for k in ['mpvpe_mm','visible_rad','apd_cm','seconds']]
            lines.append(f'| {phase} | {method} | {mask} | '+ ' | '.join(f'{v:.4f}' for v in values)+' |')
        lines+=['', 'Short-trained frozen 3DPW prior; validation-only study. Exact sampled VJP agreement does not establish a covariance guarantee or mathematical exactness. No AMASS test poses used for training.']
        (self.out/'report_by_mask.md').write_text('\n'.join(lines)+'\n')
        if self.rows:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots()
            for phase,method,mask in keys:
                if phase!='confirmation':continue
                rows=[r for r in self.rows if (r['phase'],r['method'],r['mask'])==(phase,method,mask)]
                ax.scatter(np.mean([v for r in rows for v in r['seconds']]),np.mean([v for r in rows for v in r['mpvpe_mm']]),label=f'{method}/{mask}')
            ax.set(xlabel='Seconds per 64-hypothesis optimization batch',ylabel='Hidden MPVPE (mm)')
            if any(k[0]=='confirmation' for k in keys):ax.legend(fontsize=6)
            fig.savefig(self.out/'speed_quality.png',bbox_inches='tight');plt.close(fig)

    def audit_model(self,model,name):
        errors=[];cosines=[]
        for begin in range(0,1024,32):
            self.guard()
            x=self.audit['source'][begin:begin+32]
            flow=lambda z:integrate(model,z,4,'midpoint',False)[0]
            for k in range(4):
                _,g=exact_pullback(flow,x,self.audit['covectors'][begin:begin+32,k],self.audit['endpoint'][begin:begin+32])
                target=self.audit['pullbacks'][begin:begin+32,k]
                errors.append(float(normalized_pullback_error(g,target)))
                cosines.extend(torch.nn.functional.cosine_similarity(g.flatten(1),target.flatten(1)).detach().cpu().tolist())
        self.save(f'{name}_audit.json',{'normalized_mse':float(np.mean(errors)),'cosine':float(np.mean(cosines)),'samples':1024,'covectors':4})

    def execute(self):
        from posed_flow_colab_poc import PoseFlowMLP
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.set_num_threads(2);torch.manual_seed(42)
        self.trainposes,td=poses(self.args.data_path,'cuda');self.validation,vd=poses(self.args.validation_path,'cuda')
        if set(td['sequence'])&set(vd['sequence']): raise ValueError('Training/validation overlap')
        seq=np.unique(vd['sequence']);np.random.default_rng(42).shuffle(seq)
        if len(seq)!=12: raise ValueError('Expected 12 validation sequences')
        old=json.loads((Path(self.args.teacher_run)/'manifest.json').read_text())
        excluded=set(old['validation_ids'])
        def cohort(sequences,exclude):
            groups={str(s):[int(i) for i in np.flatnonzero((vd['sequence']==s)&(vd['frame']%10==0)) if i not in exclude] for s in sequences}
            return balanced(groups,32,np.random.default_rng(42))
        self.dev=cohort(seq[:8],set());self.confirm=cohort(seq[8:],excluded)
        teacher_path=Path(self.args.teacher_run)/'teacher.pt'
        network=PoseFlowMLP(num_joints=21,hidden_dim=512,riemannian=True).cuda()
        network.load_state_dict(torch.load(teacher_path,map_location='cuda',weights_only=False)['state_dict'])
        self.teacher=BodyField(network).requires_grad_(False).eval()
        self.body=OfficialBody(self.args.model_root,'cuda')
        self.save('manifest.json',{'teacher_sha256':sha256(teacher_path),'train_sha256':sha256(self.args.data_path),'validation_sha256':sha256(self.args.validation_path),'development_sequences':seq[:8].tolist(),'confirmation_sequences':seq[8:].tolist(),'development_ids':self.dev,'confirmation_ids':self.confirm,'excluded_pilot_ids':sorted(excluded),'source_hashes':{str(p):sha256(p) for p in [Path(__file__),Path('src/acceleration/distillation.py'),Path('src/acceleration/adjoint.py'),Path('posed_flow_colab_poc.py')]},'budget_hours':self.args.budget_hours,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'recipes':{'endpoint':[0],'random':[.001,.01,.1],'task':[.001,.01,.1]},'updates':1500,'batch_size':32,'confirmation_iterations':200,'hypotheses':2,'inference_seeds':[0,1],'promotion':{'speedup':8,'hidden_ratio':1.05,'visible_ratio':1.05,'apd_ratio':.9}})
        self.status['phase']='cache'; chunks=[]; started=time.monotonic()
        (self.out/'cache_shards').mkdir()
        for begin in range(0,5120,32):
            self.guard((time.monotonic()-started)/max(1,begin/32) if begin else 0)
            x=sample((32,21),'cuda'); flow=lambda z:integrate(self.teacher,z,100,'euler',False)[0]
            with torch.no_grad(): y=flow(x)
            vectors=[torch.randn(32,21,3,device='cuda') for _ in range(2)]
            target=self.trainposes[torch.randint(len(self.trainposes),(32,),device='cuda')]
            for missing in MASKS.values():
                visible=torch.ones(21,device='cuda',dtype=torch.bool);visible[missing]=False
                delta=torch.zeros(32,21,3,device='cuda',requires_grad=True)
                task=log((y@exp(delta)).transpose(-1,-2)@target)[:,visible].norm(dim=-1).sum()
                v,=torch.autograd.grad(task,delta)
                vectors.append(v/v.flatten(1).norm(dim=1).clamp_min(1e-8)[:,None,None])
            pulls=[exact_pullback(flow,x,v)[1].detach() for v in vectors]
            chunks.append({'source':x.cpu(),'endpoint':y.cpu(),'covectors':torch.stack(vectors,1).cpu(),'pullbacks':torch.stack(pulls,1).cpu()})
            torch.save(chunks[-1],self.out/'cache_shards'/f'{begin:05d}.pt')
            self.status['cache_completed_samples']=begin+32
            self.status['cache_remaining_forecast_seconds']=(time.monotonic()-started)/(begin+32)*(5120-begin-32)
        cache={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]};torch.save(cache,self.out/'targets.pt')
        self.save('cache_manifest.json',{'sha256':sha256(self.out/'targets.pt'),'training_sources':4096,'audit_sources':1024})
        self.cache={k:v[:4096].cuda() for k,v in cache.items()};self.audit={k:v[4096:].cuda() for k,v in cache.items()}
        self.status['phase']='screen'
        base=self.evaluate(self.teacher,'teacher100',self.dev,'development')
        self.baseline={m:np.mean([v for r in base if r['mask']==m for v in r['mpvpe_mm']]) for m in MASKS}
        raw=self.evaluate(self.teacher,'midpoint4',self.dev,'development')
        candidates=[];endpoint=None
        for recipe,coefficients in [('endpoint',[0]),('random',[.001,.01,.1]),('task',[.001,.01,.1])]:
            for coefficient in coefficients:
                started=time.monotonic();model,name,score=self.train(recipe,coefficient,0)
                if recipe=='endpoint':endpoint=(model,name)
                else:candidates.append((score,model,name,recipe,coefficient))
                self.status['last_recipe_seconds']=time.monotonic()-started
                self.guard()
        _,best,name,recipe,coefficient=min(candidates,key=lambda c:c[0])
        self.save('selection.json',{'method':name,'recipe':recipe,'coefficient':coefficient})
        for model,label in [(self.teacher,'teacher100'),(self.teacher,'midpoint4'),endpoint,(best,name)]:
            self.evaluate(model,label,self.dev,'zero_trajectory',weight=False)
        second,second_name,_=self.train(recipe,coefficient,1)
        self.status['phase']='confirmation'
        forecast=sum((100 if label=='teacher100' else 50 if label=='teacher50' else 8) for label in ['teacher100','teacher50','midpoint4',name,second_name])*4*200*2/100
        teacher_seconds=np.mean([t for r in base for t in r['seconds']])
        self.status['confirmation_forecast_seconds']=forecast*teacher_seconds/20
        self.guard(self.status['confirmation_forecast_seconds']*1.2)
        for model,label in [(self.teacher,'teacher100'),(self.teacher,'teacher50'),(self.teacher,'midpoint4'),(best,name),(second,second_name)]:
            self.evaluate(model,label,self.confirm,'confirmation',200,2)
        if sha256(teacher_path)!=json.loads((self.out/'manifest.json').read_text())['teacher_sha256']:
            raise RuntimeError('Teacher file changed')
        self.status['confirmation_complete']=True
        gates=[]
        for label in (name,second_name):
            for mask in MASKS:
                ref=[r for r in self.rows if r['phase']=='confirmation' and r['method']=='teacher100' and r['mask']==mask]
                test=[r for r in self.rows if r['phase']=='confirmation' and r['method']==label and r['mask']==mask]
                ratios={k:np.mean([v for r in test for v in r[k]])/max(1e-12,np.mean([v for r in ref for v in r[k]])) for k in ['mpvpe_mm','visible_rad','apd_cm']}
                ratios['speedup']=np.mean([v for r in ref for v in r['seconds']])/np.mean([v for r in test for v in r['seconds']])
                gates.append({'method':label,'mask':mask,**ratios,'pass':ratios['mpvpe_mm']<=1.05 and ratios['visible_rad']<=1.05 and ratios['apd_cm']>=.9 and ratios['speedup']>=8})
        promoted=all(g['pass'] for g in gates)
        self.save('promotion.json',{'promoted':promoted,'gates':gates})
        self.report(promoted,gates)
        self.status['phase']='complete';self.guard()


def run(args):
    screen=Screen(args)
    try:screen.execute()
    except BudgetExpired as error:
        screen.status.update(phase='budget_exhausted',reason=str(error));screen.save('status.json',screen.status);screen.report()
    except Exception as error:
        screen.status.update(phase='failed',reason=repr(error));screen.save('status.json',screen.status);raise
